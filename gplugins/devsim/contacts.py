# SPDX-License-Identifier: MIT
"""Typed contacts and material-pair interfaces for gplugins.devsim.

A :class:`Contact` names a boundary of the device and its boundary condition:

- ``location`` is a :class:`LayerRef` (2D only: the region drawn on a GDS contact layer; the
  contact is its boundary with the one region it touches) or an :class:`InterfaceRef`
  (the mesh boundary between two regions, ``a___b``, or the outer boundary of one region,
  ``a___None``). In ``InterfaceRef(a, b)`` the contact is placed on region ``b``, the device
  side, and ``a`` is the electrode side; ``InterfaceRef(a, None)`` places it on ``a``. In 1D a
  contact is always ``InterfaceRef(<layer>, None)``, the free end of that layer.
- ``kind="ohmic"``: on silicon (a poly-silicon gate is silicon), charge-neutral equilibrium
  carriers and ``Potential = bias + Vt ln(n/n_i)`` (DEVSIM ``simple_physics``).
- ``kind="gate"``: a metal electrode on oxide, ``Potential = bias - workfunction_offset_v``.
  With offset 0 the metal's Fermi level is at the intrinsic level, so on a body of bulk
  potential psi_b the flat band is at ``Vg = psi_b + workfunction_offset_v``. DEVSIM's
  ``CreateOxideContact`` has no such term.
- ``circuit=True`` attaches the contact to a circuit node ``<name>_bias`` driven by a voltage
  source ``V_<name>`` (DC and small-signal AC); otherwise ``<name>_bias`` is a device
  parameter. DEVSIM's circuit is global: node and source names are not per device.

Interfaces are chosen by the material pair (:func:`interface_kind`): silicon/oxide,
poly/oxide and oxide/oxide get a continuous potential; silicon/silicon gets drift-diffusion
continuity (potential, electrons and holes); an interface with an equation-free region
(``None``, e.g. a metal electrode) gets nothing.

The gplugins <= 0.8 ``contact_info`` dict is still accepted through
:func:`contacts_from_contact_info` (deprecated).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import devsim
from devsim.python_packages import model_create, simple_physics

ContactKind = Literal["ohmic", "gate"]
InterfaceKind = Literal["potential", "drift_diffusion"]

_KINDS = ("ohmic", "gate")


@dataclass(frozen=True)
class LayerRef:
    """A GDS contact layer ``(layer, datatype)`` (2D only)."""

    layer: tuple[int, int]

    def __post_init__(self) -> None:
        layer = tuple(self.layer)
        if len(layer) != 2 or not all(isinstance(v, int) for v in layer):
            raise ValueError(f"LayerRef.layer must be (layer, datatype) ints, got {self.layer!r}")
        object.__setattr__(self, "layer", layer)


@dataclass(frozen=True)
class InterfaceRef:
    """The boundary between regions ``a`` and ``b``; ``b=None`` is ``a``'s outer boundary."""

    a: str
    b: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.a, str) or not self.a:
            raise ValueError(f"InterfaceRef.a must be a region name, got {self.a!r}")
        if self.b is not None and (not isinstance(self.b, str) or not self.b):
            raise ValueError(f"InterfaceRef.b must be a region name or None, got {self.b!r}")

    @property
    def gmsh_names(self) -> tuple[str, ...]:
        """The mesh physical-group names this boundary may have (meshwell's ``a___b``)."""
        if self.b is None:
            return (f"{self.a}___None",)
        return (f"{self.a}___{self.b}", f"{self.b}___{self.a}")


@dataclass(frozen=True)
class Contact:
    """A device contact: a named boundary with an ohmic or metal-gate boundary condition."""

    name: str
    location: LayerRef | InterfaceRef
    kind: ContactKind
    workfunction_offset_v: float = 0.0
    circuit: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"Contact.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.location, LayerRef | InterfaceRef):
            raise TypeError(
                f"Contact.location must be a LayerRef or InterfaceRef, got {self.location!r}"
            )
        if self.kind not in _KINDS:
            raise ValueError(f"Contact.kind must be one of {_KINDS}, got {self.kind!r}")
        if self.kind == "ohmic" and self.workfunction_offset_v != 0.0:
            raise ValueError(
                f"contact {self.name!r}: workfunction_offset_v is a metal-gate (kind='gate') "
                "property; an ohmic contact has none"
            )


def bias_name(contact: str) -> str:
    """The bias parameter / circuit node of a contact (as ``simple_physics``)."""
    return simple_physics.GetContactBiasName(contact)


def source_name(contact: str) -> str:
    """The voltage source driving a ``circuit=True`` contact."""
    return f"V_{contact}"


def delete_circuit() -> None:
    """Delete DEVSIM's (global) circuit, if any.

    DEVSIM 2.11.1's ``delete_circuit`` deletes the circuit and then raises an empty
    ``devsim.error``, also when there is no circuit; that empty error is swallowed, any other
    is raised. Do not probe for a circuit with ``get_circuit_node_list``: with no circuit it
    leaves DEVSIM's circuit in a state that makes every later solve abort
    (``CIRCUIT_UNEXPECTED``).
    """
    try:
        devsim.delete_circuit()
    except devsim.error as e:
        if str(e):
            raise


def gds_layer(layer) -> tuple[int, int] | None:
    """``(layer, datatype)`` of a tuple, a gdsfactory layer enum or a ``LogicalLayer``.

    ``None`` for anything else (e.g. a ``DerivedLayer``).
    """
    for _ in range(3):
        if hasattr(layer, "datatype") and hasattr(layer, "layer"):
            return (int(layer.layer), int(layer.datatype))
        if isinstance(layer, tuple | list) and len(layer) == 2:
            try:
                return (int(layer[0]), int(layer[1]))
            except (TypeError, ValueError):
                return None
        if not hasattr(layer, "layer"):
            return None
        layer = layer.layer
    return None


# --------------------------------------------------------------------------------------------
# The gplugins <= 0.8 contact_info dict
# --------------------------------------------------------------------------------------------
def contacts_from_contact_info(contact_info: Mapping[str, Mapping], full_layer_stack):
    """``(contacts, levels)`` for a gplugins <= 0.8 ``contact_info`` dict. Deprecated.

    Each entry ``name: {"gds_layer", "physical_layerlevel_to_replace",
    "physical_layerlevel_to_contact"}`` becomes ``Contact(name, InterfaceRef(name,
    <to_contact>), "ohmic")`` and a ``LayerLevel`` named ``name`` on ``gds_layer`` with the
    geometry and material of ``full_layer_stack.layers[<to_replace>]``, which the caller adds
    to the simulated stack (as gplugins <= 0.8 did).
    """
    from gdsfactory.technology import LayerLevel

    warnings.warn(
        "contact_info dicts are deprecated; pass contacts=[Contact(name, LayerRef(gds_layer) "
        "or InterfaceRef(a, b), kind)] (gplugins.devsim.contacts) instead",
        DeprecationWarning,
        stacklevel=3,
    )
    contacts, levels = [], {}
    for name, info in contact_info.items():
        level = full_layer_stack.layers[info["physical_layerlevel_to_replace"]]
        levels[name] = LayerLevel(
            layer=info["gds_layer"],
            thickness=level.thickness,
            zmin=level.zmin,
            material=level.material,
            sidewall_angle=level.sidewall_angle,
            info=level.info,
        )
        contacts.append(
            Contact(name, InterfaceRef(name, info["physical_layerlevel_to_contact"]), "ohmic")
        )
    return contacts, levels


# --------------------------------------------------------------------------------------------
# Interfaces
# --------------------------------------------------------------------------------------------
def interface_kind(class0: str | None, class1: str | None) -> InterfaceKind | None:
    """The interface model for a pair of region classes (``"silicon"``, ``"oxide"``, None)."""
    if class0 is None or class1 is None:
        return None
    if class0 == class1 == "silicon":
        return "drift_diffusion"
    return "potential"


def create_interface_potential(device: str, interface: str) -> None:
    """Continuous ``Potential`` across ``interface``."""
    simple_physics.CreateSiliconOxideInterface(device, interface)


def create_interface_carriers(device: str, interface: str) -> None:
    """Continuous electrons and holes across a silicon/silicon ``interface``.

    With :func:`create_interface_potential` this is ``simple_physics``'s
    ``CreateSiliconSiliconInterface``.
    """
    for carrier, eq in (
        ("Electrons", "ElectronContinuityEquation"),
        ("Holes", "HoleContinuityEquation"),
    ):
        model = model_create.CreateContinuousInterfaceModel(device, interface, carrier)
        devsim.interface_equation(
            device=device,
            interface=interface,
            name=eq,
            interface_model=model,
            type="continuous",
        )


# --------------------------------------------------------------------------------------------
# Contact equations
# --------------------------------------------------------------------------------------------
def check_contact_region(contact: Contact, region: str, region_class: str | None) -> None:
    """Raise unless ``contact``'s kind fits the class of the region it is placed on."""
    want = {"ohmic": "silicon", "gate": "oxide"}[contact.kind]
    if region_class != want:
        raise ValueError(
            f"contact {contact.name!r} ({contact.kind}) needs a {want} region, but "
            f"{region!r} is {region_class or 'equation-free'}"
        )


def create_contact_potential_only(
    device: str, region: str, contact: Contact, region_class: str | None
) -> None:
    """The ``PotentialEquation`` boundary condition of ``contact`` on ``region``."""
    check_contact_region(contact, region, region_class)
    if contact.kind == "ohmic":
        simple_physics.CreateSiliconPotentialOnlyContact(
            device, region, contact.name, contact.circuit
        )
        return
    _create_gate_contact(device, region, contact)


def create_contact_drift_diffusion(
    device: str, region: str, contact: Contact, region_class: str | None
) -> None:
    """The carrier boundary conditions of an ohmic ``contact`` (a gate contact has none)."""
    check_contact_region(contact, region, region_class)
    if contact.kind == "ohmic":
        simple_physics.CreateSiliconDriftDiffusionAtContact(
            device, region, contact.name, contact.circuit
        )


def _create_gate_contact(device: str, region: str, contact: Contact) -> None:
    """Metal gate on oxide: ``Potential - bias + workfunction_offset_v = 0``."""
    bias = bias_name(contact.name)
    model = simple_physics.GetContactNodeModelName(contact.name)
    offset = float(contact.workfunction_offset_v)
    model_create.CreateContactNodeModel(
        device, contact.name, model, f"Potential - {bias} + ({offset!r})"
    )
    model_create.CreateContactNodeModel(device, contact.name, f"{model}:Potential", "1")
    if contact.circuit:
        model_create.CreateContactNodeModel(device, contact.name, f"{model}:{bias}", "-1")
    charge = simple_physics.contactcharge_edge
    if not model_create.InEdgeModelList(device, region, charge):
        model_create.CreateEdgeModel(device, region, charge, "Permittivity*ElectricField")
        model_create.CreateEdgeModelDerivatives(
            device, region, charge, "Permittivity*ElectricField", "Potential"
        )
    kw = {"circuit_node": bias} if contact.circuit else {}
    devsim.contact_equation(
        device=device,
        contact=contact.name,
        name="PotentialEquation",
        node_model=model,
        edge_charge_model=charge,
        **kw,
    )
