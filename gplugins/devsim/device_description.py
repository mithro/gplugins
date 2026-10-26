# SPDX-License-Identifier: MIT
"""The backend-neutral device description of gplugins.devsim (the backend seam).

**The seam.** A TCAD job is described once, as plain frozen data, and a *backend* turns that
description into a simulation. This module holds the description; ``gplugins.devsim.backends``
holds the backends. Today there is one, :mod:`gplugins.devsim.backends.devsim` (DEVSIM). A
``backends/charon.py`` (Charon) is to be added beside it so that both simulators solve the same
description (gf180-tcad spec §12.1, https://github.com/mithro/gf180-tcad, issue #44).

The description covers:

- **mesh**: a 1D stack's layers and node spacing (:class:`Layer1D`, :class:`Mesh1D`,
  :class:`Stack1D`, whose :meth:`Stack1D.node_positions_um` are the mesh nodes), or a 2D
  cross-section of a gdsfactory component that a mesher turns into a mesh
  (:class:`CrossSection2D`);
- **regions and materials**: a region's class, ``"silicon"``, ``"oxide"`` or ``None`` (no
  equations, e.g. a metal electrode), from its material name (:func:`material_class`);
- **contacts**: :class:`Contact` on a :class:`LayerRef` or an :class:`InterfaceRef`, with the
  kind-to-region rule (:func:`check_contact_region`) and the interface model of a pair of
  region classes (:func:`interface_kind`);
- **doping**: :mod:`gplugins.devsim.doping` profiles (pure functions of position), on
  :class:`Layer1D` with the depth origin of :meth:`Stack1D.doping_surfaces_um`, or the
  ``doping_info`` of a :class:`CrossSection2D`;
- **physical constants and models**: :class:`~gplugins.devsim.constants.Constants` (the one
  constants table; Boltzmann statistics, constant mobility, SRH);
- **solver settings**: :class:`SolverSettings`;
- **bias plan**: :class:`CVSweep`, a gate C-V sweep.

**The contract for a backend** (``gplugins.devsim.backends.<name>``):

1. This module, and everything it imports, imports no simulator or mesher (no ``devsim``, no
   ``meshwell``, no ``gplugins.devsim.backends``). ``test_device_description.py`` checks this
   in a fresh interpreter in which those imports fail.
2. A backend consumes the description read-only and adds nothing to it. Everything a result
   depends on (geometry, node positions, doping, constants, contacts, tolerances, biases) is in
   the description; what a backend chooses itself (equation assembly, discretisation, file
   formats, device and mesh names) stays in the backend.
3. The physics is the description's: a backend sets every constant from
   :class:`~gplugins.devsim.constants.Constants` and never uses a simulator default. A model a
   backend lacks is not approximated: the backend raises, and the job is marked not
   comparable for that model (spec §12.1).
4. A backend provides ``create_1d_device(stack: Stack1D)`` (a solved equilibrium device) and
   ``run_cv_sweep(device, sweep: CVSweep, **options)``; the DEVSIM backend's are
   :func:`gplugins.devsim.backends.devsim.create_1d_device` and
   :func:`gplugins.devsim.backends.devsim.run_cv_sweep`.
5. Names are shared: regions are the layer names, the interface between layers ``a`` and ``b``
   is ``a___b`` and contacts are the :class:`Contact` names, so results of two backends can be
   compared per region and per contact.

The public API predating the seam is unchanged: ``gplugins.devsim.contacts``,
``gplugins.devsim.materials`` and ``gplugins.devsim.stack1d`` re-export the names that moved
here, and ``gplugins.devsim.get_solver`` is :mod:`gplugins.devsim.backends.devsim`.

The package ``gplugins/devsim/__init__.py`` still imports DEVSIM eagerly, so in an image without
DEVSIM ``import gplugins.devsim.device_description`` fails through the package; making that
import lazy is left to the Charon backend's change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from gplugins.devsim.constants import CONSTANTS, Constants
from gplugins.devsim.doping import DopantKind, DopingProfile

if TYPE_CHECKING:
    from gplugins.devsim.doping import DopingLayerLevel

__all__ = [
    "OXIDE_MATERIALS",
    "SILICON_MATERIALS",
    "CVSweep",
    "Contact",
    "ContactKind",
    "CrossSection2D",
    "InterfaceKind",
    "InterfaceRef",
    "Layer1D",
    "LayerRef",
    "Mesh1D",
    "RegionClass",
    "SolverSettings",
    "Stack1D",
    "check_1d_contacts",
    "check_contact_region",
    "graded_spacings",
    "interface_kind",
    "material_class",
]

# --------------------------------------------------------------------------------------------
# Regions and materials
# --------------------------------------------------------------------------------------------
RegionClass = Literal["silicon", "oxide"]

#: Material names (as in gdsfactory layer stacks and :class:`Layer1D`) solved as silicon and
#: as oxide.
SILICON_MATERIALS = frozenset({"silicon", "si", "Si", "poly", "polysilicon", "poly_si"})
OXIDE_MATERIALS = frozenset({"oxide", "sio2", "SiO2", "Oxide", "gate_oxide"})


def material_class(material: str | None) -> RegionClass | None:
    """``"silicon"``, ``"oxide"`` or ``None`` for a material name."""
    if material in SILICON_MATERIALS:
        return "silicon"
    if material in OXIDE_MATERIALS:
        return "oxide"
    return None


# --------------------------------------------------------------------------------------------
# Contacts and interfaces
# --------------------------------------------------------------------------------------------
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


def interface_kind(class0: str | None, class1: str | None) -> InterfaceKind | None:
    """The interface model for a pair of region classes (``"silicon"``, ``"oxide"``, None)."""
    if class0 is None or class1 is None:
        return None
    if class0 == class1 == "silicon":
        return "drift_diffusion"
    return "potential"


def check_contact_region(contact: Contact, region: str, region_class: str | None) -> None:
    """Raise unless ``contact``'s kind fits the class of the region it is placed on."""
    want = {"ohmic": "silicon", "gate": "oxide"}[contact.kind]
    if region_class != want:
        raise ValueError(
            f"contact {contact.name!r} ({contact.kind}) needs a {want} region, but "
            f"{region!r} is {region_class or 'equation-free'}"
        )


# --------------------------------------------------------------------------------------------
# Solver settings and the bias plan
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SolverSettings:
    """DC solve tolerances: absolute and relative error, Newton iterations, precision."""

    absolute_error: float
    relative_error: float
    maximum_iterations: int
    extended_precision: bool = True


@dataclass(frozen=True)
class CVSweep:
    """A gate C-V sweep: ``gate`` over ``vg`` (V), every other contact at 0 V.

    ``method`` is ``"qs"`` (quasi-static, central difference with half step ``dv_qs`` V) or
    ``"ac"`` (small-signal at ``freq_hz``); see :func:`gplugins.devsim.cv.sweep_cv`.
    """

    gate: str
    vg: tuple[float, ...]
    method: Literal["qs", "ac"]
    freq_hz: float | None = None
    dv_qs: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(self, "vg", tuple(self.vg))
        if self.method not in ("qs", "ac"):
            raise ValueError(f"CVSweep.method must be 'qs' or 'ac', got {self.method!r}")


# --------------------------------------------------------------------------------------------
# 1D stack
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layer1D:
    """One layer of a 1D stack.

    ``doping`` (cm^-3, see :mod:`gplugins.devsim.doping`) and its ``dopant`` kind are required
    for silicon and not allowed for oxide. The profile's depth ``y_um`` is measured from the
    layer's face towards an adjacent oxide (into the layer), else from its first face.
    """

    name: str
    material: Literal["silicon", "oxide"]
    thickness_um: float
    doping: DopingProfile | None = None
    dopant: DopantKind | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or "___" in self.name:
            raise ValueError(f"Layer1D.name must be a non-empty name without '___': {self.name!r}")
        if self.material not in ("silicon", "oxide"):
            raise ValueError(f"layer {self.name!r}: material must be 'silicon' or 'oxide'")
        if not (self.thickness_um > 0 and math.isfinite(self.thickness_um)):
            raise ValueError(f"layer {self.name!r}: thickness_um must be > 0")
        if self.material == "oxide" and (self.doping is not None or self.dopant is not None):
            raise ValueError(f"oxide layer {self.name!r} cannot have doping")
        if self.material == "silicon":
            if not isinstance(self.doping, DopingProfile):
                raise ValueError(f"silicon layer {self.name!r} needs a DopingProfile")
            if self.dopant not in ("donor", "acceptor"):
                raise ValueError(
                    f"silicon layer {self.name!r}: dopant must be 'donor' or 'acceptor'"
                )


@dataclass(frozen=True)
class Mesh1D:
    """Node spacing (um) per layer: ``{layer: (at its first face, at its last face)}``.

    Inside a layer the spacing is graded geometrically from the first to the second value
    (a constant ratio between neighbouring spacings). ``max_ratio`` (> 1) bounds the ratio of
    any two neighbouring spacings, inside a layer (more, finer intervals are used) and across
    a layer boundary (:class:`Stack1D` raises if the requested spacings jump more).
    """

    spacing_um: Mapping[str, tuple[float, float]]
    max_ratio: float = 1.25

    def __post_init__(self) -> None:
        if not (self.max_ratio > 1 and math.isfinite(self.max_ratio)):
            raise ValueError(f"Mesh1D.max_ratio must be > 1, got {self.max_ratio!r}")
        spacing = {}
        for name, pair in dict(self.spacing_um).items():
            h0, h1 = (float(v) for v in pair)
            if not (h0 > 0 and h1 > 0 and math.isfinite(h0) and math.isfinite(h1)):
                raise ValueError(f"Mesh1D spacing of {name!r} must be > 0, got {pair!r}")
            spacing[name] = (h0, h1)
        object.__setattr__(self, "spacing_um", spacing)


def graded_spacings(
    length: float, h0: float, h1: float, max_ratio: float = 1.25
) -> list[float]:
    """Spacings (same unit as the arguments) that fill ``length``, graded from ``h0`` to ``h1``.

    The fewest n intervals h_i = s h0 (h1/h0)^(i/(n-1)) whose unscaled sum reaches ``length``
    and whose neighbour ratio (h1/h0)^(1/(n-1)) is within ``max_ratio``, scaled by s <= 1 to
    fill it exactly. So no spacing exceeds the requested one at its position, the neighbour
    ratio is constant and at most ``max_ratio``, and only a layer thinner than min(h0, h1)
    gets a single interval (``length``).
    """
    log_max = math.log(max_ratio)

    def unscaled(n: int) -> list[float]:
        if n == 1:
            return [min(h0, h1)]
        return [h0 * (h1 / h0) ** (i / (n - 1)) for i in range(n)]

    def too_steep(n: int) -> bool:
        return n > 1 and abs(math.log(h1 / h0)) / (n - 1) > log_max * (1 + 1e-12)

    n = 1
    while math.fsum(unscaled(n)) < length or too_steep(n):
        n += 1
    hs = unscaled(n)
    s = length / math.fsum(hs)
    return [h * s for h in hs]


def check_1d_contacts(
    layers: Sequence[Layer1D], contacts: Sequence[Contact]
) -> dict[str, int]:
    """``{contact: 0 or len(layers)}`` (the boundary index of its end); raises if invalid."""
    if len(contacts) != 2:
        raise ValueError(
            f"a 1D stack has exactly two contacts (one at each end), got {len(contacts)}"
        )
    if contacts[0].name == contacts[1].name:
        raise ValueError(f"contact names must differ, got {contacts[0].name!r} twice")
    ends = {layers[0].name: 0, layers[-1].name: len(layers)}
    if len(layers) == 1:
        raise ValueError(
            "a one-layer stack has the same InterfaceRef(<layer>, None) at both ends; split "
            "the layer in two"
        )
    out = {}
    for contact in contacts:
        loc = contact.location
        if not isinstance(loc, InterfaceRef):
            raise ValueError(
                f"contact {contact.name!r}: a 1D stack has no GDS layers; use "
                f"InterfaceRef(<end layer>, None), not {loc!r}"
            )
        if loc.b is not None or loc.a not in ends:
            raise ValueError(
                f"contact {contact.name!r}: {loc!r} is not a free end of the stack; a 1D "
                f"contact is InterfaceRef(<layer>, None) on {sorted(ends)}"
            )
        index = 0 if loc.a == layers[0].name else len(layers) - 1
        check_contact_region(contact, loc.a, layers[index].material)
        out[contact.name] = ends[loc.a]
    if len(set(out.values())) != 2:
        raise ValueError("the two contacts must be at the two different ends of the stack")
    return out


@dataclass(frozen=True)
class Stack1D:
    """A 1D device: ``layers`` along x from x = 0, two end ``contacts``, ``mesh``, physics.

    Validated on construction (layers, names, a spacing per layer, the contacts, the spacing
    jumps across layer boundaries), so a backend receives only a valid stack. Positions are
    um; a backend converts to its own unit.
    """

    name: str
    layers: tuple[Layer1D, ...]
    contacts: tuple[Contact, ...]
    mesh: Mesh1D
    constants: Constants = CONSTANTS
    solver: SolverSettings = field(
        default_factory=lambda: SolverSettings(
            absolute_error=1e10, relative_error=1e-10, maximum_iterations=100
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "contacts", tuple(self.contacts))
        if not self.layers:
            raise ValueError("a 1D stack needs at least one layer")
        names = [layer.name for layer in self.layers]
        if len(set(names)) != len(names):
            raise ValueError(f"layer names must be unique: {names}")
        missing = [n for n in names if n not in self.mesh.spacing_um]
        if missing:
            raise ValueError(f"Mesh1D has no spacing for layers {missing}")
        self.contact_ends()
        self.node_positions_um()

    @property
    def region_classes(self) -> dict[str, RegionClass]:
        """``{layer name: its material}`` (every layer of a 1D stack is silicon or oxide)."""
        return {layer.name: layer.material for layer in self.layers}

    @property
    def interfaces(self) -> list[str]:
        """The interface names ``a___b`` between neighbouring layers, in order."""
        return [f"{a.name}___{b.name}" for a, b in zip(self.layers, self.layers[1:])]

    def contact_ends(self) -> dict[str, int]:
        """``{contact: boundary index}`` (0 or ``len(layers)``); raises if invalid."""
        return check_1d_contacts(self.layers, self.contacts)

    def boundaries_um(self) -> list[float]:
        """The ``len(layers) + 1`` layer boundaries (um), from 0."""
        bounds = [0.0]
        for layer in self.layers:
            bounds.append(bounds[-1] + layer.thickness_um)
        return bounds

    def node_positions_um(self) -> tuple[list[float], list[int]]:
        """The mesh nodes (um) and, per boundary, the index of its node.

        Each layer's nodes are :func:`graded_spacings` of its :class:`Mesh1D` spacing; each
        boundary node is shared by its two layers. Raises if a spacing jumps across a layer
        boundary by more than ``mesh.max_ratio``.
        """
        layers, mesh = self.layers, self.mesh
        bounds = self.boundaries_um()
        spacings = [
            graded_spacings(layer.thickness_um, *mesh.spacing_um[layer.name], mesh.max_ratio)
            for layer in layers
        ]
        for i in range(1, len(layers)):
            a, b = spacings[i - 1][-1], spacings[i][0]
            if max(a / b, b / a) > mesh.max_ratio * (1 + 1e-12):
                raise ValueError(
                    f"the spacing jumps from {a!r} um ({layers[i - 1].name}) to {b!r} um "
                    f"({layers[i].name}), a ratio above Mesh1D.max_ratio = {mesh.max_ratio!r}"
                )
        xs: list[float] = [0.0]
        boundary_node = [0]
        for i in range(len(layers)):
            x = bounds[i]
            for h in spacings[i][:-1]:
                x += h
                xs.append(x)
            xs.append(bounds[i + 1])
            boundary_node.append(len(xs) - 1)
        return xs, boundary_node

    def doping_surfaces_um(self) -> dict[str, tuple[float, int]]:
        """``{silicon layer: (surface x in um, depth sign)}``, the origin of its doping depth.

        The depth is measured from the face towards the next layer if that is oxide (sign
        -1, into the layer), else from the layer's first face (sign +1).
        """
        layers = self.layers
        bounds = self.boundaries_um()
        out = {}
        for i, layer in enumerate(layers):
            if layer.material != "silicon":
                continue
            if i + 1 < len(layers) and layers[i + 1].material == "oxide":
                out[layer.name] = (bounds[i + 1], -1)
            else:
                out[layer.name] = (bounds[i], 1)
        return out


# --------------------------------------------------------------------------------------------
# 2D cross-section
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossSection2D:
    """A 2D device: the cross-section of a gdsfactory component along ``xsection_bounds``.

    The fields are the geometry, doping, contact and mesh inputs of
    :class:`gplugins.devsim.backends.devsim.DDComponent` (see there); the mesher turns them
    into regions named by layer and interfaces ``a___b``. ``mesh_scaling_factor`` converts um
    to the backend's length unit (1e-4: cm).
    """

    component: Any
    xsection_bounds: tuple[tuple[float, float], tuple[float, float]]
    full_layer_stack: Any
    physical_layer_stack: Any
    doping_info: Mapping[str, DopingLayerLevel]
    contacts: tuple[Contact, ...]
    constants: Constants = CONSTANTS
    solver: SolverSettings = field(
        default_factory=lambda: SolverSettings(
            absolute_error=1e12, relative_error=1e-12, maximum_iterations=200
        )
    )
    resolutions: Mapping[str, dict] | None = None
    mesh_scaling_factor: float = 1e-4
    background_tag: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contacts", tuple(self.contacts))
