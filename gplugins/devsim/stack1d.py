# SPDX-License-Identifier: MIT
"""A 1D layer stack (e.g. poly gate / oxide / silicon body) built directly as a DEVSIM device.

:func:`create_1d_stack` lays the :class:`Layer1D` layers out along x, in order, from x = 0,
meshes them with ``devsim.create_1d_mesh`` (node spacing from :class:`Mesh1D`, geometrically
graded inside each layer), assigns each silicon layer's doping (:mod:`gplugins.devsim.doping`),
sets the material parameters from explicit :class:`~gplugins.devsim.materials.Constants`,
adds the material-pair interfaces and the two typed contacts, and solves the equilibrium
(0 V on every contact). It returns a :class:`~gplugins.devsim.get_solver.DDComponent`.

Names: the regions are the layer names, the interface between layers ``a`` and ``b`` is
``a___b``, and the contacts are the :class:`~gplugins.devsim.contacts.Contact` names. A 1D
DEVSIM mesh has exactly two contacts, one at each end; each is ``InterfaceRef(<end layer>,
None)``. DEVSIM works in cm: positions are cm, a charge is C/cm^2 and a capacitance F/cm^2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import devsim

from gplugins.devsim.contacts import (
    Contact,
    InterfaceRef,
    check_contact_region,
    delete_circuit,
)
from gplugins.devsim.doping import DopantKind, DopingProfile, assign_doping
from gplugins.devsim.materials import CONSTANTS, Constants

UM_TO_CM = 1e-4


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
    (a constant ratio between neighbouring spacings).
    """

    spacing_um: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        spacing = {}
        for name, pair in dict(self.spacing_um).items():
            h0, h1 = (float(v) for v in pair)
            if not (h0 > 0 and h1 > 0 and math.isfinite(h0) and math.isfinite(h1)):
                raise ValueError(f"Mesh1D spacing of {name!r} must be > 0, got {pair!r}")
            spacing[name] = (h0, h1)
        object.__setattr__(self, "spacing_um", spacing)


def graded_spacings(length: float, h0: float, h1: float) -> list[float]:
    """Spacings (same unit as the arguments) that fill ``length``, graded from ``h0`` to ``h1``.

    The fewest n intervals h_i = s h0 (h1/h0)^(i/(n-1)) whose unscaled sum reaches ``length``,
    scaled by s <= 1 to fill it exactly. So no spacing exceeds the requested one at its
    position, the neighbour ratio is constant, and only a layer thinner than min(h0, h1) gets a
    single interval (``length``).
    """

    def unscaled(n: int) -> list[float]:
        if n == 1:
            return [min(h0, h1)]
        return [h0 * (h1 / h0) ** (i / (n - 1)) for i in range(n)]

    n = 1
    while math.fsum(unscaled(n)) < length:
        n += 1
    hs = unscaled(n)
    s = length / math.fsum(hs)
    return [h * s for h in hs]


def _check_contacts(layers: Sequence[Layer1D], contacts: Sequence[Contact]) -> dict[str, int]:
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


def create_1d_stack(
    name: str,
    layers: Sequence[Layer1D],
    contacts: Sequence[Contact],
    mesh: Mesh1D,
    constants: Constants = CONSTANTS,
    *,
    relative_error: float = 1e-10,
    absolute_error: float = 1e10,
    maximum_iterations: int = 100,
    extended_precision: bool = True,
):
    """Build, and solve at equilibrium, the 1D DEVSIM device ``name`` (mesh ``<name>_mesh``).

    Args:
        layers: the stack in order along x (x = 0 at the first face of ``layers[0]``).
        contacts: two contacts, ``InterfaceRef(layers[0].name, None)`` and
            ``InterfaceRef(layers[-1].name, None)``; ``ohmic`` on silicon, ``gate`` on oxide.
            A :class:`~gplugins.devsim.contacts.LayerRef` or an interior end raises.
        mesh: the node spacing of every layer.
        constants: the material constants (never DEVSIM ``simple_physics``' defaults).
        relative_error, absolute_error, maximum_iterations: DEVSIM's DC tolerances, also used
            by the returned component's :meth:`~DDComponent.solve_dc`. A solve that does not
            converge raises :class:`~gplugins.devsim.convergence.ConvergenceError`.
        extended_precision: set DEVSIM's (global) extended solver/model/equation precision.

    Returns:
        A :class:`~gplugins.devsim.get_solver.DDComponent` holding the solved device.
    """
    from gplugins.devsim.get_solver import DDComponent

    layers = list(layers)
    contacts = list(contacts)
    if not layers:
        raise ValueError("a 1D stack needs at least one layer")
    names = [layer.name for layer in layers]
    if len(set(names)) != len(names):
        raise ValueError(f"layer names must be unique: {names}")
    missing = [n for n in names if n not in mesh.spacing_um]
    if missing:
        raise ValueError(f"Mesh1D has no spacing for layers {missing}")
    contact_end = _check_contacts(layers, contacts)

    # Boundaries (um) and node positions (um), each boundary shared by its two layers.
    bounds = [0.0]
    for layer in layers:
        bounds.append(bounds[-1] + layer.thickness_um)
    xs: list[float] = [0.0]
    boundary_node = [0]
    for i, layer in enumerate(layers):
        h0, h1 = mesh.spacing_um[layer.name]
        x = bounds[i]
        for h in graded_spacings(layer.thickness_um, h0, h1)[:-1]:
            x += h
            xs.append(x)
        xs.append(bounds[i + 1])
        boundary_node.append(len(xs) - 1)

    mesh_name = f"{name}_mesh"
    devsim.create_1d_mesh(mesh=mesh_name)
    tags = {node: f"b{i}" for i, node in enumerate(boundary_node)}
    for k, x in enumerate(xs):
        right = xs[k + 1] - x if k + 1 < len(xs) else x - xs[k - 1]
        left = x - xs[k - 1] if k > 0 else right
        kw = {"tag": tags[k]} if k in tags else {}
        devsim.add_1d_mesh_line(
            mesh=mesh_name, pos=x * UM_TO_CM, ps=right * UM_TO_CM, ns=left * UM_TO_CM, **kw
        )
    for i, layer in enumerate(layers):
        devsim.add_1d_region(
            mesh=mesh_name,
            material=layer.material,
            region=layer.name,
            tag1=f"b{i}",
            tag2=f"b{i + 1}",
        )
    for i in range(1, len(layers)):
        devsim.add_1d_interface(
            mesh=mesh_name, name=f"{layers[i - 1].name}___{layers[i].name}", tag=f"b{i}"
        )
    for contact in contacts:
        devsim.add_1d_contact(
            mesh=mesh_name, name=contact.name, tag=f"b{contact_end[contact.name]}", material="metal"
        )
    devsim.finalize_mesh(mesh=mesh_name)
    devsim.create_device(mesh=mesh_name, device=name)

    try:
        for i, layer in enumerate(layers):
            if layer.material != "silicon":
                continue
            if i + 1 < len(layers) and layers[i + 1].material == "oxide":
                surface, sign = bounds[i + 1], -1
            else:
                surface, sign = bounds[i], 1
            profile = {"donors" if layer.dopant == "donor" else "acceptors": layer.doping}
            assign_doping(
                name, layer.name, depth_axis="x", surface_um=surface, depth_sign=sign, **profile
            )
        return DDComponent.from_device(
            device=name,
            mesh=mesh_name,
            region_classes={layer.name: layer.material for layer in layers},
            contacts=contacts,
            constants=constants,
            rtol=relative_error,
            atol=absolute_error,
            max_iter=maximum_iterations,
            extended_precision=extended_precision,
        )
    except BaseException:
        if name in devsim.get_device_list():
            devsim.delete_device(device=name)
        devsim.delete_mesh(mesh=mesh_name)
        if any(c.circuit for c in contacts):
            delete_circuit()
        raise
