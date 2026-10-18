# SPDX-License-Identifier: MIT
"""The meshwell 2.5 mesher behind DDComponent, re-verified against the gplugins v0.8.0 record.

The reference is the "once worked" record of gf180-tcad (https://github.com/mithro/gf180-tcad),
``results/plugin-history/v0.8.0.json`` and ``v0.8.0-mesh0p5.json`` at commit
a781119a4d13001fdc52b90e2788fff203220e2e: gplugins v0.8.0 on its own Sept-2023 stack (devsim
2.6.3, gmsh 4.11.1, gdsfactory 7.6.1), driven by gf180-tcad
``experiments/plugin-history/checks_v080.py``. The PN slab here is that driver's PN slab
(the ``__main__`` block of v0.8.0 ``gplugins/devsim/get_solver.py``), ported to gdsfactory 9:
``.copy()`` before ``trim``, ``gf.boolean(..., layer1=, layer2=)`` and the ``gpdk`` layer map
and stack. Its ``resolutions`` are passed in the v0.8.0 form and converted by
``meshing.legacy_resolutions``. The mesh is meshwell's OCC pipeline, not v0.8.0's pygmsh one,
so node counts differ; they are reported (``record_property``), not asserted.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import devsim
import pytest

V080 = "gf180-tcad results/plugin-history/v0.8.0.json @ a781119a4d13001fdc52b90e2788fff203220e2e"
# --- v0.8.0 record values (v0.8.0.json "ddcomponent"; mesh x0.5 from v0.8.0-mesh0p5.json) ---
V080_REGIONS = {"Aluminum": ["anode", "cathode"], "si": ["slab90", "core"]}
V080_DEVSIM_REGIONS = ["anode", "cathode", "core", "slab90"]
V080_CONTACTS = ["anode", "cathode"]
V080_INTERFACES = ["slab90___anode", "slab90___cathode"]
V080_DV_EQ = {1.0: 0.9586540658880733, 0.5: 0.9586540658880742}  # anode - cathode, V
V080_DV_BIAS = {  # anode - cathode at cathode bias 0.1 / 0.2 V
    1.0: (0.8586540658880735, 0.7586540658880733),
    0.5: (0.8586540658880739, 0.7586540658880736),
}
V080_NET_DOPING = (-1.1e18, 1.1e18)  # cm^-3, both mesh scales
V080_DEPLETION_UM = {1.0: 0.0386279954111326, 0.5: 0.06882806455074479}  # 2 and 8 nodes
V080_NODE_COUNT = {1.0: 3523, 0.5: 11326}

PN_VOLTAGES = (0.1, 0.2)
#: Task 3 (gf180-tcad): every PN-slab current of the v0.8.0 record is below this, A/cm.
CURRENT_NOISE_FLOOR = 1e-26
#: The generic stack's interface that v0.8.0's mesher never tagged (see the currents test).
RIB_SLAB_INTERFACE = "core___slab90"

# simple_physics (DEVSIM) constants that DDComponent.set_parameters uses at 300 K.
_VT = 1.3806503e-23 * 300 / 1.6e-19
_NI = 1e10


# --------------------------------------------------------------------------------------------
# The PN slab (gf180-tcad checks_v080.pn_slab_inputs, on gdsfactory 9)
# --------------------------------------------------------------------------------------------
def pn_slab_inputs(mesh_scale: float) -> dict:
    import gdsfactory as gf
    from gdsfactory.gpdk import LAYER, LAYER_STACK, PDK
    from gdsfactory.technology import LayerStack

    from gplugins.devsim.doping import get_doping_info_generic

    PDK.activate()
    waveguide = gf.Component()
    waveguide.add_ref(
        gf.functions.trim(
            component=gf.components.straight_pn(length=10, taper=None).copy(),
            domain=[[3, -4], [3, 4], [5, 4], [5, -4]],
        )
    )
    physical_layer_stack = LayerStack(
        layers={k: LAYER_STACK.layers[k].model_copy() for k in ("slab90", "core")}
    )
    anode_layer, cathode_layer = (105, 0), (106, 0)
    via = waveguide.extract([LAYER.VIAC])
    npp = waveguide.extract([LAYER.NPP])
    ppp = waveguide.extract([LAYER.PPP])
    waveguide.add_ref(
        gf.boolean(
            via, npp, "and", layer=anode_layer, layer1=LAYER.VIAC, layer2=LAYER.NPP
        )
    )
    waveguide.add_ref(
        gf.boolean(
            via, ppp, "and", layer=cathode_layer, layer1=LAYER.VIAC, layer2=LAYER.PPP
        )
    )
    contact_info = {
        name: {
            "gds_layer": layer,
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        }
        for name, layer in (("anode", anode_layer), ("cathode", cathode_layer))
    }
    resolutions = {
        k: {"resolution": 0.02 * mesh_scale, "distance": 1 * mesh_scale}
        for k in ("core", "slab90")
    }
    return {
        "component": waveguide,
        "xsection_bounds": [(4, -4), (4, 4)],
        "full_layer_stack": LAYER_STACK,
        "physical_layer_stack": physical_layer_stack,
        "doping_info": get_doping_info_generic(),
        "contact_info": contact_info,
        "resolutions": resolutions,
        "mesh_scaling_factor": 1e-4,
        "background_tag": None,
    }


def write_pn_slab_mesh(path: str) -> None:
    """Mesh the PN slab (mesh scale 1) with ``meshing.mesh_xsection`` only, to ``path``."""
    from gdsfactory.technology import LayerLevel

    from gplugins.devsim.get_simulation import xsection_entities
    from gplugins.devsim.meshing import legacy_resolutions, mesh_xsection

    kw = pn_slab_inputs(1.0)
    stack = kw["physical_layer_stack"]
    via = kw["full_layer_stack"].layers["via_contact"]
    for name, info in kw["contact_info"].items():
        stack.layers[name] = LayerLevel(
            layer=info["gds_layer"],
            thickness=via.thickness,
            zmin=via.zmin,
            material=via.material,
            sidewall_angle=via.sidewall_angle,
            info=via.info,
        )
    entities = xsection_entities(kw["component"], kw["xsection_bounds"], stack)
    mesh_xsection(
        entities,
        path,
        legacy_resolutions(kw["resolutions"], 0.2),
        resolution_min=0.001,
        resolution_max=0.2,
    )


# --------------------------------------------------------------------------------------------
# Measurements (the definitions of gf180-tcad experiments/plugin-history/harness.py)
# --------------------------------------------------------------------------------------------
def _si(c, name) -> list[float]:
    out = []
    for r in c.regions["si"]:
        out += devsim.get_node_model_values(device=c.device, region=r, name=name)
    return out


def _contact_potential(c, contact: str) -> float:
    """Mean Potential over the nodes of ``contact`` (harness.contact_potential)."""
    for region in c.regions["si"]:
        try:
            elements = devsim.get_element_node_list(
                device=c.device, region=region, contact=contact
            )
        except devsim.error:
            continue
        if not elements:
            continue
        nodes = sorted({i for element in elements for i in element})
        pot = devsim.get_node_model_values(
            device=c.device, region=region, name="Potential"
        )
        return sum(pot[i] for i in nodes) / len(nodes)
    raise ValueError(f"contact {contact!r} has no silicon nodes")


def _currents(c) -> dict[str, float]:
    return {
        k: devsim.get_contact_current(
            device=c.device, contact=k, equation="ElectronContinuityEquation"
        )
        + devsim.get_contact_current(
            device=c.device, contact=k, equation="HoleContinuityEquation"
        )
        for k in ("anode", "cathode")
    }


def _depletion_width_um(c) -> tuple[float, int]:
    """Span of the bottom-edge (y = 0) nodes with n, p < 1e-2 |NetDoping| (harness)."""
    x, y, n, p, d = (_si(c, k) for k in ("x", "y", "Electrons", "Holes", "NetDoping"))
    y_cut = min(sorted(set(y)), key=lambda v: abs(v - 0.0))
    row = sorted(
        {
            xi: (ni, pi, di) for xi, yi, ni, pi, di in zip(x, y, n, p, d) if yi == y_cut
        }.items()
    )
    depleted = [
        xi for xi, (ni, pi, di) in row if ni < 1e-2 * abs(di) and pi < 1e-2 * abs(di)
    ]
    return ((max(depleted) - min(depleted)) * 1e4 if depleted else 0.0), len(depleted)


def _bottom_spacing_um(c, half_window_um: float = 0.1) -> float:
    """Largest bottom-edge (y = 0) node spacing within +-half_window_um of the junction."""
    x, y, d = (_si(c, k) for k in ("x", "y", "NetDoping"))
    row = sorted({xi * 1e4: di for xi, yi, di in zip(x, y, d) if yi == 0.0}.items())
    xj = 0.5 * (
        max(xi for xi, di in row if di > 0) + min(xi for xi, di in row if di < 0)
    )
    xs = [xi for xi, _ in row if abs(xi - xj) <= half_window_um]
    return max(b - a for a, b in zip(xs, xs[1:]))


def _analytic_inner_width_um(n_cm3: float = 1e17, fraction: float = 1e-2) -> float:
    """Depletion-approximation width where n and p < ``fraction`` |N|, symmetric junction.

    With the simple_physics constants DDComponent solves with (eps_si = 11.1, eps_0 =
    8.85e-14 F/cm, q = 1.6e-19 C, n_i = 1e10 cm^-3, 300 K): each side depletes over
    x_d = sqrt(eps Vbi / (q N)), and n < fraction N where the Potential is more than
    Vt ln(1 / fraction) below the neutral level, i.e. inside x_d - sqrt(2 eps Vt ln(1/f) / (q N))
    of the junction. 0.0666 um for 1e17 / 1e17 (0.068 um with eps_si = 11.7).
    """
    from devsim.python_packages import simple_physics as sp

    eps = sp.eps_si * sp.eps_0
    vbi = _VT * math.log(n_cm3 * n_cm3 / sp.n_i**2)
    xd = math.sqrt(eps * vbi / (sp.q * n_cm3))
    inner = xd - math.sqrt(2 * eps * _VT * math.log(1 / fraction) / (sp.q * n_cm3))
    return 2 * inner * 1e4


JUNCTION_OFFSET_UM = (
    0.2  # > the ~0.075 um depletion half-width at 1e17 / 1e17, < the rib half-width
)


def _bottom_row(c, regions):
    out = []
    for r in regions:
        v = {
            k: devsim.get_node_model_values(device=c.device, region=r, name=k)
            for k in ("x", "y", "Potential", "NetDoping")
        }
        out += [
            (xi * 1e4, r, pi, di)
            for xi, yi, pi, di in zip(v["x"], v["y"], v["Potential"], v["NetDoping"])
            if yi == 0.0
        ]
    return sorted(out)


def _junction(c) -> dict:
    """Si Potential at the junction itself, on the bottom edge (y = 0) through it.

    ``x_um`` is the metallurgical junction (midway between the last NetDoping > 0 and first
    < 0 node). ``n_v``/``p_v`` are the Potential at the nodes nearest ``x_um`` -/+
    JUNCTION_OFFSET_UM (the neutral n and p sides, inside the rib), ``drop_v = n_v - p_v`` the
    junction's potential drop, and ``boundary_jump_v`` the largest Potential difference
    between the ``core`` and ``slab90`` nodes at the rib's two bottom corners (0 if the two
    regions are coupled).
    """
    row = _bottom_row(c, c.regions["si"])
    xj = 0.5 * (
        max(x for x, _, _, d in row if d > 0) + min(x for x, _, _, d in row if d < 0)
    )

    def at(x0, rows=row):
        return min(rows, key=lambda t: (abs(t[0] - x0), t[1]))[2]

    core = _bottom_row(c, ["core"])
    slab = _bottom_row(c, ["slab90"])
    corners = (core[0][0], core[-1][0])
    jump = max(abs(at(x0, core) - at(x0, slab)) for x0 in corners)
    n_v, p_v = at(xj - JUNCTION_OFFSET_UM), at(xj + JUNCTION_OFFSET_UM)
    return {
        "x_um": xj,
        "junction_v": at(xj),
        "n_v": n_v,
        "p_v": p_v,
        "drop_v": n_v - p_v,
        "boundary_jump_v": jump,
    }


def _bias_state(c) -> dict:
    pot = {k: _contact_potential(c, k) for k in ("anode", "cathode")}
    return {
        "dv": pot["anode"] - pot["cathode"],
        "currents": _currents(c),
        "junction": _junction(c),
    }


def _solve(workdir: Path, mesh_scale: float, v080_topology: bool) -> dict:
    """Build, solve and ramp the PN slab; every measurement, then the device is deleted.

    ``v080_topology``: the ``core___slab90`` group of the mesh is not added as a DEVSIM
    interface, which is the device v0.8.0's mesher produced (its mesh has no such group).
    """
    from gplugins.devsim import get_simulation
    from gplugins.devsim.get_solver import DDComponent

    workdir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(workdir)
    real_add_interface = get_simulation.add_gmsh_interface
    if v080_topology:

        def add_interface(**kw):
            if kw["gmsh_name"] != RIB_SLAB_INTERFACE:
                real_add_interface(**kw)

        get_simulation.add_gmsh_interface = add_interface
    try:
        c = DDComponent(**pn_slab_inputs(mesh_scale))
        c.ddsolver()
        regions = list(devsim.get_region_list(device=c.device))
        counts = {
            r: len(devsim.get_node_model_values(device=c.device, region=r, name="x"))
            for r in regions
        }
        doping = _si(c, "NetDoping")
        width, nodes = _depletion_width_um(c)
        out = {
            "regions": {k: list(v) for k, v in c.regions.items()},
            "devsim_regions": regions,
            "contacts": list(devsim.get_contact_list(device=c.device)),
            "interfaces": list(devsim.get_interface_list(device=c.device)),
            "node_count": sum(counts.values()),
            "net_doping": (min(doping), max(doping)),
            "depletion_width_um": width,
            "depleted_nodes": nodes,
            "bottom_spacing_um": _bottom_spacing_um(c),
            "equilibrium": _bias_state(c),
            "bias": [],
        }
        for i, v in enumerate(PN_VOLTAGES):
            vinit = 0.0 if i == 0 else PN_VOLTAGES[i - 1]
            c.ramp_voltage(Vfinal=v, Vstep=0.1, Vinit=vinit, contact_name="cathode")
            out["bias"].append(_bias_state(c))
        c.delete_device()
        return out
    finally:
        get_simulation.add_gmsh_interface = real_add_interface
        os.chdir(cwd)


@pytest.fixture(scope="module")
def pn_slab(tmp_path_factory):
    """``pn_slab(mesh_scale, v080_topology=False)``: the solved PN slab, computed once each."""
    cache: dict = {}
    root = tmp_path_factory.mktemp("pn_slab")

    def get(mesh_scale: float, v080_topology: bool = False) -> dict:
        key = (mesh_scale, v080_topology)
        if key not in cache:
            name = f"scale{mesh_scale}_{'v080' if v080_topology else 'port'}"
            cache[key] = _solve(root / name, mesh_scale, v080_topology)
        return cache[key]

    yield get
    # DDComponent turns DEVSIM's extended precision on globally; restore the default.
    for name in ("extended_solver", "extended_model", "extended_equation"):
        devsim.set_parameter(name=name, value=False)


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
def test_pn_slab_physical_anchors_match_v080(pn_slab, record_property) -> None:
    """The PN slab's names and physical anchors are those of the v0.8.0 record.

    Names: the same regions, region materials and contacts. The interfaces are v0.8.0's plus
    ``core___slab90``, between the rib and the slab: meshwell tags it, v0.8.0's mesher did not
    (its mesh has no such physical group), so v0.8.0 left the rib, which holds the junction,
    uncoupled. ``test_pn_slab_currents_explained`` shows the consequences.

    Tolerances are the larger of the two meshers' self-differences (mesh x1 vs x0.5): v0.8.0
    (``v0.8.0.json`` vs ``v0.8.0-mesh0p5.json``) and this port (the same ``resolutions`` x1 vs
    x0.5). Measured on the TCAD image (devsim 2.11.1, meshwell 2.5.2, gmsh 4.15.2) on
    2026-10-19:

    - anode - cathode contact potential at 0 / 0.1 / 0.2 V: v0.8.0 0.9586540658880733 /
      0.8586540658880735 / 0.7586540658880733 V; its self-difference is 8.9e-16 V, the port's
      is 0, and the port agrees with v0.8.0 within it. It is the ohmic-contact difference
      2 Vt ln(1.1e18 / n_i), set by the contact doping alone, not by the mesh: this is a
      boundary-condition check, not a mesh check. Tolerance: the larger self-difference, at
      least 4 ulp(1 V) = 8.9e-16 V.
    - each 0.1 V cathode step lowers it by 0.1 V, to the same tolerance (+2 ulp).
    - NetDoping range: exactly -1.1e18 ... 1.1e18 cm^-3 at both scales, as v0.8.0.
    - depletion width (bottom edge, n and p < 1e-2 |NetDoping|), compared at mesh x0.5
      because at x1 v0.8.0 has only 2 depleted nodes: v0.8.0 0.03863 / 0.06883 um at x1 / x0.5
      (self-difference 0.03020 um), the port 0.06000 / 0.06000 um. The width is a difference
      of two node x positions, so it is quantised to the bottom-edge node spacing (about
      0.02 um at x1, 0.01 um at x0.5): the port's self-difference of 0 is a quantisation
      coincidence, not mesh convergence. Against v0.8.0 (port at x0.5 is 0.0088 um from v0.8.0
      at x0.5) the tolerance is the larger self-difference, 0.0302 um, which is loose. The
      tighter check is against the depletion-approximation width of the n, p < 1 % |N|
      region, 0.0666 um (``_analytic_inner_width_um``), within one bottom-edge node spacing
      near the junction at x0.5 (measured and recorded, ~0.01 um).
    - node counts (reported only): v0.8.0 3523 / 11326, the port 3384 / 10424.
    """
    full, half = pn_slab(1.0), pn_slab(0.5)
    for mesh_scale, res in ((1.0, full), (0.5, half)):
        record_property(f"node_count_x{mesh_scale}", res["node_count"])
        record_property(f"v080_node_count_x{mesh_scale}", V080_NODE_COUNT[mesh_scale])
        assert res["regions"] == V080_REGIONS
        assert sorted(res["devsim_regions"]) == V080_DEVSIM_REGIONS
        assert sorted(res["contacts"]) == V080_CONTACTS
        assert sorted(res["interfaces"]) == sorted(
            [*V080_INTERFACES, RIB_SLAB_INTERFACE]
        )
        assert res["net_doping"] == V080_NET_DOPING

    # contact potentials
    v080_self = max(
        abs(V080_DV_EQ[1.0] - V080_DV_EQ[0.5]),
        *(abs(a - b) for a, b in zip(V080_DV_BIAS[1.0], V080_DV_BIAS[0.5])),
    )
    port_self = max(
        abs(full["equilibrium"]["dv"] - half["equilibrium"]["dv"]),
        *(abs(a["dv"] - b["dv"]) for a, b in zip(full["bias"], half["bias"])),
    )
    tol_dv = max(v080_self, port_self, 4 * math.ulp(1.0))
    record_property("dv_tolerance_v", tol_dv)
    for mesh_scale, res in ((1.0, full), (0.5, half)):
        record_property(
            f"dv_x{mesh_scale}",
            [res["equilibrium"]["dv"]] + [b["dv"] for b in res["bias"]],
        )
        record_property(f"depleted_nodes_x{mesh_scale}", res["depleted_nodes"])
    for mesh_scale, res in ((1.0, full), (0.5, half)):
        dv = [res["equilibrium"]["dv"]] + [b["dv"] for b in res["bias"]]
        expected = [V080_DV_EQ[mesh_scale], *V080_DV_BIAS[mesh_scale]]
        assert all(abs(a - b) <= tol_dv for a, b in zip(dv, expected)), (
            dv,
            expected,
            tol_dv,
        )
        steps = [dv[i] - dv[i + 1] for i in range(len(dv) - 1)]
        assert all(abs(s - 0.1) <= tol_dv + 2 * math.ulp(1.0) for s in steps), steps
    # the analytic ohmic-contact difference (simple_physics constants, 300 K)
    assert abs(full["equilibrium"]["dv"] - 2 * _VT * math.log(1.1e18 / _NI)) < 1e-12

    # depletion width
    tol_w = max(
        abs(V080_DEPLETION_UM[1.0] - V080_DEPLETION_UM[0.5]),
        abs(full["depletion_width_um"] - half["depletion_width_um"]),
    )
    record_property(
        "depletion_width_um", (full["depletion_width_um"], half["depletion_width_um"])
    )
    record_property("depletion_tolerance_um", tol_w)
    assert abs(half["depletion_width_um"] - V080_DEPLETION_UM[0.5]) <= tol_w
    analytic = _analytic_inner_width_um()
    spacing = half["bottom_spacing_um"]
    record_property("depletion_analytic_um", analytic)
    record_property("bottom_spacing_x0.5_um", spacing)
    assert abs(half["depletion_width_um"] - analytic) <= spacing, (
        half["depletion_width_um"],
        analytic,
        spacing,
    )


def test_pn_slab_currents_explained(pn_slab, record_property) -> None:
    """Why v0.8.0's PN-slab currents are noise: the bias never reached the junction.

    **Verdict: the hypothesis holds.** The junction sits in the rib (``core``) region. v0.8.0's
    mesher did not tag the rib/slab boundary, so DEVSIM had no ``core___slab90`` interface:
    the rib was a floating region with no path to either contact. The contacts are on
    ``slab90``, so the bias moved the contact potentials (by exactly 0.1 V per step) and nothing
    else. Measured in the v0.8.0 venv itself (the venv ``python -m gf180tcad.history --version
    v0.8.0 --workdir DIR`` keeps, running checks_v080's PN slab with this file's junction
    probe, mesh x1 and x0.5; recorded in gf180-tcad's Phase 2 Task 5 report): its mesh has no
    ``core___slab90`` group; the junction drop (Potential at x_j - 0.2 um minus x_j + 0.2 um,
    bottom edge) is 0.83450 V at 0, 0.1 and 0.2 V, the built-in potential
    Vt ln(N_A N_D / n_i^2) of the 1e17 / 1e17 junction; the core and slab90 Potentials at the
    rib's bottom corners differ by 0.46 V (x1) and the rib's absolute Potential is
    mesh-dependent (-0.470 V at x1, -0.671 V at x0.5 at the junction), which is the
    non-physical, mesh-dependent Potential span of the record (0.4793 + 0.8820 = 1.3613 V,
    0.4793 + 1.1474 = 1.6267 V).

    TODO(gf180-tcad, after-Task-8 bump PR): commit that v0.8.0 junction probe to gf180-tcad
    ``experiments/plugin-history/``, its output to ``results/plugin-history/``, and point this
    docstring at them instead of the report.

    This test shows both devices on the latest stack:

    - **v0.8.0 topology** (the same mesh with ``core___slab90`` left out of DEVSIM): the
      currents stay below the 1e-26 A/cm noise floor, the junction drop is 0.83450 V at every
      step, and the regions disagree at the rib's corners (non-zero corner jump): v0.8.0's
      signatures, reproduced. The floating rib's absolute Potential differs from v0.8.0's
      (and between meshes) because nothing fixes it: with no contact or interface, the rib's
      offset is set only by the solver's initial guess and the mesh, so only differences
      inside the rib (the drop) are comparable.
    - **ported topology** (all meshwell interfaces): the drop falls by the applied 0.1 V per
      step (0.8345 / 0.7345 / 0.6345 V), the regions agree at the corners, and the contact
      currents are a forward diode current (9.7e-15, then 1.0e-13 A/cm; anode = -cathode)
      whose implied ideality is between 1 and 2: for I = I0 (exp(V / n Vt) - 1) at 0.1 and
      0.2 V, I2 / I1 = exp(a) + 1 with a = 0.1 / (n Vt), so n = 0.1 / (Vt ln(I2 / I1 - 1))
      (measured 1.73).
    """
    vbi = _VT * math.log(1e17 * 1e17 / _NI**2)  # 0.834503 V

    old = pn_slab(1.0, v080_topology=True)
    assert sorted(old["interfaces"]) == V080_INTERFACES
    old_states = [old["equilibrium"], *old["bias"]]
    for s in old_states:
        j = s["junction"]
        record_property(f"v080_topology_junction_v{s['dv']:.4f}", j)
        assert all(abs(i) < CURRENT_NOISE_FLOOR for i in s["currents"].values()), s[
            "currents"
        ]
        assert abs(j["drop_v"] - vbi) < 1e-4, j
        assert abs(j["drop_v"] - old_states[0]["junction"]["drop_v"]) < 1e-9, j
        assert abs(j["junction_v"] - old_states[0]["junction"]["junction_v"]) < 1e-9, j
        assert j["boundary_jump_v"] > 0.05, j  # the rib floats

    new = pn_slab(1.0)
    states = [new["equilibrium"], *new["bias"]]
    for s, v in zip(states, (0.0, *PN_VOLTAGES)):
        j = s["junction"]
        record_property(f"port_junction_v{v}", j)
        record_property(f"port_currents_v{v}", s["currents"])
        assert abs(j["drop_v"] - (vbi - v)) < 1e-3, (v, j)
        assert j["boundary_jump_v"] < 1e-6, j
    # equilibrium: no current; forward bias: a conserved current that grows like a diode's
    assert all(abs(i) < CURRENT_NOISE_FLOOR for i in states[0]["currents"].values())
    i1, i2 = (s["currents"]["cathode"] for s in states[1:])
    for s in states[1:]:
        a, k = s["currents"]["anode"], s["currents"]["cathode"]
        assert k > 1e6 * CURRENT_NOISE_FLOOR and abs(a + k) <= 1e-6 * abs(k), s[
            "currents"
        ]
    ideality = 0.1 / (_VT * math.log(i2 / i1 - 1))
    record_property("port_diode_ideality", ideality)
    assert 1.0 <= ideality <= 2.0, (i1, i2, ideality)


_MESH_SCRIPT = """
import json, os, runpy, sys
import gmsh
m = runpy.run_path(sys.argv[1])
m["write_pn_slab_mesh"](sys.argv[2])
opts = {k: gmsh.option.getNumber(k) for k in (
    "General.NumThreads", "Mesh.MaxNumThreads1D", "Mesh.MaxNumThreads2D",
    "Mesh.MaxNumThreads3D", "Geometry.OCCParallel", "Mesh.RandomSeed")}
print(json.dumps({"cpu_count": os.cpu_count(),
                  "PYTHON_CPU_COUNT": os.environ.get("PYTHON_CPU_COUNT"), "options": opts}))
"""


def _mesh_in_subprocess(path: Path) -> dict:
    """Mesh the PN slab in a fresh Python with ``PYTHON_CPU_COUNT`` unset."""
    import gplugins

    env = {k: v for k, v in os.environ.items() if k != "PYTHON_CPU_COUNT"}
    repo = str(Path(gplugins.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([repo, env.get("PYTHONPATH", "")])
    res = subprocess.run(
        [sys.executable, "-c", _MESH_SCRIPT, __file__, str(path)],
        env=env,
        cwd=path.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr[-4000:]
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_mesh_deterministic(tmp_path, monkeypatch) -> None:
    """Two builds give byte-identical .msh files, also in a process with another cpu_count."""
    monkeypatch.chdir(tmp_path)
    first, second, other = (
        tmp_path / f"{n}.msh2" for n in ("first", "second", "other")
    )
    write_pn_slab_mesh(str(first))
    write_pn_slab_mesh(str(second))
    data = first.read_bytes()
    assert data.startswith(b"$MeshFormat\n2.2 ")
    assert data == second.read_bytes()
    _mesh_in_subprocess(other)
    assert data == other.read_bytes()


def test_gmsh_threads_one(tmp_path, monkeypatch) -> None:
    """After mesh_xsection gmsh is single-threaded with the pinned seed.

    Checked in this process and in a fresh one with ``PYTHON_CPU_COUNT`` unset, where meshwell's
    import-time ``cpu_count()`` default is the host's, so the fork does not rely on the TCAD
    image's ``ENV PYTHON_CPU_COUNT=1``.
    """
    import gmsh

    from gplugins.devsim.meshing import GMSH_RANDOM_SEED

    expected = {
        "General.NumThreads": 1,
        "Mesh.MaxNumThreads1D": 1,
        "Mesh.MaxNumThreads2D": 1,
        "Mesh.MaxNumThreads3D": 1,
        "Geometry.OCCParallel": 0,
        "Mesh.RandomSeed": GMSH_RANDOM_SEED,
    }
    monkeypatch.chdir(tmp_path)
    write_pn_slab_mesh(str(tmp_path / "here.msh2"))
    assert {k: gmsh.option.getNumber(k) for k in expected} == expected

    out = _mesh_in_subprocess(tmp_path / "unset.msh2")
    assert out["PYTHON_CPU_COUNT"] is None
    assert out["options"] == expected
    if len(os.sched_getaffinity(0)) > 1:
        # a meaningful check: there, meshwell's default would have been more than one thread
        assert out["cpu_count"] > 1
