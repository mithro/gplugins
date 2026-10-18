# SPDX-License-Identifier: MIT
"""The C-V solver (:func:`gplugins.devsim.cv.sweep_cv`): quasi-static and small-signal AC Cgg.

Every hand value uses :data:`gplugins.devsim.materials.CONSTANTS`, so both sides of each
comparison use the same numbers. DEVSIM works in cm: in 1D a capacitance is F/cm^2, in 2D F/cm.

The 1D stacks are the gf180-tcad Phase 2 MOSCAPs: nfet n+ poly 1e20 on p 1.7e17 with 15.2 nm
oxide, pfet p+ poly 3.6e19 on n 1.7e17 with 15.6 nm oxide (Boltzmann flat band -1.029 V and
+1.003 V). With only a body contact the inversion layer of a 1D MOSCAP responds in about
2 tau_g N_A / n_i ~ 4e2 s (Nicollian & Brews), so 1D AC is the high-frequency curve in
inversion at any practical frequency and equals the quasi-static curve only in accumulation
and depletion.
"""

from __future__ import annotations

import math

import devsim
import pytest

from gplugins.devsim import doping
from gplugins.devsim.contacts import Contact, InterfaceRef, delete_circuit
from gplugins.devsim.cv import CVConvergenceError, CVResult, PointInfo, sweep_cv
from gplugins.devsim.get_solver import DDComponent
from gplugins.devsim.materials import CONSTANTS
from gplugins.devsim.stack1d import Layer1D, Mesh1D, create_1d_stack

UM = 1e-4  # cm
N_BODY = 1.7e17
NFET = {"name": "nfet", "gate": ("donor", 1e20), "body": "acceptor", "tox_um": 0.0152, "pol": 1}
PFET = {"name": "pfet", "gate": ("acceptor", 3.6e19), "body": "donor", "tox_um": 0.0156, "pol": -1}


@pytest.fixture(autouse=True)
def clean_devsim():
    """Fail on, and remove, any device a test leaves; delete leftover circuits and meshes;
    restore DEVSIM precision."""
    yield
    left = list(devsim.get_device_list())
    for device in left:
        devsim.delete_device(device=device)
    delete_circuit()
    for mesh in devsim.get_mesh_list():
        devsim.delete_mesh(mesh=mesh)
    for name in ("extended_solver", "extended_model", "extended_equation"):
        devsim.set_parameter(name=name, value=False)
    assert not left, f"devices left behind: {left}"


def cox(case) -> float:
    """Oxide capacitance (F/cm^2)."""
    return CONSTANTS.eps_ox * CONSTANTS.eps0_f_per_cm / (case["tox_um"] * UM)


def vfb_poly(case) -> float:
    """Boltzmann poly-gate flat band: -pol Vt ln(NGATE N_body / n_i^2)."""
    return -case["pol"] * CONSTANTS.vt_v * math.log(case["gate"][1] * N_BODY / CONSTANTS.n_i_cm3**2)


def _poly_stack(name: str, case: dict) -> DDComponent:
    gate_kind, gate_conc = case["gate"]
    layers = [
        Layer1D("gate_poly", "silicon", 0.2, doping.constant(gate_conc), gate_kind),
        Layer1D("oxide", "oxide", case["tox_um"], None),
        Layer1D("body", "silicon", 1.0, doping.constant(N_BODY), case["body"]),
    ]
    contacts = [
        Contact("gate", InterfaceRef("gate_poly", None), "ohmic"),
        Contact("body", InterfaceRef("body", None), "ohmic"),
    ]
    mesh = Mesh1D({"gate_poly": (0.01, 0.0002), "oxide": (0.0002, 0.0002), "body": (0.0002, 0.02)})
    return create_1d_stack(name, layers, contacts, mesh, CONSTANTS)


def _metal_stack(name: str) -> DDComponent:
    """Metal gate (work-function offset 0, so Vfb = psi_b) on 15.2 nm oxide on p 1.7e17."""
    layers = [
        Layer1D("oxide", "oxide", 0.0152, None),
        Layer1D("body", "silicon", 1.0, doping.constant(N_BODY), "acceptor"),
    ]
    contacts = [
        Contact("gate", InterfaceRef("oxide", None), "gate"),
        Contact("body", InterfaceRef("body", None), "ohmic"),
    ]
    mesh = Mesh1D({"oxide": (0.0002, 0.0002), "body": (0.0002, 0.02)})
    return create_1d_stack(name, layers, contacts, mesh, CONSTANTS)


def _state(dd: DDComponent) -> dict:
    """Every equation command and solution value of ``dd`` (to check that a sweep restores it)."""
    device = dd.device
    out = {}
    for region in devsim.get_region_list(device=device):
        for eq in devsim.get_equation_list(device=device, region=region):
            out[("eq", region, eq)] = devsim.get_equation_command(device=device, region=region, name=eq)
        for var in ("Potential", "Electrons", "Holes"):
            if var in devsim.get_node_model_list(device=device, region=region):
                out[("val", region, var)] = tuple(
                    devsim.get_node_model_values(device=device, region=region, name=var)
                )
    for contact in devsim.get_contact_list(device=device):
        for eq in devsim.get_contact_equation_list(device=device, contact=contact):
            out[("ceq", contact, eq)] = devsim.get_contact_equation_command(
                device=device, contact=contact, name=eq
            )
        out[("bias", contact)] = devsim.get_parameter(device=device, name=f"{contact}_bias")
    for interface in devsim.get_interface_list(device=device):
        for eq in devsim.get_interface_equation_list(device=device, interface=interface):
            out[("ieq", interface, eq)] = devsim.get_interface_equation_command(
                device=device, interface=interface, name=eq
            )
    return out


# --------------------------------------------------------------------------------------------
# 1D: quasi-static vs small-signal AC
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", [NFET, PFET], ids=["nfet", "pfet"])
def test_qs_equals_ac_low_freq_acc_dep(case) -> None:
    """At 1 Hz, in accumulation and depletion (pol*vg <= pol*Vfb + 0.3 V), AC equals QS (0.1 %)."""
    pol, vfb = case["pol"], vfb_poly(case)
    pvg = [-3.0, -2.0, -1.5, -1.25, -1.1, pol * vfb, -0.9, -0.8, pol * vfb + 0.3]
    vg = [pol * v for v in pvg]
    assert all(pol * v <= pol * vfb + 0.3 + 1e-12 for v in vg)
    dd = _poly_stack(f"qsac_{case['name']}", case)
    before = _state(dd)
    qs = sweep_cv(dd, "gate", vg, method="qs")
    ac = sweep_cv(dd, "gate", vg, method="ac", freq_hz=1.0)
    assert isinstance(qs, CVResult) and isinstance(ac, CVResult)
    assert qs.unit == ac.unit == "F/cm^2"
    assert qs.method == "qs" and ac.method == "ac" and ac.freq_hz == 1.0 and qs.freq_hz is None
    assert qs.vg == ac.vg == tuple(vg)
    c0 = cox(case)
    worst = 0.0
    for v, cq, ca in zip(vg, qs.cgg, ac.cgg):
        rel = ca / cq - 1
        worst = max(worst, abs(rel))
        print(f"{case['name']} vg {v:+.4f}: QS {cq / c0:.6f} Cox, AC(1 Hz) {ca / c0:.6f} Cox, rel {rel:+.3e}")
        assert 0.05 * c0 < cq < c0
    print(f"{case['name']}: max |AC/QS - 1| = {worst:.3e}")
    assert worst <= 1e-3
    assert _state(dd) == before  # the sweeps restore the device's equations and solution
    dd.delete_device()


@pytest.mark.parametrize("case", [NFET, PFET], ids=["nfet", "pfet"])
def test_ac_inversion_is_hf(case) -> None:
    """At 1 MHz, 1D AC in strong inversion is the HF curve: more than 30 % below QS."""
    pol = case["pol"]
    vg = [pol * 3.0, pol * 5.0]
    dd = _poly_stack(f"hf_{case['name']}", case)
    qs = sweep_cv(dd, "gate", vg, method="qs")
    ac = sweep_cv(dd, "gate", vg, method="ac", freq_hz=1e6)
    c0 = cox(case)
    for v, cq, ca in zip(vg, qs.cgg, ac.cgg):
        print(
            f"{case['name']} vg {v:+.1f}: QS {cq / c0:.6f} Cox, AC(1 MHz) {ca / c0:.6f} Cox, "
            f"AC/QS {ca / cq:.6f}"
        )
        assert cq > 0.7 * c0  # QS recovers towards Cox in inversion (poly depletion aside)
        assert 0.0 < ca < 0.7 * cq
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# 1D metal gate: accumulation equals Cox in series with the exact Poisson-Boltzmann C_s
# --------------------------------------------------------------------------------------------
def semiconductor_capacitance(psi_s: float, p0: float, n0: float) -> float:
    """|dQ_s/dpsi_s| (F/cm^2) of a p-type surface, exact Poisson-Boltzmann (Boltzmann carriers).

    Sze & Ng, *Physics of Semiconductor Devices*, 3rd ed., ch. 4 (the MIS capacitor's space
    charge, Q_s = -sign(psi_s) sqrt(2) eps_s/(beta L_D) F(beta psi_s, n0/p0)),
    https://doi.org/10.1002/0470068329: with beta = 1/V_t and L_D = sqrt(eps_s V_t/(q p0)),
    F = sqrt((e^(-b) + b - 1) + (n0/p0)(e^b - b - 1)) and
    C_s = eps_s/(sqrt(2) L_D) [1 - e^(-b) + (n0/p0)(e^b - 1)] / F, b = beta psi_s.
    """
    eps_s = CONSTANTS.eps_si * CONSTANTS.eps0_f_per_cm
    vt, q = CONSTANTS.vt_v, CONSTANTS.q_c
    b = psi_s / vt
    l_d = math.sqrt(eps_s * vt / (q * p0))
    f = math.sqrt((math.exp(-b) + b - 1) + (n0 / p0) * (math.exp(b) - b - 1))
    return abs(eps_s / (math.sqrt(2) * l_d) * (1 - math.exp(-b) + (n0 / p0) * (math.exp(b) - 1)) / f)


def test_accumulation_series_exact() -> None:
    """Metal gate, accumulation: Cgg_qs = (1/Cox + 1/C_s(psi_s))^-1 within 0.5 %.

    psi_s (surface minus bulk potential) is read from the solution at each bias; the
    accumulation DC point is the equilibrium, so drift-diffusion reaches it directly.
    """
    vg = [-2.5, -1.5, -1.0]  # Vfb = psi_b = -0.433 V
    dd = _metal_stack("series")
    qs = sweep_cv(dd, "gate", vg, method="qs")
    c_ox = CONSTANTS.eps_ox * CONSTANTS.eps0_f_per_cm / (0.0152 * UM)
    ni = CONSTANTS.n_i_cm3
    p0 = 1e-10 + 0.5 * abs(N_BODY + math.sqrt(N_BODY**2 + 4 * ni**2))  # simple_physics' value
    n0 = ni**2 / p0
    x = devsim.get_node_model_values(device=dd.device, region="body", name="x")
    i_surf, i_bulk = x.index(min(x)), x.index(max(x))
    state = 0.0
    for v, c in zip(vg, qs.cgg):
        n = max(1, math.ceil(abs(v - state) / 0.1))
        for k in range(1, n + 1):
            dd.set_bias("gate", state + (v - state) * k / n)
            dd.solve_dc()
        state = v
        pot = devsim.get_node_model_values(device=dd.device, region="body", name="Potential")
        psi_s = pot[i_surf] - pot[i_bulk]
        cs = semiconductor_capacitance(psi_s, p0, n0)
        c_series = 1 / (1 / c_ox + 1 / cs)
        rel = c / c_series - 1
        print(
            f"metal gate vg {v:+.2f}: psi_s {psi_s:+.6f} V, C_s {cs!r}, series {c_series!r}, "
            f"QS {c!r} F/cm^2, rel {rel:+.3e}"
        )
        assert psi_s < 0  # accumulation of holes
        assert abs(rel) <= 5e-3
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# QS step convergence and point bookkeeping
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", [NFET, PFET], ids=["nfet", "pfet"])
def test_qs_step_converged(case) -> None:
    """Halving dv_qs changes Cgg by less than 0.01 % at every point of -5..+5 V."""
    vg = [0.5 * k for k in range(-10, 11)]
    dd = _poly_stack(f"step_{case['name']}", case)
    a = sweep_cv(dd, "gate", vg, method="qs", dv_qs=1e-3)
    b = sweep_cv(dd, "gate", vg, method="qs", dv_qs=5e-4)
    assert a.dv_qs == 1e-3 and b.dv_qs == 5e-4
    c0 = cox(case)
    worst = 0.0
    for v, ca, cb in zip(vg, a.cgg, b.cgg):
        rel = cb / ca - 1
        worst = max(worst, abs(rel))
        print(f"{case['name']} vg {v:+.1f}: Cgg {ca / c0:.6f} Cox, halved-step change {rel:+.3e}")
        assert abs(rel) < 1e-4
    print(f"{case['name']}: max |dC/C| on halving dv_qs = {worst:.3e}")
    assert len(a.points) == len(vg)
    for p in a.points + b.points:
        assert isinstance(p, PointInfo)
        assert p.converged is True and p.iterations >= 1
        assert p.rel_error <= 1e-10  # the requested relative tolerance
        assert p.abs_error <= 1e-12  # the potential update, V (abs_tol default)
    dd.delete_device()


def test_nonconvergence_raises() -> None:
    """A point that cannot meet the tolerance raises CVConvergenceError naming it; never dropped."""
    dd = _poly_stack("noconv", NFET)
    before = _state(dd)
    with pytest.raises(CVConvergenceError) as err:
        sweep_cv(dd, "gate", [2.0], method="qs", rel_tol=1e-16, max_iterations=2)
    assert err.value.points == [2.0]
    with pytest.raises(CVConvergenceError) as err:
        sweep_cv(dd, "gate", [-1.0, 2.0], method="ac", freq_hz=1e3, rel_tol=1e-16, max_iterations=2)
    assert err.value.points == [-1.0, 2.0]
    assert _state(dd) == before
    dd.delete_device()


def test_bad_arguments() -> None:
    dd = _poly_stack("badargs", NFET)
    with pytest.raises(ValueError):
        sweep_cv(dd, "nope", [0.0], method="qs")
    with pytest.raises(ValueError):
        sweep_cv(dd, "gate", [0.0], method="ac")  # no frequency
    with pytest.raises(ValueError):
        sweep_cv(dd, "gate", [0.0], method="qs", freq_hz=1.0)
    with pytest.raises(ValueError):
        sweep_cv(dd, "gate", [0.0], method="hf")
    with pytest.raises(ValueError):
        sweep_cv(dd, "gate", [], method="qs")
    with pytest.raises(ValueError):
        sweep_cv(dd, "gate", [0.0], method="qs", dv_qs=0.0)
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# 2D: a third contact, charge conservation over every non-gate contact
# --------------------------------------------------------------------------------------------
TOX_2D_UM = 0.0152


def _device_2d(name: str) -> DDComponent:
    """1 um wide: metal gate on 15.2 nm oxide on p 1.7e17 (1 um deep), body contact at the
    bottom, and a side ohmic contact on the left edge of an n+ 1e20 pocket (x < 0.2 um,
    y < 0.1 um) under the gate's edge."""
    mesh = f"{name}_mesh"
    devsim.create_2d_mesh(mesh=mesh)
    # DEVSIM's 2D mesher makes a contact where its region meets another one, so the device is
    # surrounded by an equation-free "air" margin (as DEVSIM's testing/mos_2d_create.py).
    m = 0.01  # um
    for pos, ps, ns in ((-m, m, m), (0.0, 0.01, 0.01), (0.2, 0.004, 0.004), (1.0, 0.05, 0.05), (1.0 + m, m, m)):
        devsim.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos * UM, ps=ps * UM, ns=ns * UM)
    for pos, ps, ns in (
        (-TOX_2D_UM - m, m, m),
        (-TOX_2D_UM, 0.004, 0.004),
        (0.0, 0.0005, 0.002),
        (0.1, 0.004, 0.004),
        (1.0, 0.1, 0.1),
        (1.0 + m, m, m),
    ):
        devsim.add_2d_mesh_line(mesh=mesh, dir="y", pos=pos * UM, ps=ps * UM, ns=ns * UM)
    y_top = -TOX_2D_UM * UM
    devsim.add_2d_region(mesh=mesh, material="air", region="air")
    devsim.add_2d_region(mesh=mesh, material="oxide", region="oxide", xl=0.0, xh=UM, yl=y_top, yh=0.0)
    devsim.add_2d_region(mesh=mesh, material="silicon", region="si", xl=0.0, xh=UM, yl=0.0, yh=UM)
    devsim.add_2d_interface(mesh=mesh, name="oxide___si", region0="oxide", region1="si")
    devsim.add_2d_contact(mesh=mesh, name="gate", region="oxide", material="metal", yl=y_top, yh=y_top)
    devsim.add_2d_contact(mesh=mesh, name="body", region="si", material="metal", yl=UM, yh=UM)
    devsim.add_2d_contact(
        mesh=mesh, name="side", region="si", material="metal", xl=0.0, xh=0.0, yl=0.0, yh=0.1 * UM
    )
    devsim.finalize_mesh(mesh=mesh)
    devsim.create_device(mesh=mesh, device=name)
    devsim.node_model(
        device=name,
        region="si",
        name="NetDoping",
        equation=f"1e20*step({0.2 * UM!r} - x)*step({0.1 * UM!r} - y) - {N_BODY!r}",
    )
    contacts = [
        Contact("gate", InterfaceRef("oxide", None), "gate"),
        Contact("body", InterfaceRef("si", None), "ohmic"),
        Contact("side", InterfaceRef("si", None), "ohmic"),
    ]
    return DDComponent.from_device(
        device=name,
        mesh=mesh,
        region_classes={"air": None, "oxide": "oxide", "si": "silicon"},
        contacts=contacts,
        constants=CONSTANTS,
        atol=1e10,
        rtol=1e-10,
        max_iter=100,
    )


def test_cgx_sum_2d() -> None:
    """At 1 kHz, Cgg = sum of Cxg over every non-gate contact (1e-6), in depletion and inversion.

    The metal-gate flat band is psi_b = -0.433 V; 0 V is depletion, +3 V strong inversion. In
    inversion both the body and the side (n+) contact carry a non-zero share.
    """
    dd = _device_2d("cgx2d")
    before = _state(dd)
    res = sweep_cv(dd, "gate", [0.0, 3.0], method="ac", freq_hz=1e3)
    assert res.unit == "F/cm"
    for i, v in enumerate(res.vg):
        cgg = res.cgg[i]
        parts = {x: res.cgx[x][i] for x in res.cgx}
        total = math.fsum(parts.values())
        print(f"2D vg {v:+.1f}: Cgg {cgg!r} F/cm, Cxg {parts!r}, sum rel err {total / cgg - 1:+.3e}")
        assert cgg > 0
        assert abs(total - cgg) <= 1e-6 * abs(cgg)
    inv = {x: res.cgx[x][1] for x in res.cgx}
    assert all(abs(c) > 1e-3 * res.cgg[1] for c in inv.values()), inv
    assert _state(dd) == before
    dd.delete_device()


def test_every_non_gate_contact_sourced() -> None:
    """CVResult.cgx has exactly the non-gate contacts; the sweep leaves no circuit behind."""
    dd = _device_2d("sourced")
    res = sweep_cv(dd, "gate", [0.0], method="ac", freq_hz=1e3)
    assert set(res.cgx) == {"body", "side"}
    assert all(len(v) == 1 for v in res.cgx.values())
    for contact in devsim.get_contact_list(device=dd.device):
        for eq in devsim.get_contact_equation_list(device=dd.device, contact=contact):
            cmd = devsim.get_contact_equation_command(device=dd.device, contact=contact, name=eq)
            assert cmd["circuit_node"] == ""
    qs = sweep_cv(dd, "gate", [0.0], method="qs")
    assert qs.cgx == {}
    dd.delete_device()


def test_ac_refuses_foreign_circuit() -> None:
    """DEVSIM's circuit is global: an AC sweep refuses to run while any device uses it."""
    layers = [Layer1D("ox_a", "oxide", 0.0076, None), Layer1D("ox_b", "oxide", 0.0076, None)]
    contacts = [
        Contact("top", InterfaceRef("ox_a", None), "gate", circuit=True),
        Contact("bot", InterfaceRef("ox_b", None), "gate"),
    ]
    other = create_1d_stack(
        "other", layers, contacts, Mesh1D({"ox_a": (0.001, 0.001), "ox_b": (0.001, 0.001)}), CONSTANTS
    )
    dd = _poly_stack("victim", NFET)
    with pytest.raises(ValueError, match="circuit"):
        sweep_cv(dd, "gate", [0.0], method="ac", freq_hz=1.0)
    dd.delete_device()
    other.delete_device()


def test_package_exports() -> None:
    import gplugins.devsim as gd
    from gplugins.devsim import cv
    from gplugins.devsim.convergence import ConvergenceError

    assert gd.sweep_cv is cv.sweep_cv and gd.CVResult is cv.CVResult
    assert gd.PointInfo is cv.PointInfo and gd.CVConvergenceError is cv.CVConvergenceError
    assert issubclass(CVConvergenceError, ConvergenceError)


def test_abs_tol_continues_newton() -> None:
    """A tighter abs_tol makes Newton continue until the potential update meets it."""
    dd = _poly_stack("abstol", NFET)
    loose = sweep_cv(dd, "gate", [-1.0, 2.0], method="qs")
    tight = sweep_cv(dd, "gate", [-1.0, 2.0], method="qs", abs_tol=1e-20)
    for a, b in zip(loose.points, tight.points):
        print(f"abs_tol 1e-12: {a}; abs_tol 1e-20: {b}")
        assert b.abs_error <= 1e-20 and b.iterations > a.iterations
    for ca, cb in zip(loose.cgg, tight.cgg):
        assert abs(cb / ca - 1) <= 1e-9
    dd.delete_device()
