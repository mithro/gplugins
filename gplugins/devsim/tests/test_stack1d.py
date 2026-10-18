# SPDX-License-Identifier: MIT
"""The 1D stack builder: geometry, mesh grading, cited constants and the physics it solves.

Every hand value is computed from :data:`gplugins.devsim.materials.CONSTANTS` (the plugin's one
constants table), so both sides of each comparison use the same numbers. DEVSIM works in cm,
so in 1D a charge is C/cm^2 and a capacitance F/cm^2.
"""

from __future__ import annotations

import math

import devsim
import pytest

from gplugins.devsim import doping
from gplugins.devsim.contacts import Contact, InterfaceRef, LayerRef, delete_circuit
from gplugins.devsim.convergence import ConvergenceError
from gplugins.devsim.materials import CONSTANTS, Constants
from gplugins.devsim.stack1d import Layer1D, Mesh1D, create_1d_stack, graded_spacings

UM = 1e-4  # cm

# The Phase 2 MOSCAP stacks (gf180-tcad plan, Global Constraints): nfet n+ poly 1e20 on
# p 1.7e17 with 15.2 nm oxide, pfet p+ poly 3.6e19 on n 1.7e17 with 15.6 nm oxide.
N_BODY = 1.7e17
NFET = {"gate": ("donor", 1e20), "body": "acceptor", "tox_um": 0.0152, "vfb": -1.029}
PFET = {"gate": ("acceptor", 3.6e19), "body": "donor", "tox_um": 0.0156, "vfb": 1.003}


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


def _poly_stack(name: str, case: dict, *, circuit: bool = False):
    gate_kind, gate_conc = case["gate"]
    layers = [
        Layer1D("gate_poly", "silicon", 0.2, doping.constant(gate_conc), gate_kind),
        Layer1D("oxide", "oxide", case["tox_um"], None),
        Layer1D("body", "silicon", 1.0, doping.constant(N_BODY), case["body"]),
    ]
    contacts = [
        Contact("gate", InterfaceRef("gate_poly", None), "ohmic", circuit=circuit),
        Contact("body", InterfaceRef("body", None), "ohmic"),
    ]
    mesh = Mesh1D(
        {
            "gate_poly": (0.01, 0.0002),
            "oxide": (0.0002, 0.0002),
            "body": (0.0002, 0.02),
        }
    )
    return create_1d_stack(name, layers, contacts, mesh, CONSTANTS)


def _node_x_um(dd, region: str) -> list[float]:
    return [
        x / UM for x in devsim.get_node_model_values(device=dd.device, region=region, name="x")
    ]


# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
def test_ni_scaling() -> None:
    """n_i(T) = n_i(300 K) (T/300)^2.54 exp(-6726 K (1/T - 1/300 K)) (Misiakos & Tsamakis)."""
    t = 298.15
    expected = 9.65e9 * (t / 300) ** 2.54 * math.exp(-6726 * (1 / t - 1 / 300))
    assert math.isclose(Constants().n_i_cm3, expected, rel_tol=1e-12)
    assert math.isclose(Constants().n_i_cm3, 8.266e9, rel_tol=1e-3)
    assert math.isclose(Constants(t_k=300.0).n_i_cm3, 9.65e9, rel_tol=1e-12)


def test_constants_not_simple_physics_defaults() -> None:
    """The region parameters are the cited constants, not DEVSIM simple_physics' defaults."""
    dd = _poly_stack("consts", NFET)
    eps0 = 8.8541878128e-14
    for region in ("gate_poly", "body"):

        def get(name, region=region):
            return devsim.get_parameter(device=dd.device, region=region, name=name)

        assert math.isclose(get("Permittivity"), 11.7 * eps0, rel_tol=1e-12)
        assert get("T") == 298.15
        assert math.isclose(get("n_i"), 8.266e9, rel_tol=1e-3)
        assert get("n_i") == CONSTANTS.n_i_cm3
        assert get("mu_n") == 1417.0
        assert get("mu_p") == 470.5
        assert get("taun") == 1e-5
        assert get("taup") == 1e-5
        assert get("n1") == get("p1") == CONSTANTS.n_i_cm3
        assert get("ElectronCharge") == 1.602176634e-19
        assert math.isclose(get("V_t"), 1.380649e-23 * 298.15 / 1.602176634e-19, rel_tol=1e-15)
        assert math.isclose(get("V_t"), 25.6926e-3, rel_tol=1e-5)
    ox = devsim.get_parameter(device=dd.device, region="oxide", name="Permittivity")
    assert math.isclose(ox, 3.9 * eps0, rel_tol=1e-12)
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# Geometry and mesh
# --------------------------------------------------------------------------------------------
def test_stack1d_regions_and_spacing() -> None:
    dd = _poly_stack("geom", NFET)
    device = dd.device
    assert sorted(devsim.get_region_list(device=device)) == ["body", "gate_poly", "oxide"]
    assert sorted(devsim.get_interface_list(device=device)) == [
        "gate_poly___oxide",
        "oxide___body",
    ]
    assert sorted(devsim.get_contact_list(device=device)) == ["body", "gate"]
    assert devsim.get_region_list(device=device, contact="gate") == ("gate_poly",)
    assert devsim.get_region_list(device=device, contact="body") == ("body",)
    assert dd.region_classes == {
        "gate_poly": "silicon",
        "oxide": "oxide",
        "body": "silicon",
    }
    spacing = {
        "gate_poly": (0.01, 0.0002),
        "oxide": (0.0002, 0.0002),
        "body": (0.0002, 0.02),
    }
    bounds = {"gate_poly": (0.0, 0.2), "oxide": (0.2, 0.2152), "body": (0.2152, 1.2152)}
    for region, (h0, h1) in spacing.items():
        x = _node_x_um(dd, region)
        assert x == sorted(x)
        lo, hi = bounds[region]
        assert math.isclose(x[0], lo, abs_tol=1e-12) and math.isclose(x[-1], hi, rel_tol=1e-12)
        h = [b - a for a, b in zip(x, x[1:])]
        # within the Mesh1D limits: never coarser than requested at either end, the ends at
        # least half the requested size, and geometric grading in between
        assert h[0] <= h0 * (1 + 1e-9) and h[-1] <= h1 * (1 + 1e-9), (region, h[0], h[-1])
        assert h[0] >= 0.5 * h0 and h[-1] >= 0.5 * h1, (region, h[0], h[-1])
        assert max(h) <= max(h0, h1) * (1 + 1e-9)
        assert min(h) >= 0.5 * min(h0, h1)
        assert all(max(b / a, a / b) <= 1.25 * (1 + 1e-12) for a, b in zip(h, h[1:]))
        if len(h) > 2:
            ratios = [b / a for a, b in zip(h, h[1:])]
            r = (h1 / h0) ** (1 / (len(h) - 1))
            assert all(math.isclose(q, r, rel_tol=1e-6) for q in ratios), (region, ratios)
    # neighbouring spacings across every layer boundary are within the default max_ratio too
    x_all = sorted({x for region in spacing for x in _node_x_um(dd, region)})
    h_all = [b - a for a, b in zip(x_all, x_all[1:])]
    assert all(max(b / a, a / b) <= 1.25 * (1 + 1e-12) for a, b in zip(h_all, h_all[1:]))
    dd.delete_device()


def test_mesh1d_max_ratio() -> None:
    """Mesh1D.max_ratio (default 1.25) bounds the ratio of neighbouring spacings."""
    assert Mesh1D({"a": (1.0, 1.0)}).max_ratio == 1.25
    with pytest.raises(ValueError):
        Mesh1D({"a": (1.0, 1.0)}, max_ratio=1.0)
    # 0.01 -> 0.0002 over 0.03 um: filling the length alone would need only a few intervals,
    # with a neighbour ratio far above 1.25; max_ratio forces more, never coarser ones
    for max_ratio in (1.25, 1.1):
        h = graded_spacings(0.03, 0.01, 0.0002, max_ratio=max_ratio)
        assert math.isclose(math.fsum(h), 0.03, rel_tol=1e-12)
        ratios = [a / b for a, b in zip(h, h[1:])]
        assert all(1.0 <= r <= max_ratio * (1 + 1e-12) for r in ratios), ratios
        assert h[0] <= 0.01 and h[-1] <= 0.0002 * (1 + 1e-12)
    assert len(graded_spacings(0.03, 0.01, 0.0002)) > len(
        graded_spacings(0.03, 0.01, 0.0002, max_ratio=100.0)
    )
    # a jump of requested spacings across a layer boundary beyond max_ratio is refused
    layers = [
        Layer1D("ox", "oxide", 0.01, None),
        Layer1D("si", "silicon", 0.1, doping.constant(1e17), "acceptor"),
    ]
    contacts = [
        Contact("g", InterfaceRef("ox", None), "gate"),
        Contact("b", InterfaceRef("si", None), "ohmic"),
    ]
    with pytest.raises(ValueError, match="max_ratio"):
        create_1d_stack(
            "jump", layers, contacts, Mesh1D({"ox": (0.001, 0.001), "si": (0.005, 0.01)}), CONSTANTS
        )
    assert not devsim.get_device_list()


def test_stack1d_rejects_bad_contacts() -> None:
    layers = [
        Layer1D("ox", "oxide", 0.01, None),
        Layer1D("si", "silicon", 0.1, doping.constant(1e17), "acceptor"),
    ]
    mesh = Mesh1D({"ox": (0.001, 0.001), "si": (0.001, 0.01)})
    gate = Contact("g", InterfaceRef("ox", None), "gate")
    body = Contact("b", InterfaceRef("si", None), "ohmic")
    bad = [
        Contact("g", LayerRef((1, 0)), "gate"),  # no GDS layers in 1D
        Contact("g", InterfaceRef("ox", "si"), "gate"),  # an interior interface
        Contact("g", InterfaceRef("nope", None), "gate"),  # not a layer
        Contact("g", InterfaceRef("ox", None), "ohmic"),  # ohmic contact on oxide
    ]
    for contact in bad:
        with pytest.raises(ValueError):
            create_1d_stack("bad", layers, [contact, body], mesh, CONSTANTS)
    with pytest.raises(ValueError):  # gate contact on silicon
        create_1d_stack("bad", layers, [gate, Contact("b", InterfaceRef("si", None), "gate")], mesh, CONSTANTS)
    with pytest.raises(ValueError):  # both contacts at the same end
        create_1d_stack("bad", layers, [gate, Contact("b", InterfaceRef("ox", None), "gate")], mesh, CONSTANTS)
    with pytest.raises(ValueError):  # a 1D stack has exactly two contacts
        create_1d_stack("bad", layers, [gate], mesh, CONSTANTS)
    with pytest.raises(ValueError):  # missing spacing
        create_1d_stack("bad", layers, [gate, body], Mesh1D({"ox": (0.001, 0.001)}), CONSTANTS)
    with pytest.raises(ValueError):  # silicon without doping
        create_1d_stack(
            "bad", [layers[0], Layer1D("si", "silicon", 0.1, None)], [gate, body], mesh, CONSTANTS
        )
    assert not devsim.get_device_list()


def test_nonconvergence_raises() -> None:
    """A solve that does not converge raises; it is never silently accepted."""
    dd = _poly_stack("noconv", NFET)
    dd.set_bias("gate", -1.0)
    with pytest.raises(ConvergenceError) as info:
        dd.solve_dc(relative_error=1e-30, absolute_error=1e-30, maximum_iterations=1)
    assert info.value.info["converged"] is False
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# Parallel plate: C = eps_ox eps0 / t, by charge and by small-signal AC
# --------------------------------------------------------------------------------------------
def _parallel_plate(name: str, *, circuit: bool):
    t_um = 0.0152
    layers = [Layer1D("ox_a", "oxide", t_um / 2, None), Layer1D("ox_b", "oxide", t_um / 2, None)]
    contacts = [
        Contact("top", InterfaceRef("ox_a", None), "gate", circuit=circuit),
        Contact("bot", InterfaceRef("ox_b", None), "gate"),
    ]
    mesh = Mesh1D({"ox_a": (0.001, 0.0005), "ox_b": (0.0005, 0.002)})
    dd = create_1d_stack(name, layers, contacts, mesh, CONSTANTS)
    c_analytic = CONSTANTS.eps_ox * CONSTANTS.eps0_f_per_cm / (t_um * UM)  # F/cm^2
    return dd, c_analytic


def test_parallel_plate_charge() -> None:
    dd, c_analytic = _parallel_plate("pp_q", circuit=False)
    q0 = dd.contact_charge("top")
    dd.set_bias("top", 1.0)
    dd.solve_dc()
    q1, q1_bot = dd.contact_charge("top"), dd.contact_charge("bot")
    c = (q1 - q0) / 1.0
    print(f"parallel plate: dQ/dV {c!r} F/cm^2, analytic {c_analytic!r} F/cm^2")
    assert abs(c / c_analytic - 1) <= 1e-6
    assert abs(q1 + q1_bot) <= 1e-9 * abs(q1)  # charge neutral
    dd.delete_device()


def test_parallel_plate_ac() -> None:
    dd, c_analytic = _parallel_plate("pp_ac", circuit=True)
    dd.set_bias("top", 0.5)
    dd.solve_dc()
    for freq in (1.0, 1e6):
        y = dd.ac_admittance("top", freq)  # A/cm^2 per V, I = j w C V
        c = y.imag / (2 * math.pi * freq)
        print(f"parallel plate AC {freq:g} Hz: C {c!r} F/cm^2, analytic {c_analytic!r}")
        assert abs(c / c_analytic - 1) <= 1e-6
        assert abs(y.real) <= 1e-9 * abs(y.imag)
    dd.delete_device()


# --------------------------------------------------------------------------------------------
# Flat band
# --------------------------------------------------------------------------------------------
def flatband(dd, charge, v0: float, v1: float, tol: float = 1e-7) -> float:
    """The gate bias where ``charge(dd)`` is 0, by the secant method from ``v0``, ``v1``.

    The device is ramped to each new bias in steps of at most 0.1 V; every solve must converge.
    """
    state = {"v": 0.0}

    def q_at(v: float) -> float:
        n = max(1, math.ceil(abs(v - state["v"]) / 0.1))
        for k in range(1, n + 1):
            dd.set_bias("gate", state["v"] + (v - state["v"]) * k / n)
            dd.solve_dc()
        state["v"] = v
        return charge(dd)

    q0, q1 = q_at(v0), q_at(v1)
    for _ in range(50):
        v2 = v1 - q1 * (v1 - v0) / (q1 - q0)
        if abs(v2 - v1) < tol:
            return v2
        v0, q0, v1, q1 = v1, q1, v2, q_at(v2)
    raise AssertionError("flat band search did not converge")


def _poly_gate_charge(dd) -> float:
    """Gate charge (C/cm^2): the poly region's space charge plus its contact's charge."""
    return dd.electrode_charge("gate", ["gate_poly"])


@pytest.mark.parametrize("case", [NFET, PFET], ids=["nfet", "pfet"])
@pytest.mark.parametrize("pvg", [-2.0, 0.0], ids=["accumulation", "depletion"])
def test_gate_charge_neutral_and_gauss(case, pvg) -> None:
    """Away from flat band, Q_gate = -(Q_body) = eps_ox E_ox (C/cm^2).

    At polarity * Vg = -2 V (accumulation) and 0 V (depletion; |Vg - Vfb| ~ 1 V). Strong
    inversion is not used: with only a body contact the DC ramp there does not converge
    (it raises ConvergenceError), a Task 8 matter.

    Q_gate is the gate electrode charge (poly space charge + gate contact charge), Q_body
    the body's (body space charge + body contact charge); the oxide holds no charge, so the
    device is neutral and, by Gauss, both equal the oxide D field (uniform in 1D). Task 8's
    C-V uses this gate charge.
    """
    vg = pvg if case["body"] == "acceptor" else 0.0 - pvg  # 0.0 - 0.0 is +0.0
    dd = _poly_stack(f"neutral_{case['body']}_{vg:+.0f}", case)
    n = 20
    for k in range(1, n + 1):
        dd.set_bias("gate", vg * k / n)
        dd.solve_dc()
    q_gate = dd.electrode_charge("gate", ["gate_poly"])
    q_body = dd.electrode_charge("body", ["body"])
    d_ox = devsim.get_edge_model_values(device=dd.device, region="oxide", name="PotentialEdgeFlux")
    cox = CONSTANTS.eps_ox * CONSTANTS.eps0_f_per_cm / (case["tox_um"] * UM)
    print(f"{case['body']} Vg {vg}: Q_gate {q_gate!r}, Q_body {q_body!r}, D_ox {d_ox[0]!r} C/cm^2")
    assert abs(q_gate) > cox * 0.1  # at least a 0.1 V oxide drop: well away from flat band
    assert abs(q_gate + q_body) <= 1e-6 * abs(q_gate)
    assert all(abs(d - q_gate) <= 1e-6 * abs(q_gate) for d in d_ox)
    dd.delete_device()


@pytest.mark.parametrize("case", [NFET, PFET], ids=["nfet", "pfet"])
def test_poly_gate_flatband(case) -> None:
    """Vfb = -Vt ln(NGATE N_A / n_i^2) (n+ gate on p body), +Vt ln(...) for the pfet mirror."""
    vt, ni = CONSTANTS.vt_v, CONSTANTS.n_i_cm3
    sign = -1.0 if case["body"] == "acceptor" else 1.0
    expected = sign * vt * math.log(case["gate"][1] * N_BODY / ni**2)
    assert round(expected, 3) == case["vfb"]

    dd = _poly_stack(f"vfb_{case['body']}", case)
    # seeds independent of the answer: 0 V and 0.5 V towards the expected side
    vfb = flatband(dd, _poly_gate_charge, 0.0, sign * 0.5)
    # at flat band the oxide field (the gate charge by Gauss) is zero too
    d_ox = devsim.get_edge_model_values(device=dd.device, region="oxide", name="PotentialEdgeFlux")
    print(f"{case['body']} body: Vfb {vfb!r} V, Boltzmann {expected!r} V, D_ox {d_ox[0]!r}")
    assert abs(vfb - expected) <= 1e-3
    assert max(abs(d) for d in d_ox) <= 1e-6 * CONSTANTS.eps_ox * CONSTANTS.eps0_f_per_cm / (
        case["tox_um"] * UM
    )
    dd.delete_device()


def test_package_exports() -> None:
    import gplugins.devsim as gd
    from gplugins.devsim import contacts, materials, stack1d

    assert gd.create_1d_stack is stack1d.create_1d_stack
    assert gd.Mesh1D is stack1d.Mesh1D and gd.Layer1D is stack1d.Layer1D
    assert gd.Contact is contacts.Contact and gd.InterfaceRef is contacts.InterfaceRef
    assert gd.LayerRef is contacts.LayerRef
    assert gd.Constants is materials.Constants is Constants
    assert gd.ConvergenceError is ConvergenceError
    # one constants table: materials re-exports constants.Constants
    from gplugins.devsim import constants

    assert materials.Constants is constants.Constants and materials.CONSTANTS is constants.CONSTANTS
