# SPDX-License-Identifier: MIT
"""Tests of the analytic doping profiles and of the ``gplugins.process`` implant profiles."""

from __future__ import annotations

import devsim
import numpy as np
import pytest
from devsim.python_packages import model_create
from gdsfactory.technology.processes import (
    DopingConstant,
    ImplantGaussian,
    ImplantPhysical,
)

from gplugins.devsim import doping
from gplugins.devsim.constants import CONSTANTS
from gplugins.process import diffusion, implant_tables

# --- gplugins.process.diffusion ------------------------------------------------------------

# phosphorus 100 keV: Rp = 0.1227 um, straggle = 0.0442 um (implant_tables polynomials)
_IMPLANT = dict(dopant="phosphorus", dose=1e13, E=100.0)
_Z = np.linspace(0.0, 0.6, 6001)  # um


def test_diffused_equals_gaussian_at_t0() -> None:
    """With no diffusion time the diffused profile is the as-implanted Gaussian."""
    diffused = diffusion.silicon_diffused_gaussian_profile(
        **_IMPLANT, t=0.0, T=1000.0, z=_Z
    )
    implanted = implant_tables.silicon_gaussian_profile(**_IMPLANT, z=_Z)
    np.testing.assert_allclose(diffused, implanted, rtol=1e-12, atol=0.0)


def test_diffused_peak_at_t0() -> None:
    """The prefactor alone: at z = Rp the exponent is 0, so only the prefactor is compared.

    The prefactor must be dose / (sqrt(2 pi) sigma) with sigma converted from um to cm.
    """
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    sigma_cm = implant_tables.straggle_in_silicon["phosphorus"](100.0) * 1e-4
    (peak,) = diffusion.silicon_diffused_gaussian_profile(
        **_IMPLANT, t=0.0, T=1000.0, z=np.array([rp])
    )
    assert peak == pytest.approx(1e13 / (np.sqrt(2 * np.pi) * sigma_cm), rel=1e-12)


def test_diffused_shape_at_t0() -> None:
    """The exponent alone: C(z)/C(Rp) must be exp(-(z - Rp)^2 / (2 sigma^2))."""
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    sigma = implant_tables.straggle_in_silicon["phosphorus"](100.0)
    z = rp + sigma * np.array([0.0, 1.0, 2.0])
    c = diffusion.silicon_diffused_gaussian_profile(**_IMPLANT, t=0.0, T=1000.0, z=z)
    np.testing.assert_allclose(c / c[0], np.exp(-np.array([0.0, 0.5, 2.0])), rtol=1e-12)


def test_diffused_dose_and_width_after_anneal() -> None:
    """After an anneal the dose is unchanged and the variance grows by 2 D t (um^2)."""
    t, temp = 600.0, 1000.0
    sigma = implant_tables.straggle_in_silicon["phosphorus"](100.0)
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    dt_um2 = diffusion.D("phosphorus", temp) * 1e8 * t
    s = np.sqrt(sigma**2 + 2 * dt_um2)
    z = np.linspace(
        rp - 12 * s, rp + 12 * s, 20001
    )  # the whole Gaussian, not just the wafer
    c = diffusion.silicon_diffused_gaussian_profile(**_IMPLANT, t=t, T=temp, z=z)
    dose = np.trapezoid(c, z * 1e-4)
    var = np.trapezoid(c * (z - rp) ** 2, z) / np.trapezoid(c, z)
    assert dose == pytest.approx(1e13, rel=1e-9)
    assert var == pytest.approx(sigma**2 + 2 * dt_um2, rel=1e-9)


# --- gplugins.devsim.constants -------------------------------------------------------------


def test_constants_derived_values() -> None:
    assert CONSTANTS.vt_v == pytest.approx(25.6926e-3, abs=1e-7)
    assert CONSTANTS.n_i_cm3 == pytest.approx(8.266e9, rel=1e-4)


# --- gplugins.devsim.doping ----------------------------------------------------------------

UM = 1e-4  # cm


@pytest.fixture
def clean_devsim():
    """Delete the devices (and their meshes, which share the device name) a test made."""
    yield
    for device in list(devsim.get_device_list()):
        devsim.delete_device(device=device)
        devsim.delete_mesh(mesh=device)


def _device_1d(name: str, length_um: float = 1.0, ps_um: float = 1e-3) -> str:
    """A 1D silicon device 0..``length_um`` with mesh spacing ``ps_um`` (DEVSIM-native mesh)."""
    devsim.create_1d_mesh(mesh=name)
    devsim.add_1d_mesh_line(mesh=name, pos=0.0, ps=ps_um * UM, tag="top")
    devsim.add_1d_mesh_line(mesh=name, pos=length_um * UM, ps=ps_um * UM, tag="bot")
    devsim.add_1d_region(mesh=name, material="Si", region="si", tag1="top", tag2="bot")
    devsim.add_1d_contact(mesh=name, name="top", tag="top", material="metal")
    devsim.add_1d_contact(mesh=name, name="bot", tag="bot", material="metal")
    devsim.finalize_mesh(mesh=name)
    devsim.create_device(mesh=name, device=name)
    return name


def _device_2d(
    name: str, x_um=(-1.0, 2.0), y_um=(-1.0, 0.0), ps_um: float = 0.01
) -> str:
    """A 2D silicon rectangle (DEVSIM-native mesh), with its surface at y = ``y_um[1]``."""
    devsim.create_2d_mesh(mesh=name)
    for pos in x_um:
        devsim.add_2d_mesh_line(mesh=name, dir="x", pos=pos * UM, ps=ps_um * UM)
    for pos in y_um:
        devsim.add_2d_mesh_line(mesh=name, dir="y", pos=pos * UM, ps=ps_um * UM)
    devsim.add_2d_region(
        mesh=name,
        material="Si",
        region="si",
        xl=x_um[0] * UM,
        xh=x_um[1] * UM,
        yl=y_um[0] * UM,
        yh=y_um[1] * UM,
    )
    devsim.finalize_mesh(mesh=name)
    devsim.create_device(mesh=name, device=name)
    return name


def _values(device: str, name: str, region: str = "si") -> np.ndarray:
    return np.array(
        devsim.get_node_model_values(device=device, region=region, name=name)
    )


def _integral(device: str, name: str) -> float:
    """Integral of a node model over the region: cm^-2 in 1D, cm^-1 in 2D."""
    return float(np.sum(_values(device, name) * _values(device, "NodeVolume")))


def test_profiles_hand_values() -> None:
    g = doping.gaussian(peak=1e18, depth=0.2, straggle=0.05)
    assert g(0.0, 0.2) == pytest.approx(1e18, rel=1e-15)
    assert g(0.0, 0.25) == pytest.approx(1e18 * np.exp(-0.5), rel=1e-14)
    e = doping.erfc(surface=1e20, length=0.1)
    assert e(0.0, 0.0) == pytest.approx(1e20, rel=1e-15)
    assert e(0.0, 0.1) == pytest.approx(
        1e20 * 0.15729920705028513, rel=1e-14
    )  # erfc(1)
    c = doping.constant(1.7e17, ymin=0.1, ymax=0.3)
    np.testing.assert_array_equal(
        c(0.0, np.array([0.05, 0.1, 0.2, 0.3, 0.35])), [0, 1.7e17, 1.7e17, 1.7e17, 0]
    )
    lat = doping.lateral(straggle=0.05, xmin=0.0, xmax=1.0)
    assert lat(0.0, 0.0) == pytest.approx(0.5, rel=1e-12)  # at a mask edge
    assert lat(0.5, 0.0) == pytest.approx(1.0, rel=1e-12)
    assert lat(-1.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # sums and products of profiles are profiles
    both = g + c
    assert both(0.0, 0.2) == pytest.approx(1e18 + 1.7e17, rel=1e-15)
    assert (g * lat)(0.0, 0.2) == pytest.approx(0.5e18, rel=1e-12)


def test_gaussian_from_dose() -> None:
    g = doping.gaussian_from_dose(dose=1e13, depth=0.3, straggle=0.05)
    assert g.peak == pytest.approx(1e13 / (np.sqrt(2 * np.pi) * 0.05e-4), rel=1e-15)


@pytest.mark.parametrize(
    "profile",
    [
        doping.gaussian(peak=3e18, depth=0.15, straggle=0.04),
        doping.erfc(surface=1e20, length=0.08),
        doping.constant(1.7e17, ymin=0.1, ymax=0.4),
        doping.gaussian(peak=1e18, depth=0.2, straggle=0.05)
        * doping.lateral(0.05, xmin=0.0, xmax=1.0),
        doping.lateral(0.0, xmin=0.0) * doping.constant(1e16),
    ],
    ids=["gaussian", "erfc", "box", "gaussian-lateral", "constant-sharp-edge"],
)
def test_devsim_expression_equals_python(clean_devsim, profile) -> None:
    """The DEVSIM node model of a profile equals the profile evaluated at the node positions.

    The 2D device has its surface at y = 0 and the silicon at y < 0, so depth = -y.
    """
    d = _device_2d("d_expr", ps_um=0.05)
    doping.assign_doping(
        d, "si", donors=[profile], depth_axis="y", surface_um=0.0, depth_sign=-1
    )
    x_um, y_um = _values(d, "x") / UM, _values(d, "y") / UM
    expected = profile(x_um, -y_um)
    np.testing.assert_allclose(
        _values(d, "Donors"), expected, rtol=1e-12, atol=1e-30 * expected.max()
    )


def test_net_doping_is_donors_minus_acceptors(clean_devsim) -> None:
    d = _device_1d("d_net", ps_um=0.01)
    doping.assign_doping(
        d,
        "si",
        donors=[doping.erfc(surface=1e19, length=0.1)],
        acceptors=[doping.constant(1e17), doping.gaussian(1e17, 0.5, 0.1)],
        depth_axis="x",
    )
    np.testing.assert_allclose(
        _values(d, "NetDoping"),
        _values(d, "Donors") - _values(d, "Acceptors"),
        rtol=1e-12,
    )
    assert _values(d, "Acceptors").min() > 0.0


def test_dose_conserved(clean_devsim) -> None:
    """The integrated dose of the assigned 1D doping equals the input dose within 0.5 %."""
    cases = [
        (
            "gaussian_from_dose",
            "acceptor",
            doping.gaussian_from_dose(dose=1e13, depth=0.3, straggle=0.05),
            1e13,
        ),
        (
            "erfc",
            "donor",
            doping.erfc(surface=1e20, length=0.08),
            1e20 * 0.08e-4 / np.sqrt(np.pi),
        ),
        (
            "box",
            "acceptor",
            doping.constant(1.7e17, ymin=0.2, ymax=0.6),
            1.7e17 * 0.4e-4,
        ),
        (
            "ImplantPhysical",
            *doping.profile_from_process(
                ImplantPhysical(
                    name="pw", layer=(1, 0), ion="B", energy=50.0, dose=2e13
                )
            ),
            2e13,
        ),
        (
            "ImplantGaussian",
            *doping.profile_from_process(
                ImplantGaussian(
                    name="nw",
                    layer=(1, 0),
                    ion="P",
                    peak_conc=1e17,
                    range=0.4,
                    vertical_straggle=0.1,
                )
            ),
            1e17 * np.sqrt(2 * np.pi) * 0.1e-4,
        ),
        (
            "DopingConstant",
            *doping.profile_from_process(
                DopingConstant(
                    name="sub",
                    layer=None,
                    ion="B",
                    peak_conc=1e16,
                    zmin=0.0,
                    zmax=None,
                    into_materials=["si"],
                )
            ),
            1e16 * 1e-4,
        ),
    ]
    d = _device_1d("d_dose", length_um=1.0, ps_um=1e-3)
    errors = {}
    for name, kind, profile, dose in cases:
        doping.assign_doping(d, "si", **{f"{kind}s": [profile]}, depth_axis="x")
        got = _integral(d, "Donors" if kind == "donor" else "Acceptors")
        errors[name] = got / dose - 1
    print("relative dose errors:", errors)
    assert all(abs(e) < 5e-3 for e in errors.values()), errors


def test_dose_conserved_2d_lateral(clean_devsim) -> None:
    """A laterally straggled implant through a 1 um opening integrates to dose x opening width."""
    profile = doping.gaussian_from_dose(
        dose=1e13, depth=0.3, straggle=0.05
    ) * doping.lateral(0.08, xmin=0.0, xmax=1.0)
    d = _device_2d("d_dose2", x_um=(-1.0, 2.0), y_um=(-1.0, 0.0), ps_um=0.01)
    doping.assign_doping(
        d, "si", acceptors=[profile], depth_axis="y", surface_um=0.0, depth_sign=-1
    )
    got = _integral(d, "Acceptors")  # cm^-1
    expected = 1e13 * 1.0 * UM
    print("2D relative dose error:", got / expected - 1)
    assert got == pytest.approx(expected, rel=5e-3)


def test_profile_from_process_types() -> None:
    kind, _ = doping.profile_from_process(
        ImplantPhysical(name="n", layer=(1, 0), ion="As", energy=100.0, dose=1e15)
    )
    assert kind == "donor"
    kind, _ = doping.profile_from_process(
        ImplantPhysical(name="p", layer=(1, 0), ion="boron", energy=10.0, dose=1e15)
    )
    assert kind == "acceptor"
    with pytest.raises(ValueError, match="ion"):
        doping.profile_from_process(
            ImplantPhysical(name="x", layer=(1, 0), ion="Xe", energy=10.0, dose=1e15)
        )
    with pytest.raises(ValueError, match="vertical_straggle"):
        doping.profile_from_process(
            ImplantGaussian(name="g", layer=(1, 0), ion="P", peak_conc=1e17, range=0.1)
        )


def _solve_equilibrium_poisson(device: str, region: str, c=CONSTANTS) -> np.ndarray:
    """Equilibrium (Boltzmann) Poisson on ``region``, with the plugin constants and no contacts.

    No contact or boundary equation is set, so both ends are zero-field (Neumann) and the
    potential follows from the doping and the constants alone. Potential is relative to the
    intrinsic level.
    """
    for name, value in (
        ("Permittivity", c.eps_si * c.eps0_f_per_cm),
        ("ElectronCharge", c.q_c),
        ("n_i", c.n_i_cm3),
        ("V_t", c.vt_v),
    ):
        devsim.set_parameter(device=device, region=region, name=name, value=value)
    model_create.CreateSolution(device, region, "Potential")
    model_create.CreateNodeModel(
        device, region, "IntrinsicElectrons", "n_i*exp(Potential/V_t)"
    )
    model_create.CreateNodeModelDerivative(
        device, region, "IntrinsicElectrons", "n_i*exp(Potential/V_t)", "Potential"
    )
    model_create.CreateNodeModel(
        device, region, "IntrinsicHoles", "n_i^2/IntrinsicElectrons"
    )
    model_create.CreateNodeModelDerivative(
        device, region, "IntrinsicHoles", "n_i^2/IntrinsicElectrons", "Potential"
    )
    charge = "-ElectronCharge*(IntrinsicHoles - IntrinsicElectrons + NetDoping)"
    model_create.CreateNodeModel(device, region, "PotentialIntrinsicCharge", charge)
    model_create.CreateNodeModelDerivative(
        device, region, "PotentialIntrinsicCharge", charge, "Potential"
    )
    field = "(Potential@n0 - Potential@n1)*EdgeInverseLength"
    model_create.CreateEdgeModel(device, region, "ElectricField", field)
    model_create.CreateEdgeModelDerivatives(
        device, region, "ElectricField", field, "Potential"
    )
    model_create.CreateEdgeModel(
        device, region, "PotentialEdgeFlux", "Permittivity*ElectricField"
    )
    model_create.CreateEdgeModelDerivatives(
        device, region, "PotentialEdgeFlux", "Permittivity*ElectricField", "Potential"
    )
    devsim.equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model="PotentialIntrinsicCharge",
        edge_model="PotentialEdgeFlux",
        variable_update="log_damp",
    )
    devsim.solve(
        type="dc", absolute_error=1e-12, relative_error=1e-14, maximum_iterations=100
    )
    return _values(device, "Potential", region)


def test_uniform_bulk_potential(clean_devsim) -> None:
    """A uniform 1.7e17 acceptor region sits at the analytic bulk potential -Vt ln(N_A/n_i).

    The solve starts from Potential = 0; log_damp limits the Newton updates.
    """
    na = 1.7e17
    d = _device_1d("d_bulk", length_um=1.0, ps_um=0.01)
    doping.assign_doping(d, "si", acceptors=[doping.constant(na)], depth_axis="x")
    psi = _solve_equilibrium_poisson(d, "si")
    expected = -CONSTANTS.vt_v * np.log(na / CONSTANTS.n_i_cm3)
    err = np.abs(psi - expected).max()
    print(
        f"bulk potential: expected {expected!r} V, solved {psi.min()!r}..{psi.max()!r} V, max |err| {err!r} V"
    )
    assert err < 1e-4
