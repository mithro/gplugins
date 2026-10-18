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
from scipy import special

from gplugins.devsim import doping
from gplugins.devsim.constants import CONSTANTS
from gplugins.process import diffusion, implant_tables

# --- gplugins.process.diffusion ------------------------------------------------------------

# phosphorus 100 keV: Rp = 0.12445 um, straggle = 0.045268 um (implant_tables polynomials)
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
        # a point exactly on an edge gets half the concentration
        c(0.0, np.array([0.05, 0.1, 0.2, 0.3, 0.35])),
        [0, 0.85e17, 1.7e17, 0.85e17, 0],
    )
    lat = doping.lateral(straggle=0.05, xmin=0.0, xmax=1.0)
    assert lat(0.0, 0.0) == pytest.approx(0.5, rel=1e-12)  # at a mask edge
    assert lat(0.5, 0.0) == pytest.approx(1.0, rel=1e-12)
    assert lat(-1.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # sums and products of profiles are profiles
    both = g + c
    assert both(0.0, 0.2) == pytest.approx(1e18 + 1.7e17, rel=1e-15)
    assert (g * lat)(0.0, 0.2) == pytest.approx(0.5e18, rel=1e-12)


def _in_silicon(depth: float, straggle: float) -> float:
    """Fraction of a whole Gaussian at y >= 0: erfc(-Rp / (sqrt(2) sigma)) / 2."""
    return 0.5 * float(special.erfc(-depth / (np.sqrt(2) * straggle)))


def test_gaussian_from_dose() -> None:
    """By default the peak is renormalised so that the dose in silicon (y >= 0) is the dose."""
    g = doping.gaussian_from_dose(dose=1e13, depth=0.05, straggle=0.03)
    whole = 1e13 / (np.sqrt(2 * np.pi) * 0.03e-4)
    assert g.peak == pytest.approx(whole / _in_silicon(0.05, 0.03), rel=1e-15)
    assert g.dose_in_silicon == pytest.approx(1e13, rel=1e-14)
    clipped = doping.gaussian_from_dose(
        dose=1e13, depth=0.05, straggle=0.03, truncation="clip"
    )
    assert clipped.peak == pytest.approx(whole, rel=1e-15)
    assert clipped.dose == pytest.approx(1e13, rel=1e-14)
    with pytest.raises(ValueError, match="truncation"):
        doping.gaussian_from_dose(
            dose=1e13, depth=0.05, straggle=0.03, truncation="none"
        )


@pytest.mark.parametrize(
    "profile",
    [
        doping.gaussian(peak=3e18, depth=0.15, straggle=0.04),
        doping.erfc(surface=1e20, length=0.08),
        # box edges on the 0.05 um mesh lines: the edge nodes get half the concentration
        doping.constant(1.7e17, ymin=0.1, ymax=0.4),
        doping.gaussian(peak=1e18, depth=0.2, straggle=0.05)
        * doping.lateral(0.05, xmin=0.0, xmax=1.0),
        doping.lateral(0.05, xmin=0.2) * doping.gaussian(1e18, 0.3, 0.1),
        doping.lateral(0.05, xmax=0.5) * doping.constant(1e17),
        doping.lateral(0.0, xmin=0.0) * doping.constant(1e16),
        doping.lateral(0.0, xmax=0.5) * doping.constant(1e16, ymax=0.5),
        doping.gaussian(1e18, 0.3, 0.1)
        + doping.erfc(1e19, 0.1)
        + doping.constant(1e15),
    ],
    ids=[
        "gaussian",
        "erfc",
        "box",
        "gaussian-lateral",
        "lateral-xmin-only",
        "lateral-xmax-only",
        "sharp-xmin-only",
        "sharp-xmax-only",
        "sum",
    ],
)
@pytest.mark.parametrize(
    ("depth_sign", "surface_um"),
    [(-1, 0.0), (1, -1.0)],
    ids=["y-up-surface-0", "y-down-surface-minus1"],
)
def test_devsim_expression_equals_python(
    clean_devsim, profile, depth_sign, surface_um
) -> None:
    """The DEVSIM node model of a profile equals the profile evaluated at the node positions.

    The 2D device spans y = -1..0 um. With depth_sign=-1 and surface_um=0 the depth is -y; with
    depth_sign=+1 and surface_um=-1 it is y + 1.
    """
    d = _device_2d("d_expr", ps_um=0.05)
    doping.assign_doping(
        d,
        "si",
        donors=[profile],
        depth_axis="y",
        surface_um=surface_um,
        depth_sign=depth_sign,
    )
    # cm -> um as assign_doping does it (cm * 1e4)
    x_um, y_um = _values(d, "x") * 1e4, _values(d, "y") * 1e4
    depth = depth_sign * (y_um - surface_um)
    assert depth.min() == pytest.approx(
        0.0, abs=1e-12
    ) and depth.max() == pytest.approx(1.0)
    expected = profile(x_um, depth)
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


def _implant(ion: str, energy: float, dose: float) -> ImplantPhysical:
    return ImplantPhysical(
        name=f"{ion}{energy}", layer=(1, 0), ion=ion, energy=energy, dose=dose
    )


def _table_in_silicon(dopant: str, energy: float) -> float:
    rp = float(implant_tables.depth_in_silicon[dopant](energy))
    sigma = float(implant_tables.straggle_in_silicon[dopant](energy))
    return _in_silicon(rp, sigma)


def test_dose_conserved(clean_devsim) -> None:
    """The integrated dose of the assigned 1D doping equals the expected dose.

    A dose-specified implant (``gaussian_from_dose``, ``ImplantPhysical``) is renormalised by
    default, so its dose in silicon is the stated dose; with ``truncation="clip"`` it is the
    stated dose times the in-silicon fraction. A peak-specified Gaussian (``ImplantGaussian``)
    keeps its peak, so its dose is the analytic truncated dose. The B 15 keV and P 50 keV cases
    are the shallow implants of gdsfactory's gpdk process. The node-volume integral on a 0.5 nm
    1D mesh is checked to rel 1e-5 (well inside the brief's 0.5 %).
    """
    ig = ImplantGaussian(
        name="nw",
        layer=(1, 0),
        ion="P",
        peak_conc=1e17,
        range=0.4,
        vertical_straggle=0.1,
    )
    dc = DopingConstant(
        name="sub",
        layer=None,
        ion="B",
        peak_conc=1e16,
        zmin=-0.6,
        zmax=0.0,
        into_materials=["si"],
    )
    cases = [
        (
            "gaussian_from_dose",
            "acceptor",
            doping.gaussian_from_dose(1e13, 0.3, 0.05),
            1e13,
        ),
        (
            "gaussian_from_dose shallow",
            "acceptor",
            doping.gaussian_from_dose(1e13, 0.05, 0.03),
            1e13,
        ),
        (
            "gaussian_from_dose shallow clip",
            "acceptor",
            doping.gaussian_from_dose(1e13, 0.05, 0.03, truncation="clip"),
            1e13 * _in_silicon(0.05, 0.03),
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
            "ImplantPhysical B 50 keV",
            *doping.profile_from_process(_implant("B", 50.0, 2e13)),
            2e13,
        ),
        (
            "ImplantPhysical B 15 keV (gpdk)",
            *doping.profile_from_process(_implant("B", 15.0, 1e12)),
            1e12,
        ),
        (
            "ImplantPhysical P 50 keV (gpdk)",
            *doping.profile_from_process(_implant("P", 50.0, 1e12)),
            1e12,
        ),
        (
            "ImplantPhysical B 15 keV clip",
            *doping.profile_from_process(_implant("B", 15.0, 1e12), truncation="clip"),
            1e12 * _table_in_silicon("boron", 15.0),
        ),
        (
            "ImplantGaussian (peak-specified)",
            *doping.profile_from_process(ig),
            1e17 * np.sqrt(2 * np.pi) * 0.1e-4 * _in_silicon(0.4, 0.1),
        ),
        (
            "DopingConstant",
            *doping.profile_from_process(dc, surface_z_um=0.0),
            1e16 * 0.6e-4,
        ),
    ]
    d = _device_1d("d_dose", length_um=1.0, ps_um=5e-4)
    errors = {}
    for name, kind, profile, dose in cases:
        doping.assign_doping(d, "si", **{f"{kind}s": [profile]}, depth_axis="x")
        got = _integral(d, "Donors" if kind == "donor" else "Acceptors")
        errors[name] = float(got / dose - 1)
    print("relative dose errors:", errors)
    print("in-silicon fraction B 15 keV:", _table_in_silicon("boron", 15.0))
    print("in-silicon fraction P 50 keV:", _table_in_silicon("phosphorus", 50.0))
    assert all(abs(e) < 1e-5 for e in errors.values()), errors


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


def _doping_constant(zmin: float, zmax: float | None) -> DopingConstant:
    return DopingConstant(
        name="c",
        layer=None,
        ion="B",
        peak_conc=1e17,
        zmin=zmin,
        zmax=zmax,
        into_materials=["si"],
    )


def test_doping_constant_frame() -> None:
    """DopingConstant zmin/zmax are absolute z-up coordinates; depth = surface_z_um - z."""
    # the gdsfactory-style substrate box zmin=-1, zmax=0 under a surface at z = 0
    kind, p = doping.profile_from_process(_doping_constant(-1.0, 0.0), surface_z_um=0.0)
    assert kind == "acceptor"
    # a box reaching the surface has no upper edge there (ymin None), so the surface nodes get
    # the full concentration; the edge at depth 1 gets half
    assert (p.ymin, p.ymax) == (None, 1.0)
    np.testing.assert_array_equal(
        p(0.0, np.array([0.0, 0.5, 1.0, 1.01])), [1e17, 1e17, 0.5e17, 0]
    )
    # zmax=None means the surface (the top of the doped material)
    _, p = doping.profile_from_process(_doping_constant(-0.5, None), surface_z_um=0.2)
    assert (p.ymin, p.ymax) == (None, pytest.approx(0.7))
    # a box below the surface
    _, p = doping.profile_from_process(_doping_constant(-0.6, -0.2), surface_z_um=0.0)
    assert (p.ymin, p.ymax) == (0.2, 0.6)
    # a box straddling the surface is clipped to it
    _, p = doping.profile_from_process(_doping_constant(-0.5, 0.3), surface_z_um=0.0)
    assert (p.ymin, p.ymax) == (None, 0.5)


@pytest.mark.parametrize(
    ("zmin", "zmax", "surface_z_um", "match"),
    [
        (-1.0, 0.0, None, "surface_z_um"),
        (0.1, 0.5, 0.0, "above the surface"),
        (0.0, 0.5, 0.0, "above the surface"),
        (-0.2, -0.5, 0.0, "empty"),
        (-0.2, -0.2, 0.0, "empty"),
    ],
    ids=[
        "no-surface",
        "above",
        "touching-from-above",
        "zmax-below-zmin",
        "zero-thickness",
    ],
)
def test_doping_constant_frame_errors(zmin, zmax, surface_z_um, match) -> None:
    """A DopingConstant window that would give no doping raises instead of silently giving 0."""
    with pytest.raises(ValueError, match=match):
        doping.profile_from_process(
            _doping_constant(zmin, zmax), surface_z_um=surface_z_um
        )


def test_erfc_from_junction() -> None:
    """erfc_from_junction puts the metallurgical junction (N = background) at depth xj."""
    e = doping.erfc_from_junction(surface=1e20, xj=0.2, background=1e16)
    assert e.length == pytest.approx(0.2 / float(special.erfcinv(1e-4)), rel=1e-15)
    assert e(0.0, 0.2) == pytest.approx(1e16, rel=1e-10)
    assert e(0.0, 0.0) == pytest.approx(1e20, rel=1e-15)
    for bad in (1e20, 2e20, 0.0, -1.0):
        with pytest.raises(ValueError, match="background"):
            doping.erfc_from_junction(surface=1e20, xj=0.2, background=bad)


# the two frames of the 2D test device (y = -1..0 um): depth = -y and depth = y + 1
_FRAMES_2D = [(-1, 0.0), (1, -1.0)]


@pytest.mark.parametrize(
    ("depth_sign", "surface_um"), _FRAMES_2D, ids=["y-up", "y-down"]
)
def test_box_on_mesh_lines_dose_2d(clean_devsim, depth_sign, surface_um) -> None:
    """A box whose edges lie on mesh lines integrates to its nominal dose, in either frame."""
    d = _device_2d("d_box2", ps_um=0.05)
    doping.assign_doping(
        d,
        "si",
        acceptors=[doping.constant(1.0, ymin=0.1, ymax=0.4)],
        depth_axis="y",
        surface_um=surface_um,
        depth_sign=depth_sign,
    )
    expected = 1.0 * 0.3 * 3.0 * UM * UM  # conc x depth window x width (cm^2)
    got = _integral(d, "Acceptors")
    print(
        f"2D on-mesh box, frame {depth_sign:+d}/{surface_um}: rel error {got / expected - 1!r}"
    )
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    ("depth_sign", "surface_um"), [(1, 0.0), (-1, 1.0)], ids=["x-down", "x-up"]
)
def test_box_on_mesh_lines_dose_1d(clean_devsim, depth_sign, surface_um) -> None:
    """The same in 1D (0..1 um, 0.05 um mesh), with depth = x and depth = 1 - x."""
    d = _device_1d("d_box1", length_um=1.0, ps_um=0.05)
    doping.assign_doping(
        d,
        "si",
        acceptors=[doping.constant(1.0, ymin=0.1, ymax=0.4)],
        depth_axis="x",
        surface_um=surface_um,
        depth_sign=depth_sign,
    )
    expected = 1.0 * 0.3 * UM
    got = _integral(d, "Acceptors")
    print(
        f"1D on-mesh box, frame {depth_sign:+d}/{surface_um}: rel error {got / expected - 1!r}"
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_frames_give_same_node_values(clean_devsim) -> None:
    """Both 2D frames give the same value at every (x, depth), with sharp edges on mesh lines."""
    profile = doping.constant(1.7e17, ymin=0.1, ymax=0.4) * doping.lateral(
        0.0, xmin=0.0, xmax=1.0
    ) + doping.gaussian(1e18, 0.3, 0.1) * doping.lateral(0.0, xmax=0.5)
    by_frame = []
    for i, (depth_sign, surface_um) in enumerate(_FRAMES_2D):
        d = _device_2d(f"d_frame{i}", ps_um=0.05)
        doping.assign_doping(
            d,
            "si",
            donors=[profile],
            depth_axis="y",
            surface_um=surface_um,
            depth_sign=depth_sign,
        )
        depth = depth_sign * (_values(d, "y") * 1e4 - surface_um)
        keys = zip(np.round(_values(d, "x") * 1e4, 9), np.round(depth, 9) + 0.0)
        by_frame.append(dict(zip(keys, _values(d, "Donors"))))
    assert by_frame[0].keys() == by_frame[1].keys()
    a = np.array([by_frame[0][k] for k in sorted(by_frame[0])])
    b = np.array([by_frame[1][k] for k in sorted(by_frame[0])])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=0.0)
    assert (a == 0.5 * 1.7e17).any()  # some nodes do lie on an edge


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
