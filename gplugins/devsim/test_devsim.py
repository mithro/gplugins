"""DEVSIM-only tests of gplugins.devsim (no tidy3d; see test_devsim_tidy3d.py for the mode solves).

The PIN-waveguide expectations are the gplugins v0.8.0 record of gf180-tcad
(https://github.com/mithro/gf180-tcad), file ``results/plugin-history/v0.8.0.json`` at commit
3c386f95ff2955be6367bd5ec7c66298b812a4fe: v0.8.0 on its own Sept-2023 stack (devsim 2.6.3,
Python 3.11). The bias sequence and the depletion-width definition are those of that record's
driver, ``experiments/plugin-history/checks_v080.py`` (``ramp_pin``, ``pin_bias_point``) and
``experiments/plugin-history/harness.py`` (``depletion_width``, ``contact_currents``).
"""

from __future__ import annotations

import math
import os

import devsim
import pytest

from gplugins.devsim import get_simulation_xsection

PIN_DEVICE = "MyDevice"  # hard-coded in PINWaveguide
PIN_VOLTAGES = (0.0, -1.0)
CM = 1e-2

# Depletion width (um) on the mesh row nearest the rib's vertical centre, from v0.8.0.json
# ("pin" -> "bias" -> "depletion_width_um").
V080_DEPLETION_WIDTH_UM = {
    0.0: 0.06898074973467581,
    -1.0: 0.13782918124914934,
}
DEPLETION_WIDTH_RTOL = 1e-6

# The PIN contact currents are not compared with v0.8.0: there they are ~1e-15 A/cm, nonzero at
# 0 V, and left + right != 0 at either bias (v0.8.0.json "pin" -> "bias" -> "currents"; see
# gf180-tcad docs/limitations/plugin-history.md, "All recorded currents are below the solver's
# current floor"). They are solver noise. This floor checks that they are still noise on the
# latest stack and that no real current appeared.
# Measured on the latest stack (gf180-tcad TCAD image: devsim 2.11.1, Python 3.13.16), total
# (electron + hole) A/cm, bit-identical to v0.8.0.json:
#   0 V:  left 1.778368790494735e-15,  right 4.4879986596680306e-15
#   -1 V: left -1.0420928010546398e-15, right 4.895919713614902e-15
# The depletion widths are bit-identical to v0.8.0 too (difference 0), well inside the 1e-6.
CURRENT_FLOOR_A_PER_CM = 1e-13


def depletion_width(x, y, n, p, net_doping, *, y_target: float):
    """``(width, y_cut, depleted_nodes)`` along the mesh row nearest ``y_target``.

    Same definition as gf180-tcad ``harness.depletion_width``: the row is the node y value
    closest to ``y_target`` (the smaller one on a tie); a node is depleted where both carrier
    densities are below 1e-2 |NetDoping|; the width is the x span of the depleted nodes of the
    row, in the units of ``x`` (0 if none).
    """
    y_cut = min(sorted(set(y)), key=lambda v: abs(v - y_target))
    row = sorted(
        {
            xi: (ni, pi, di)
            for xi, yi, ni, pi, di in zip(x, y, n, p, net_doping)
            if yi == y_cut
        }.items()
    )
    depleted = [
        xi for xi, (ni, pi, di) in row if ni < 1e-2 * abs(di) and pi < 1e-2 * abs(di)
    ]
    width = max(depleted) - min(depleted) if depleted else 0.0
    return width, y_cut, len(depleted)


def contact_currents(device: str, contacts) -> dict[str, dict[str, float]]:
    """Electron, hole and total current of each contact (A/cm in 2D)."""
    out = {}
    for contact in contacts:
        e = devsim.get_contact_current(
            device=device, contact=contact, equation="ElectronContinuityEquation"
        )
        h = devsim.get_contact_current(
            device=device, contact=contact, equation="HoleContinuityEquation"
        )
        out[contact] = {"electron": e, "hole": h, "total": e + h}
    return out


def pin_bias_point(c) -> dict:
    cols = {k: [] for k in ("x", "y", "Electrons", "Holes", "NetDoping")}
    for region in ("core", "slab"):
        for k in cols:
            cols[k] += devsim.get_node_model_values(
                device=PIN_DEVICE, region=region, name=k
            )
    width_cm, y_cut_cm, nodes = depletion_width(
        cols["x"],
        cols["y"],
        cols["Electrons"],
        cols["Holes"],
        cols["NetDoping"],
        y_target=c.core_thickness / 2 / CM,
    )
    return {
        "depletion_width_um": width_cm * 1e4,
        "depletion_cut_y_um": y_cut_cm * 1e4,
        "depleted_nodes": nodes,
        "currents": contact_currents(PIN_DEVICE, ("left", "right")),
    }


@pytest.fixture(scope="module")
def pin_results(tmp_path_factory):
    """PINWaveguide solved and ramped to 0 and -1 V, as test_pin_waveguide / v0.8.0 do."""
    # create_2d_mesh and set_net_doping write tecplot files into the working directory.
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("pin"))
    try:
        nm = 1e-9
        c = get_simulation_xsection.PINWaveguide(
            core_width=500 * nm,
            core_thickness=220 * nm,
            slab_thickness=90 * nm,
        )
        c.ddsolver()
        out = {}
        for ind, voltage in enumerate(PIN_VOLTAGES):
            # Vstep is the final voltage, Vinit the previous one (as test_pin_waveguide)
            vinit = 0 if ind == 0 else PIN_VOLTAGES[ind - 1]
            c.ramp_voltage(Vfinal=voltage, Vstep=PIN_VOLTAGES[-1], Vinit=vinit)
            out[voltage] = pin_bias_point(c)
        print(f"PIN results: {out!r}")
        yield out
    finally:
        get_simulation_xsection.clear_devsim_cache()
        os.chdir(cwd)


@pytest.mark.parametrize("voltage", PIN_VOLTAGES)
def test_pin_depletion_width_match_v080(pin_results, voltage) -> None:
    got = pin_results[voltage]["depletion_width_um"]
    want = V080_DEPLETION_WIDTH_UM[voltage]
    rel = abs(got - want) / want
    assert rel <= DEPLETION_WIDTH_RTOL, (
        f"V={voltage}: depletion width {got!r} um, v0.8.0 {want!r} um, rel diff {rel!r}"
    )


@pytest.mark.parametrize("voltage", PIN_VOLTAGES)
def test_pin_currents_below_floor(pin_results, voltage) -> None:
    currents = pin_results[voltage]["currents"]
    for contact, parts in currents.items():
        for name, value in parts.items():
            assert math.isfinite(value) and abs(value) < CURRENT_FLOOR_A_PER_CM, (
                f"V={voltage}: {contact} {name} current {value!r} A/cm is not below "
                f"the {CURRENT_FLOOR_A_PER_CM!r} A/cm floor"
            )
