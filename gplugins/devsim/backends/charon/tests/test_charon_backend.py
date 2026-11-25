# SPDX-License-Identifier: MIT
"""The Charon backend (gplugins.devsim.backends.charon).

They need numpy and netCDF4, and the tests that run Charon need ``charon_mp.exe`` under
``CHARON_EXECUTABLE_PATH``: the gf180-tcad Charon image has both (the ``test_charon`` job of
``.github/workflows/test_code.yml``). The image also has DEVSIM, so the two backends are compared
on one stack (:func:`test_qs_equals_devsim_1d`). gf180-tcad checks the Exodus file with meshio and
the C-V against its exact Poisson-Boltzmann reference.
"""

from __future__ import annotations

import math
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from gplugins.devsim import doping
from gplugins.devsim.backends import charon
from gplugins.devsim.backends.charon import deck as deck_mod
from gplugins.devsim.backends.charon import exodus, parse
from gplugins.devsim.constants import Constants
from gplugins.devsim.device_description import (
    Contact,
    CVSweep,
    InterfaceRef,
    Layer1D,
    Mesh1D,
    Stack1D,
)

netCDF4 = pytest.importorskip("netCDF4")

CHARON = Path(os.environ.get("CHARON_EXECUTABLE_PATH", "/opt/charon/bin")) / "charon_mp.exe"
needs_charon = pytest.mark.skipif(not CHARON.is_file(), reason="needs Charon (the Charon image)")

# The nfet_05v0 1D stack of gf180-tcad at its calibrated values
# (results/tcad/moscap1d/nfet_05v0/calibrated/per_area.meta.json).
NGATE, N_CH, T_OX_NM = 1e20, 8.813323306821749e16, 16.050483616382458
SPACING = {"gate_poly": (0.02, 5e-05), "oxide": (5e-05, 5e-05), "body": (5e-05, 0.02)}


def nfet_stack(*, body=None, spacing=SPACING, eps_ox=3.9) -> Stack1D:
    layers = [
        Layer1D("gate_poly", "silicon", 0.2, doping.constant(NGATE), "donor"),
        Layer1D("oxide", "oxide", T_OX_NM * 1e-3, None),
        Layer1D("body", "silicon", 1.0, body or doping.constant(N_CH), "acceptor"),
    ]
    contacts = [
        Contact("gate", InterfaceRef("gate_poly", None), "ohmic"),
        Contact("body", InterfaceRef("body", None), "ohmic"),
    ]
    return Stack1D("nfet", layers, contacts, Mesh1D(spacing, max_ratio=1.1),
                   Constants(eps_ox=eps_ox))  # fmt: skip


def coarse_stack() -> Stack1D:
    """The same stack on a coarse mesh, for the quick Charon runs."""
    return nfet_stack(spacing={"gate_poly": (0.02, 0.0005), "oxide": (0.0005, 0.0005),
                               "body": (0.0005, 0.02)})  # fmt: skip


# --- Exodus --------------------------------------------------------------------------------


def test_exodus_nodes_identical(tmp_path):
    stack = nfet_stack()
    mesh = exodus.strip_of(stack)
    exodus.write_exodus(mesh, tmp_path / "m.exo")
    xs, bnodes = stack.node_positions_um()
    ds = netCDF4.Dataset(tmp_path / "m.exo")
    ds.set_auto_mask(False)
    x, y = ds["coordx"][:], ds["coordy"][:]
    assert np.array_equal(x[0::2], np.asarray(xs)) and np.array_equal(x[1::2], np.asarray(xs))
    assert np.all(y[0::2] == 0) and np.all(y[1::2] == exodus.STRIP_WIDTH_UM)
    assert ds.dimensions["num_elem"].size == len(xs) - 1
    names = ["".join(c.decode() for c in row if c) for row in ds["eb_names"][:]]
    assert names == [layer.name for layer in stack.layers]
    for k, (lo, hi) in enumerate(zip(bnodes, bnodes[1:]), 1):
        conn = ds[f"connect{k}"][:] - 1
        js = np.arange(lo, hi)
        assert np.array_equal(conn, np.column_stack([2 * js, 2 * js + 2, 2 * js + 3, 2 * js + 1]))
    ss = ["".join(c.decode() for c in row if c) for row in ds["ss_names"][:]]
    assert ss == ["gate", "body_contact"]  # "body" is a block's name
    assert [int(ds["side_ss1"][0]), int(ds["side_ss2"][0])] == [4, 2]
    assert [int(ds["elem_ss1"][0]), int(ds["elem_ss2"][0])] == [1, len(xs) - 1]
    ds.close()


def test_exodus_write_is_byte_identical(tmp_path):
    mesh = exodus.strip_of(nfet_stack())
    exodus.write_exodus(mesh, tmp_path / "a.exo")
    exodus.write_exodus(mesh, tmp_path / "b.exo")
    assert (tmp_path / "a.exo").read_bytes() == (tmp_path / "b.exo").read_bytes()


def test_exodus_refuses_side_set_named_like_a_block(tmp_path):
    mesh = exodus.strip_of(nfet_stack())
    bad = exodus.Mesh(mesh.xy_um, mesh.blocks,
                      (exodus.SideSet("body", (1,), (4,)),))  # fmt: skip
    with pytest.raises(ValueError, match="named like element blocks"):
        exodus.write_exodus(bad, tmp_path / "m.exo")


# --- the deck ------------------------------------------------------------------------------


def _deck(stack=None, **kw) -> str:
    stack = stack or nfet_stack()
    args = dict(mesh_file="m.exo", out_file="o.exo", gate="gate", gate_bc="fixed at 0.0",
                problem="steady state", init="equilibrium")  # fmt: skip
    args.update(kw)
    return deck_mod.write_deck(stack, **args)


def test_deck_constants_match_description():
    stack = nfet_stack()
    c = stack.constants
    d = _deck(stack)
    nums = lambda pat: [float(v) for v in re.findall(pat, d, re.M)]  # noqa: E731
    assert nums(r"relative permittivity = (\S+)") == [c.eps_si, c.eps_ox, c.eps_si]
    assert nums(r"lattice temperature = (\S+)") == [c.t_k]
    assert nums(r"constant intrinsic concentration = (\S+)") == [c.n_i_cm3] * 2
    assert re.findall(r"fixed mobility for electrons is (\S+) holes is (\S+)", d) == [
        (repr(c.mu_n_cm2_per_vs), repr(c.mu_p_cm2_per_vs))] * 2
    assert nums(r"lifetime is constant = (\S+)") == [deck_mod.SRH_LIFETIME_S] * 4
    assert nums(r"^ +concentration = (\S+)$") == [NGATE, N_CH]  # the doping blocks only
    assert re.findall(r"type is (\w+)", d) == ["donor", "acceptor"]
    assert "BC is ohmic for gate on gate_poly fixed at 0.0" in d
    assert "BC is ohmic for body_contact on body fixed at 0.0" in d
    for absent in ("Fermi", "band gap narrowing", "incomplete ionization", "work function"):
        assert absent.lower() not in d.lower(), absent
    assert d.count("uniform doping named Function1") == 4  # start and end, two blocks
    for line in deck_mod.KLU2_LINES:
        assert line in d


def test_not_comparable_profile():
    stack = nfet_stack(body=doping.gaussian(1e17, 0.1, 0.05))
    with pytest.raises(deck_mod.NotComparable, match="no exact Charon form"):
        deck_mod.check_comparable(stack)
    stack = nfet_stack(body=doping.constant(N_CH, ymin=0.1))
    with pytest.raises(deck_mod.NotComparable):
        _deck(stack)


def test_deck_fixture_byte_identical():
    fixture = Path(__file__).parent / "fixtures" / "charon_nfet_equilibrium.inp"
    assert _deck() == fixture.read_text(encoding="utf-8")


# --- parsing -------------------------------------------------------------------------------


def test_sweep_grid_alignment():
    want = np.array([-1.0, -0.95, -0.9])
    v = np.array([-1.0, -0.99, -0.98, -0.97, -0.96, -0.95, -0.94, -0.93, -0.92, -0.91, -0.9,
                  -0.9])  # fmt: skip
    assert list(parse.grid_states(v, want, 5)) == [0, 5, 10]
    with pytest.raises(ValueError, match="sweep cut"):
        parse.grid_states(v[:8], want, 5)
    off = v.copy()
    off[5] += 1e-6
    with pytest.raises(ValueError, match="grid point"):
        parse.grid_states(off, want, 5)
    with pytest.raises(ValueError, match="after the last grid point"):
        parse.grid_states(np.append(v, -0.85), want, 5)


def test_central_difference():
    assert np.allclose(parse.central_difference([1.0, 2.0], [3.0, 6.0], 0.5), [2.0, 4.0])


def test_grid_must_be_uniform():
    with pytest.raises(ValueError, match="uniform"):
        charon._uniform_step((0.0, 0.1, 0.3))


# --- Charon runs ---------------------------------------------------------------------------


@needs_charon
def test_exodus_accepted_by_charon(tmp_path):
    dev = charon.create_1d_device(coarse_stack(), workdir=tmp_path)
    states = parse.read_states(tmp_path / dev.equilibrium_file)
    assert states.nodal["ELECTRIC_POTENTIAL"].shape == (1, len(dev.mesh.xy_um))


@needs_charon
def test_doping_nodal_values(tmp_path):
    """B5: Charon's nodal doping is the description's at every silicon node."""
    stack = coarse_stack()
    dev = charon.create_1d_device(stack, workdir=tmp_path)
    d = deck_mod.write_deck(stack, mesh_file=dev.equilibrium_file, out_file="dop.exo",
                            gate="gate", gate_bc="fixed at 0.0", problem="steady state",
                            init="exodus")  # fmt: skip
    d = d.replace("end output parameters",
                  '  Output nodal variables in gate_poly for "Doping"\n'
                  '  Output nodal variables in body for "Doping"\nend output parameters')
    (tmp_path / "dop.inp").write_text(d)
    log = charon.run.run_deck("dop.inp", cwd=tmp_path)
    c0 = float(re.search(r"Concentration Scaling C0 = (\S+)", log.solver).group(1))
    dop = parse.read_states(tmp_path / "dop.exo").nodal["Doping"][0, 0::2] * c0
    xs, bn = stack.node_positions_um()
    assert np.allclose(dop[: bn[1] + 1], NGATE, rtol=1e-9, atol=0)
    assert np.allclose(dop[bn[2]:], -N_CH, rtol=1e-9, atol=0)


def _sweep(stack, vg, tmp_path, name="cv"):
    return charon.run_cv(stack, CVSweep("gate", tuple(vg), "qs"), workdir=tmp_path / name)


@needs_charon
def test_strip_width_independent(tmp_path, monkeypatch):
    vg = (-1.0, -0.5, 0.0)
    a = _sweep(coarse_stack(), vg, tmp_path, "w1")
    monkeypatch.setattr(exodus, "STRIP_WIDTH_UM", 2.0)
    monkeypatch.setattr(exodus.strip_of, "__defaults__", (2.0,))
    b = _sweep(coarse_stack(), vg, tmp_path, "w2")
    assert np.allclose(a.cgg, b.cgg, rtol=1e-9, atol=0)


@needs_charon
def test_qs_through_inversion(tmp_path):
    vg = np.round(np.arange(-2.0, 2.0001, 0.25), 12)
    res = _sweep(coarse_stack(), vg, tmp_path)
    assert res.unit == "F/cm^2" and len(res.cgg) == len(vg)
    assert all(p.converged and p.iterations > 0 and math.isnan(p.abs_error) for p in res.points)
    c_ox = 3.9 * Constants().eps0_f_per_cm / (T_OX_NM * 1e-7)
    assert all(0 < c < c_ox for c in res.cgg)


@needs_charon
def test_two_runs_byte_identical(tmp_path):
    vg = (-0.5, 0.0, 0.5)
    _sweep(coarse_stack(), vg, tmp_path, "a")
    _sweep(coarse_stack(), vg, tmp_path, "b")
    for name in ("sweep_minus.currents.dat", "sweep_plus.currents.dat"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


@needs_charon
def test_qs_equals_devsim_1d(tmp_path):
    """Charon and DEVSIM on the identical 1D stack: C within 1 % at every bias."""
    from gplugins.devsim import sweep_cv
    from gplugins.devsim.stack1d import create_1d_stack

    stack = coarse_stack()
    vg = tuple(np.round(np.arange(-2.0, 2.0001, 0.25), 12))
    res = _sweep(stack, vg, tmp_path)
    dd = create_1d_stack("xcheck", list(stack.layers), list(stack.contacts), stack.mesh,
                         stack.constants, relative_error=1e-10)  # fmt: skip
    try:
        ref = sweep_cv(dd, "gate", vg, method="qs", dv_qs=1e-3, rel_tol=1e-10, abs_tol=1e-12,
                       max_iterations=50)  # fmt: skip
    finally:
        dd.delete_device()
    rel = np.abs(np.asarray(res.cgg) / np.asarray(ref.cgg) - 1)
    assert np.max(rel) < 0.01, rel


@needs_charon
def test_single_rank_single_thread(tmp_path):
    """charon_mp.exe runs as one process (no mpirun, no orted) with at most two threads: the
    solve and OpenMPI's progress thread (as gf180-tcad's image test counts them)."""
    import threading
    import time

    stack = coarse_stack()
    seen = {"procs": set(), "threads": 0}
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    status = Path(f"/proc/{pid}/status").read_text()
                except OSError:
                    continue
                name = re.search(r"Name:\s*(\S+)", status).group(1)
                if name.startswith(("charon", "orted", "mpirun")):
                    seen["procs"].add(name)
                    if name.startswith("charon"):
                        n = int(re.search(r"Threads:\s*(\d+)", status).group(1))
                        seen["threads"] = max(seen["threads"], n)
            time.sleep(0.05)

    t = threading.Thread(target=sample)
    t.start()
    try:
        charon.create_1d_device(stack, workdir=tmp_path)
    finally:
        stop.set()
        t.join()
    assert seen["procs"] and all(p.startswith("charon") for p in seen["procs"]), seen
    assert 1 <= seen["threads"] <= 2, seen
