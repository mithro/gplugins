# SPDX-License-Identifier: MIT
"""The Charon backend of gplugins.devsim: a 1D stack's quasi-static C-V on Charon 2.2.

It turns a :class:`~gplugins.devsim.device_description.Stack1D` into Charon's Exodus mesh and
interpreter deck, runs Charon and reads the result back as a
:class:`~gplugins.devsim.results.CVResult`, providing the functions of
:class:`~gplugins.devsim.device_description.Backend` (gf180-tcad spec section 12.1, Charon plan
Task B). It imports only the neutral layer, never DEVSIM, meshwell or gmsh, and needs
``netCDF4`` (Exodus I/O) and Charon itself (``CHARON_EXECUTABLE_PATH``) only when it runs.

**The method** (gf180-tcad ``docs/limitations/charon.md``, experiments B1 to B7):

- the stack's nodes extruded into a one-element strip (:mod:`.exodus`), drift-diffusion CVFEM in
  silicon and Laplace CVFEM in oxide, ohmic contacts, the constants written explicitly
  (:mod:`.deck`);
- :func:`create_1d_device`: the equilibrium state at 0 V on every contact;
- :func:`run_cv_sweep` (``method="qs"``): two LOCA sweeps of the gate at ``vg - dv_qs`` and
  ``vg + dv_qs`` over the whole grid, each a ramp from equilibrium to its first bias and then
  :data:`SUBSTEPS` fixed steps per grid step; C is the central difference of the oxide-flux gate
  charge (:mod:`.parse`), in F/cm^2, the DEVSIM backend's definition and unit. The grid must be
  uniform. A failed LOCA step fails the sweep with :class:`CVConvergenceError`; nothing is
  retried at another tolerance;
- the SRH lifetimes of the deck are :data:`.deck.SRH_LIFETIME_S`, not the description's (see
  :mod:`.deck`); the zero-current result does not depend on them.

``PointInfo`` of a point: ``iterations`` the Newton iterations of the two LOCA steps that
landed on ``vg -+ dv_qs``; ``rel_error`` the larger WRMS norm of the two (NOX's weighted RMS of
the last update, Charon User Manual v2.2 eq. 175; not DEVSIM's relative error); ``abs_error``
NaN, because Charon reports no potential update. A caller records that meaning with the result.

``method="ac"`` and 2D (:func:`create_2d_device`) are not implemented: the harmonic-balance
convention is not settled (experiment B4) and the 2D description is Task E.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gplugins.devsim.backends.charon.deck import (
    SRH_LIFETIME_S,
    NotComparable,
    Sweep,
    Tolerances,
    check_comparable,
    write_deck,
)
from gplugins.devsim.backends.charon.exodus import Mesh, strip_of, write_exodus
from gplugins.devsim.backends.charon.parse import (
    central_difference,
    gate_charge,
    grid_states,
    read_currents,
    read_states,
)
from gplugins.devsim.backends.charon.run import CharonError, CharonExe, newton_steps, run_deck
from gplugins.devsim.device_description import CrossSection2D, CVSweep, Stack1D
from gplugins.devsim.results import CVConvergenceError, CVResult, PointInfo

__all__ = [
    "SRH_LIFETIME_S",
    "SUBSTEPS",
    "CharonDevice",
    "CharonExe",
    "NotComparable",
    "Tolerances",
    "create_1d_device",
    "create_2d_device",
    "run_cv",
    "run_cv_sweep",
]

#: LOCA steps per grid step: one step per grid point stops near the onset of inversion on one
#: of the two sweeps (experiment B2), five pass the whole range.
SUBSTEPS = 5

#: The smallest LOCA step (V) of a ramp from equilibrium to a sweep's first bias.
RAMP_MIN_STEP_V = 1e-4

#: The largest LOCA step (V) of a ramp.
RAMP_MAX_STEP_V = 0.05

MESH_FILE = "strip.exo"
EQUILIBRIUM = "equilibrium"


@dataclass
class CharonDevice:
    """A 1D stack prepared for Charon: its strip mesh and solved equilibrium in ``workdir``."""

    stack: Stack1D
    workdir: Path
    mesh: Mesh
    exe: CharonExe = field(default_factory=CharonExe)
    tolerances: Tolerances = field(default_factory=Tolerances)
    owns_workdir: bool = False

    @property
    def equilibrium_file(self) -> str:
        return f"{EQUILIBRIUM}.exo"

    def delete_device(self) -> None:
        """Remove ``workdir`` if this device created it."""
        if self.owns_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)


def create_1d_device(
    stack: Stack1D,
    *,
    workdir: Path | str | None = None,
    exe: CharonExe | None = None,
    tolerances: Tolerances | None = None,
) -> CharonDevice:
    """Write ``stack``'s strip and solve its equilibrium (0 V on every contact) in ``workdir``
    (a new directory under the current one if None). Raises :class:`NotComparable` for a stack
    the deck cannot write exactly, :class:`CharonError` if the solve fails."""
    check_comparable(stack)
    owns = workdir is None
    wd = Path(tempfile.mkdtemp(prefix="charon_", dir=".")) if owns else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    mesh = strip_of(stack)
    write_exodus(mesh, wd / MESH_FILE)
    dev = CharonDevice(stack, wd, mesh, exe or CharonExe(), tolerances or Tolerances(), owns)
    gate = stack.contacts[0].name
    deck = write_deck(stack, mesh_file=MESH_FILE, out_file=dev.equilibrium_file, gate=gate,
                      gate_bc="fixed at 0.0", problem="steady state", init="equilibrium",
                      tolerances=dev.tolerances)  # fmt: skip
    (wd / f"{EQUILIBRIUM}.inp").write_text(deck, encoding="utf-8")
    run_deck(f"{EQUILIBRIUM}.inp", cwd=wd, exe=dev.exe)
    return dev


def create_2d_device(cross_section: CrossSection2D, **options: Any):
    """Not implemented: the 2D MOS capacitor description is Task E of the Charon plan."""
    raise NotImplementedError("the Charon backend has no 2D device yet (Charon plan, Task E)")


def _uniform_step(vg: tuple[float, ...]) -> float:
    if len(vg) < 2:
        raise ValueError("a Charon C-V sweep needs at least two grid points")
    steps = np.diff(vg)
    step = float(steps[0])
    if step == 0 or np.max(np.abs(steps - step)) > 1e-9 * abs(step):
        raise ValueError("a Charon C-V sweep needs a uniform vg grid")
    return step


def _sweep_side(dev: CharonDevice, gate: str, name: str, vg: np.ndarray, step: float,
                sub: int) -> tuple[np.ndarray, list]:
    """One LOCA sweep over ``vg``: ramp from equilibrium to vg[0], then ``sub`` steps per grid
    step. Returns the gate charge (C/cm^2) at each grid point and its Newton steps."""
    wd, stack = dev.workdir, dev.stack
    v0, v1 = float(vg[0]), float(vg[-1])
    ramp = Sweep(min(abs(step), RAMP_MAX_STEP_V), 10**6, min_step=RAMP_MIN_STEP_V,
                 hit_bound=True, predictor="constant")  # fmt: skip
    if v0 != 0.0:
        deck = write_deck(stack, mesh_file=dev.equilibrium_file, out_file=f"{name}_ramp.exo",
                          gate=gate, gate_bc=f"swept from 0.0 to {v0!r}",
                          problem="parameter sweep", init="exodus", sweep=ramp,
                          tolerances=dev.tolerances)  # fmt: skip
        (wd / f"{name}_ramp.inp").write_text(deck, encoding="utf-8")
        run_deck(f"{name}_ramp.inp", cwd=wd, exe=dev.exe)
        start = f"{name}_ramp.exo"
    else:
        start = dev.equilibrium_file
    h = step / sub
    sweep = Sweep(h, (len(vg) - 1) * sub + 5)
    deck = write_deck(stack, mesh_file=start, out_file=f"{name}.exo", gate=gate,
                      gate_bc=f"swept from {v0!r} to {v1!r}", problem="parameter sweep",
                      init="exodus", sweep=sweep, tolerances=dev.tolerances,
                      import_index=-1)  # fmt: skip
    (wd / f"{name}.inp").write_text(deck, encoding="utf-8")
    log = run_deck(f"{name}.inp", cwd=wd, exe=dev.exe)
    head, rows = read_currents(wd / "currents-loca.dat")
    shutil.copy(wd / "currents-loca.dat", wd / f"{name}.currents.dat")
    col = head.index(f"{gate}_Voltage")
    idx = grid_states(rows[:, col], vg, sub)
    states = read_states(wd / f"{name}.exo")
    xs, bnodes = stack.node_positions_um()
    ox = [i for i, layer in enumerate(stack.layers) if layer.material == "oxide"]
    if len(ox) != 1:
        raise NotComparable("the gate charge needs exactly one oxide layer")
    i = ox[0]
    c = stack.constants
    q, _spread = gate_charge(states, xs, (bnodes[i], bnodes[i + 1]), c.eps_ox,
                             c.eps0_f_per_cm, idx)  # fmt: skip
    steps = newton_steps(log.solver)
    return q, [steps[k] if k < len(steps) else None for k in idx]


def run_cv_sweep(device: CharonDevice, sweep: CVSweep, *, substeps: int = SUBSTEPS) -> CVResult:
    """The quasi-static C-V of ``sweep`` on ``device`` (see the module notes). The sign of the
    gate charge follows the gate: C > 0. Raises CVConvergenceError if a sweep fails."""
    if sweep.method != "qs":
        raise NotImplementedError("the Charon backend computes the quasi-static C-V only")
    if sweep.gate not in device.stack.contact_ends():
        raise ValueError(f"gate {sweep.gate!r} is not a contact of the stack")
    vg = np.asarray(sweep.vg, dtype=float)
    step = _uniform_step(sweep.vg)
    dv = float(sweep.dv_qs)
    try:
        q_lo, info_lo = _sweep_side(device, sweep.gate, "sweep_minus", vg - dv, step, substeps)
        q_hi, info_hi = _sweep_side(device, sweep.gate, "sweep_plus", vg + dv, step, substeps)
    except (CharonError, ValueError) as e:
        raise CVConvergenceError(list(sweep.vg),
                                 [(i, v, str(e)) for i, v in enumerate(sweep.vg)]) from e
    cgg = central_difference(q_lo, q_hi, dv)
    points = []
    for a, b in zip(info_lo, info_hi, strict=True):
        if a is None or b is None:
            raise CVConvergenceError(list(sweep.vg), [(-1, math.nan, "missing Newton record")])
        points.append(PointInfo(True, a.iterations + b.iterations,
                                max(a.wrms or 0.0, b.wrms or 0.0), math.nan))  # fmt: skip
    return CVResult(vg=tuple(float(v) for v in vg), cgg=tuple(float(c) for c in cgg),
                    unit="F/cm^2", points=tuple(points), cgx={}, method="qs", freq_hz=None,
                    dv_qs=dv)  # fmt: skip


def run_cv(stack: Stack1D, sweep: CVSweep, *, workdir: Path | str,
           exe: CharonExe | None = None) -> CVResult:
    """:func:`create_1d_device` then :func:`run_cv_sweep` in ``workdir``."""
    dev = create_1d_device(stack, workdir=workdir, exe=exe)
    try:
        return run_cv_sweep(dev, sweep)
    finally:
        dev.delete_device()
