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
  fixed steps of at most :data:`LOCA_STEP_V` (:func:`substeps_for`); C is the central
  difference of the oxide-flux gate charge (:mod:`.parse`), in F/cm^2, the DEVSIM backend's definition and unit. The grid must be
  uniform. A failed LOCA step fails the sweep with :class:`CVConvergenceError`; nothing is
  retried at another tolerance;
- the SRH lifetimes of these quasi-static decks are :data:`.deck.SRH_LIFETIME_S`, not the
  description's (see :mod:`.deck`); the zero-current result does not depend on them. An AC deck
  would use the description's;
- the charge checks of experiment B6 run on every sweep (:data:`FLUX_SPREAD_MAX`,
  :data:`CHARGE_MISMATCH_MAX`) and are kept in :attr:`CharonDevice.checks`;
- the gate is the contact at the stack's first end (x = 0): the gate charge is the oxide flux at
  that face. ``vg`` is the bias Charon solves at: a flat-band shift such as gf180-tcad's
  ``dv_fb_v`` is applied by the caller, as for the DEVSIM backend.

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
    check_gate,
    write_deck,
)
from gplugins.devsim.backends.charon.exodus import Mesh, strip_of, write_exodus
from gplugins.devsim.backends.charon.parse import (
    CURRENTS_RESOLUTION_V,
    central_difference,
    gate_charge,
    grid_states,
    read_currents,
    read_states,
    space_charge,
)
from gplugins.devsim.backends.charon.run import (
    CharonError,
    CharonExe,
    StepInfo,
    newton_steps,
    run_deck,
)
from gplugins.devsim.device_description import CrossSection2D, CVSweep, Stack1D
from gplugins.devsim.results import CVConvergenceError, CVResult, PointInfo

__all__ = [
    "CHARGE_MISMATCH_MAX",
    "FLUX_SPREAD_MAX",
    "LOCA_STEP_V",
    "SRH_LIFETIME_S",
    "ChargeCheckError",
    "CharonDevice",
    "CharonError",
    "CharonExe",
    "NotComparable",
    "OutputError",
    "Tolerances",
    "create_1d_device",
    "create_2d_device",
    "gate_charge",
    "read_states",
    "run_cv",
    "run_cv_sweep",
    "substeps_for",
    "write_deck",
    "write_exodus",
]

#: The largest LOCA step (V) of a C-V sweep: each grid step is cut into the fewest equal steps
#: of at most this size. 10 mV (five steps per 50 mV grid step) passes the whole range on both
#: devices (gf180-tcad experiment B2); one secant step per 50 mV grid step stops in inversion
#: (gf180-tcad ``study-nfet-tau15-secant-50mV``: at 0.552 V).
LOCA_STEP_V = 0.01


def substeps_for(step: float) -> int:
    """The LOCA steps per grid step of ``step`` (V): the fewest of at most :data:`LOCA_STEP_V`."""
    return max(1, math.ceil(abs(step) / LOCA_STEP_V - 1e-9))


#: The largest relative spread of the oxide flux over the oxide's intervals (B6: at most 1.4e-12
#: on both devices); a larger one fails the sweep.
FLUX_SPREAD_MAX = 1e-10

#: The largest relative mismatch of the poly's space charge with the gate charge, and of the
#: body's with minus the gate charge (B6: at most 8.1e-13); a larger one fails the sweep.
CHARGE_MISMATCH_MAX = 1e-9

#: The smallest LOCA step (V) of a ramp from equilibrium to a sweep's first bias.
RAMP_MIN_STEP_V = 1e-4

#: The largest LOCA step (V) of a ramp.
RAMP_MAX_STEP_V = 0.05

MESH_FILE = "strip.exo"
EQUILIBRIUM = "equilibrium"


class ChargeCheckError(RuntimeError):
    """A sweep's charges fail the conservation checks of experiment B6."""


class OutputError(RuntimeError):
    """Charon's output does not match the sweep it was asked for.

    A cut or misaligned sweep, or Newton records that do not cover its states: a wrong result,
    not a convergence failure.
    """


@dataclass
class CharonDevice:
    """A 1D stack prepared for Charon: its strip mesh and solved equilibrium in ``workdir``."""

    stack: Stack1D
    workdir: Path
    mesh: Mesh
    exe: CharonExe = field(default_factory=CharonExe)
    tolerances: Tolerances = field(default_factory=Tolerances)
    owns_workdir: bool = False
    #: The B6 checks of the last sweep: per side, the largest oxide-flux spread and the largest
    #: poly and body charge mismatches (relative).
    checks: dict = field(default_factory=dict)
    #: The Newton solves of the last sweep at each grid point: per side, one
    #: ``{"iterations", "f_norm", "wrms"}`` per point (the achieved convergence, against
    #: :attr:`tolerances`).
    newton: dict = field(default_factory=dict)

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
    """Write ``stack``'s strip and solve its equilibrium (0 V on every contact).

    The files go to ``workdir`` (a new directory under the current one if None). Raises
    :class:`NotComparable` for a stack the deck cannot write exactly, :class:`CharonError` if
    the solve fails.
    """
    check_comparable(stack)
    gate = next(k for k, e in stack.contact_ends().items() if e == 0)
    owns = workdir is None
    wd = Path(tempfile.mkdtemp(prefix="charon_", dir=".")) if owns else Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    mesh = strip_of(stack)
    write_exodus(mesh, wd / MESH_FILE)
    dev = CharonDevice(
        stack, wd, mesh, exe or CharonExe(), tolerances or Tolerances(), owns
    )
    deck = write_deck(stack, mesh_file=MESH_FILE, out_file=dev.equilibrium_file, gate=gate,
                      gate_bc="fixed at 0.0", problem="steady state", init="equilibrium",
                      tolerances=dev.tolerances, quasi_static=True)  # fmt: skip
    (wd / f"{EQUILIBRIUM}.inp").write_text(deck, encoding="utf-8")
    run_deck(f"{EQUILIBRIUM}.inp", cwd=wd, exe=dev.exe)
    return dev


def create_2d_device(cross_section: CrossSection2D, **options: Any):
    """Not implemented: the 2D MOS capacitor description is Task E of the Charon plan."""
    raise NotImplementedError(
        "the Charon backend has no 2D device yet (Charon plan, Task E)"
    )


def _uniform_step(vg: tuple[float, ...]) -> float:
    if len(vg) < 2:
        raise ValueError("a Charon C-V sweep needs at least two grid points")
    steps = np.diff(vg)
    step = float(steps[0])
    if step == 0 or np.max(np.abs(steps - step)) > 1e-9 * abs(step):
        raise ValueError("a Charon C-V sweep needs a uniform vg grid")
    return step


def _sweep_side(
    dev: CharonDevice, gate: str, name: str, vg: np.ndarray, step: float, sub: int
) -> tuple[np.ndarray, list]:
    """One LOCA sweep over ``vg``: a ramp from equilibrium to vg[0], then the sweep.

    The sweep takes ``sub`` steps per grid step. Returns the gate charge (C/cm^2) at each grid
    point, its Newton steps and the B6 checks.
    """
    wd, stack = dev.workdir, dev.stack
    v0, v1 = float(vg[0]), float(vg[-1])
    ramp = Sweep(min(abs(step), RAMP_MAX_STEP_V), 10**6, min_step=RAMP_MIN_STEP_V,
                 hit_bound=True, predictor="constant")  # fmt: skip
    if v0 != 0.0:
        deck = write_deck(stack, mesh_file=dev.equilibrium_file, out_file=f"{name}_ramp.exo",
                          gate=gate, gate_bc=f"swept from 0.0 to {v0!r}",
                          problem="parameter sweep", init="exodus", sweep=ramp,
                          tolerances=dev.tolerances, quasi_static=True)  # fmt: skip
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
                      import_index=-1, quasi_static=True)  # fmt: skip
    (wd / f"{name}.inp").write_text(deck, encoding="utf-8")
    log = run_deck(f"{name}.inp", cwd=wd, exe=dev.exe)
    head, rows = read_currents(wd / "currents-loca.dat")
    shutil.copy(wd / "currents-loca.dat", wd / f"{name}.currents.dat")
    col = head.index(f"{gate}_Voltage")
    try:
        idx = grid_states(rows[:, col], vg, sub)
    except ValueError as e:
        raise OutputError(f"{name}: {e}") from e
    states = read_states(wd / f"{name}.exo")
    xs, bnodes = stack.node_positions_um()
    i = next(k for k, layer in enumerate(stack.layers) if layer.material == "oxide")
    c = stack.constants
    q, spread = gate_charge(states, xs, (bnodes[i], bnodes[i + 1]), c.eps_ox,
                            c.eps0_f_per_cm, idx)  # fmt: skip
    scale = float(np.max(np.abs(q)))
    mismatch = {}
    for k, layer in enumerate(stack.layers):
        if layer.material != "silicon":
            continue
        sign = 1.0 if layer.dopant == "donor" else -1.0
        rho = space_charge(states, xs, (bnodes[k], bnodes[k + 1]), sign * layer.doping.conc,
                           c.q_c, idx)  # fmt: skip
        # the layer before the oxide holds the gate's charge, the one after its image
        want = q if k < i else -q
        mismatch[layer.name] = float(np.max(np.abs(rho - want)) / scale)
    checks = {"oxide_flux_spread": float(np.max(spread)), "charge_mismatch": mismatch}
    if (
        checks["oxide_flux_spread"] > FLUX_SPREAD_MAX
        or max(mismatch.values()) > CHARGE_MISMATCH_MAX
    ):
        raise ChargeCheckError(f"{name}: B6 charge checks failed: {checks}")
    steps = newton_steps(log.solver)
    if len(steps) < idx[-1] + 1:
        raise OutputError(
            f"{name}: {len(steps)} Newton records for {idx[-1] + 1} states"
        )
    return q, [steps[k] for k in idx], checks


def _failed_index(biases: np.ndarray, failed_at: float) -> int:
    """The index of the first grid point, in sweep order, at or past ``failed_at`` (V).

    That is the point a LOCA step failing at ``failed_at`` was on its way to; a failure on the
    ramp to the first point gives 0.
    """
    direction = 1.0 if len(biases) < 2 or biases[-1] >= biases[0] else -1.0
    past = np.nonzero(direction * (biases - failed_at) >= -CURRENTS_RESOLUTION_V)[0]
    return int(past[0]) if len(past) else len(biases) - 1


def _point(a: StepInfo, b: StepInfo) -> PointInfo:
    """A grid point's PointInfo from the Newton solves of its two sides.

    Their iterations and the larger WRMS norm; a WRMS norm missing from the log raises
    OutputError.
    """
    if a.wrms is None or b.wrms is None:
        raise OutputError(f"a grid point's Newton record has no WRMS norm: {a}, {b}")
    return PointInfo(True, a.iterations + b.iterations, max(a.wrms, b.wrms), math.nan)


def run_cv_sweep(
    device: CharonDevice, sweep: CVSweep, *, substeps: int | None = None
) -> CVResult:
    """The quasi-static C-V of ``sweep`` on ``device`` (see the module notes).

    ``substeps`` LOCA steps per grid step (default :func:`substeps_for`). The sign of the gate
    charge follows the gate: C > 0. Raises CVConvergenceError when a LOCA step fails, naming
    the grid point it was stepping to; CharonError when a run fails otherwise; and, for a wrong
    result rather than a convergence failure, ChargeCheckError (the B6 charge checks) or
    OutputError (output that does not match the sweep). NotComparable and the other refusals
    propagate as themselves.
    """
    if sweep.method != "qs":
        raise NotImplementedError(
            "the Charon backend computes the quasi-static C-V only"
        )
    check_gate(device.stack, sweep.gate)
    vg = np.asarray(sweep.vg, dtype=float)
    step = _uniform_step(sweep.vg)
    substeps = substeps or substeps_for(step)
    dv = float(sweep.dv_qs)
    sides = {}
    for name, sgn in (("sweep_minus", -1.0), ("sweep_plus", 1.0)):
        try:
            sides[name] = _sweep_side(
                device, sweep.gate, name, vg + sgn * dv, step, substeps
            )
        except CharonError as e:
            if e.failed_at is None:
                raise  # not a convergence failure at a bias: the run itself failed
            # LOCA gives up at its first failed step: the grid point it was stepping to
            k = _failed_index(vg + sgn * dv, e.failed_at)
            raise CVConvergenceError(
                [float(vg[k])], [(k, float(vg[k]), f"{name}: {e}")]
            ) from e
    (q_lo, info_lo, chk_lo), (q_hi, info_hi, chk_hi) = (
        sides["sweep_minus"],
        sides["sweep_plus"],
    )
    device.checks = {"sweep_minus": chk_lo, "sweep_plus": chk_hi}
    device.newton = {
        side: [
            {"iterations": i.iterations, "f_norm": i.f_norm, "wrms": i.wrms}
            for i in info
        ]
        for side, info in (("sweep_minus", info_lo), ("sweep_plus", info_hi))
    }
    cgg = central_difference(q_lo, q_hi, dv)
    points = [_point(a, b) for a, b in zip(info_lo, info_hi, strict=True)]
    return CVResult(vg=tuple(float(v) for v in vg), cgg=tuple(float(c) for c in cgg),
                    unit="F/cm^2", points=tuple(points), cgx={}, method="qs", freq_hz=None,
                    dv_qs=dv)  # fmt: skip


def run_cv(
    stack: Stack1D, sweep: CVSweep, *, workdir: Path | str, exe: CharonExe | None = None
) -> CVResult:
    """:func:`create_1d_device` then :func:`run_cv_sweep` in ``workdir``."""
    dev = create_1d_device(stack, workdir=workdir, exe=exe)
    try:
        return run_cv_sweep(dev, sweep)
    finally:
        dev.delete_device()
