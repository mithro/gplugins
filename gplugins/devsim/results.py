# SPDX-License-Identifier: MIT
"""Backend-neutral results and errors of gplugins.devsim (the backend seam).

What a backend returns and raises: :class:`CVResult` (with :class:`PointInfo` and
:class:`RoundoffPoint`) from a C-V sweep, :class:`ConvergenceError` from a solve that did not
converge and :class:`CVConvergenceError` from a sweep with such points. The fields' meanings are
those of the DEVSIM backend (:mod:`gplugins.devsim.cv`), which defined them first; another
backend fills them with the same meaning or raises. This module imports no simulator
(``tests/test_device_description.py``); ``gplugins.devsim.cv`` and
``gplugins.devsim.convergence`` re-export these names unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

__all__ = ["CVConvergenceError", "CVResult", "ConvergenceError", "PointInfo", "RoundoffPoint"]


class ConvergenceError(RuntimeError):
    """A DEVSIM solve did not converge at the requested tolerances.

    ``info`` is DEVSIM's solve info (``converged``, ``iterations`` with the per-iteration
    errors) when DEVSIM returned one, else ``None``.
    """

    def __init__(self, message: str, info: dict | None = None) -> None:
        """``message`` and DEVSIM's solve ``info`` (or None)."""
        super().__init__(message)
        self.info = info



@dataclass(frozen=True)
class PointInfo:
    """Convergence of one bias point.

    ``iterations`` counts the Newton iterations of the point's final solve(s) (QS: the
    ``vg - dv_qs`` and ``vg + dv_qs`` solves; AC: the drift-diffusion DC solve).
    ``rel_error`` is the largest per-equation relative error of the last iteration and
    ``abs_error`` the largest ``PotentialEquation`` absolute error (the potential update, V).
    """

    converged: bool
    iterations: int
    rel_error: float
    abs_error: float


@dataclass(frozen=True)
class RoundoffPoint:
    """A QS solve accepted by ``roundoff_accept`` (see the module notes).

    ``index`` is the grid point and ``vg`` the gate bias solved (``vg[index] -+ dv_qs``);
    ``rel_error``/``abs_error`` the last iteration's largest relative error and potential
    update; ``max_abs_psi`` the device's max|psi| and ``roundoff_bound`` = ROUNDOFF_K * eps *
    max_abs_psi; ``region`` the region with the largest relative error, ``psi_bound`` the |psi|
    below which a node there must lie to violate rel_tol and ``min_abs_psi`` the smallest |psi|
    of that region.
    """

    index: int
    vg: float
    rel_error: float
    abs_error: float
    max_abs_psi: float
    roundoff_bound: float
    region: str
    psi_bound: float
    min_abs_psi: float


@dataclass(frozen=True)
class CVResult:
    """A C-V sweep. ``cgg[i]`` is the gate capacitance at ``vg[i]`` in ``unit``.

    ``cgx`` (``"ac"`` only, else empty) maps every non-gate contact x to Cxg = -dQ_x/dVg, so
    that ``cgg[i] == sum(cgx[x][i] for x in cgx)`` (charge conservation).
    ``extended_fallback`` lists the indices of the points that ``precision_fallback`` solved
    (in part) in extended precision; ``roundoff_accepted`` the solves that ``roundoff_accept``
    accepted (:class:`RoundoffPoint`).
    """

    vg: tuple[float, ...]
    cgg: tuple[float, ...]
    unit: Literal["F/cm^2", "F/cm"]
    points: tuple[PointInfo, ...]
    cgx: Mapping[str, tuple[float, ...]]
    method: Literal["qs", "ac"]
    freq_hz: float | None
    dv_qs: float | None
    extended_fallback: tuple[int, ...] = ()
    roundoff_accepted: tuple[RoundoffPoint, ...] = ()


class CVConvergenceError(ConvergenceError):
    """One or more bias points of a C-V sweep did not converge at the requested tolerances.

    ``points`` lists those gate voltages (V) in sweep order (a repeated vg appears once per
    occurrence); ``reasons`` is ``[(index, vg, reason), ...]``, the same points with their
    index in the sweep's ``vg`` and why each failed.
    """

    def __init__(
        self, points: Sequence[float], reasons: Sequence[tuple[int, float, str]] | None = None
    ) -> None:
        """The failed gate voltages ``points`` and their ``(index, vg, reason)``."""
        self.points = [float(v) for v in points]
        self.reasons = [(int(i), float(v), str(r)) for i, v, r in (reasons or ())]
        detail = "; ".join(f"vg[{i}]={v!r}: {r}" for i, v, r in self.reasons) or repr(self.points)
        super().__init__(
            f"C-V sweep: {len(self.points)} bias point(s) did not converge: {detail}", None
        )
