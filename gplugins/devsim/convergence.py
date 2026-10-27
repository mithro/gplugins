# SPDX-License-Identifier: MIT
"""DC solves that never fail silently.

``devsim.solve(..., info=True)`` does **not** raise when Newton fails to converge: it returns
``{"converged": False, ...}`` (DEVSIM 2.11.1; without ``info`` it raises ``devsim.error``).
:func:`solve_dc` always asks for the info, raises :class:`ConvergenceError` on a
non-converged solve, and returns the info of a converged one. Tolerances are never relaxed.
"""

from __future__ import annotations

import devsim

# Backend-neutral; lives in gplugins.devsim.results (the backend seam), re-exported here.
from gplugins.devsim.results import ConvergenceError

__all__ = ["ConvergenceError", "solve_dc"]


def solve_dc(
    *, absolute_error: float, relative_error: float, maximum_iterations: int
) -> dict:
    """A DC solve at the given tolerances; raises :class:`ConvergenceError` unless converged.

    ``absolute_error`` and ``relative_error`` are DEVSIM's (the update norms of each equation's
    variable: V for ``Potential``, cm^-3 for the carriers). Returns DEVSIM's solve info.
    """
    try:
        info = devsim.solve(
            type="dc",
            absolute_error=absolute_error,
            relative_error=relative_error,
            maximum_iterations=maximum_iterations,
            info=True,
        )
    except devsim.error as e:
        raise ConvergenceError(f"DEVSIM DC solve failed: {e}") from e
    if not info or not info.get("converged"):
        iterations = info.get("iterations", ()) if info else ()
        last = iterations[-1] if iterations else None
        raise ConvergenceError(
            "DEVSIM DC solve did not converge "
            f"(absolute_error={absolute_error!r}, relative_error={relative_error!r}, "
            f"maximum_iterations={maximum_iterations!r}); last iteration: {last!r}",
            info,
        )
    return info
