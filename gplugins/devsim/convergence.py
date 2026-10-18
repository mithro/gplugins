# SPDX-License-Identifier: MIT
"""DC solves that never fail silently.

``devsim.solve(..., info=True)`` does **not** raise when Newton fails to converge: it returns
``{"converged": False, ...}`` (DEVSIM 2.11.1; without ``info`` it raises ``devsim.error``).
:func:`solve_dc` always asks for the info, raises :class:`ConvergenceError` on a
non-converged solve, and returns the info of a converged one. Tolerances are never relaxed.
"""

from __future__ import annotations

import devsim


class ConvergenceError(RuntimeError):
    """A DEVSIM solve did not converge at the requested tolerances.

    ``info`` is DEVSIM's solve info (``converged``, ``iterations`` with the per-iteration
    errors) when DEVSIM returned one, else ``None``.
    """

    def __init__(self, message: str, info: dict | None = None) -> None:
        super().__init__(message)
        self.info = info


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
