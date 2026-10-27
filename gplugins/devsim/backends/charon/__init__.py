# SPDX-License-Identifier: MIT
"""The Charon backend of gplugins.devsim: a stub, not implemented yet.

It is to translate a :mod:`gplugins.devsim.device_description` into Charon's Exodus mesh and
input files and run Charon (gf180-tcad spec §12.1, Charon plan), providing the functions of
:class:`~gplugins.devsim.device_description.Backend`. Until then each raises
NotImplementedError. It imports only the neutral layer, never DEVSIM, meshwell or gmsh, so it
imports in an image without them (``tests/test_device_description.py``).
"""

from __future__ import annotations

from typing import Any

from gplugins.devsim.device_description import CrossSection2D, CVSweep, Device, Stack1D
from gplugins.devsim.results import CVResult

__all__ = ["create_1d_device", "create_2d_device", "run_cv_sweep"]


def _todo(what: str):
    return NotImplementedError(f"the Charon backend is not implemented yet ({what})")


def create_1d_device(stack: Stack1D) -> Device:
    """Not implemented yet."""
    raise _todo("create_1d_device")


def create_2d_device(cross_section: CrossSection2D, **options: Any) -> Device:
    """Not implemented yet."""
    raise _todo("create_2d_device")


def run_cv_sweep(device: Device, sweep: CVSweep, **options: Any) -> CVResult:
    """Not implemented yet."""
    raise _todo("run_cv_sweep")
