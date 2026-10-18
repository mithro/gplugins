"""Meshwell plugin."""

import importlib.metadata as _metadata
import platform as _platform
import re as _re


def _require_meshwell_2_5() -> None:
    """Raise a clear ImportError if the installed meshwell predates 2.5.

    gplugins uses meshwell 2.5's API (``meshwell.orchestrator.cad``). meshwell 2.5 needs
    Python >= 3.13, so the gplugins extras install it only there; on older Python, femwell's
    own unpinned meshwell dependency resolves meshwell 2.3.6, which lacks that API.
    """
    try:
        version = _metadata.version("meshwell")
    except _metadata.PackageNotFoundError:
        return  # the meshwell imports below raise the usual ModuleNotFoundError
    major_minor = tuple(int(x) for x in _re.findall(r"\d+", version)[:2])
    if major_minor < (2, 5):
        raise ImportError(
            f"gplugins' meshwell, femwell and elmer plugins need meshwell >= 2.5, which needs "
            f"Python >= 3.13; installed: meshwell {version} on Python "
            f"{_platform.python_version()}"
        )


_require_meshwell_2_5()


from .get_meshwell_3D import get_meshwell_prisms  # noqa: E402 (after the version check)
from .get_meshwell_cross_section import (  # noqa: E402
    get_meshwell_cross_section,
    get_u_bounds_polygons,
)

__all__ = ["get_meshwell_prisms", "get_meshwell_cross_section", "get_u_bounds_polygons"]
