from __future__ import annotations

import platform
import sys

try:
    import devsim as tcad

    from gplugins.devsim.get_simulation_xsection import (
        PINWaveguide,
        alpha_to_k,
        clear_devsim_cache,
        dalpha_carriers,
        dn_carriers,
        k_to_alpha,
    )
    from gplugins.devsim.contacts import Contact, InterfaceRef, LayerRef
    from gplugins.devsim.convergence import ConvergenceError
    from gplugins.devsim.get_solver import DDComponent
    from gplugins.devsim.materials import CONSTANTS, Constants
    from gplugins.devsim.stack1d import Layer1D, Mesh1D, create_1d_stack
except ModuleNotFoundError as e:
    # The devsim extra installs devsim and meshwell only on Python >= 3.13 (meshwell 2.5
    # needs it), so on older Python they are missing even with gplugins[devsim].
    if e.name in ("devsim", "meshwell") and sys.version_info < (3, 13):
        raise ImportError(
            f"gplugins.devsim needs {e.name}, which gplugins[devsim] installs only on "
            f"Python >= 3.13 (meshwell 2.5 requires it); this is Python "
            f"{platform.python_version()}"
        ) from e
    raise

from gdsfactory import logger

logger.info(f"DEVSIM {tcad.__version__!r} installed at {tcad.__path__!r}")

__all__ = [
    "CONSTANTS",
    "Constants",
    "Contact",
    "ConvergenceError",
    "DDComponent",
    "InterfaceRef",
    "Layer1D",
    "LayerRef",
    "Mesh1D",
    "PINWaveguide",
    "alpha_to_k",
    "clear_devsim_cache",
    "create_1d_stack",
    "dalpha_carriers",
    "dn_carriers",
    "k_to_alpha",
]
__version__ = "0.0.1"
