"""gplugins.devsim: TCAD device description and simulator backends (DEVSIM first).

The package imports nothing at import time: every public name is loaded on first access
(PEP 562 ``__getattr__``). So the backend-neutral layer (:mod:`gplugins.devsim.device_description`,
:mod:`gplugins.devsim.results`, :mod:`gplugins.devsim.constants`, :mod:`gplugins.devsim.doping`)
and a backend without DEVSIM (``gplugins.devsim.backends.charon``) import in an image that has
no DEVSIM. A name that needs DEVSIM or meshwell raises, on access, the ImportError that its
import raises; on Python < 3.13 that error explains that ``gplugins[devsim]`` installs them only
on Python >= 3.13.
"""

from __future__ import annotations

import importlib
import platform
import sys

# Public name -> the module that defines it. The neutral ones need no simulator.
_NAMES = {
    "CONSTANTS": "gplugins.devsim.constants",
    "Constants": "gplugins.devsim.constants",
    "Contact": "gplugins.devsim.device_description",
    "InterfaceRef": "gplugins.devsim.device_description",
    "LayerRef": "gplugins.devsim.device_description",
    "Layer1D": "gplugins.devsim.device_description",
    "Mesh1D": "gplugins.devsim.device_description",
    "CVResult": "gplugins.devsim.results",
    "CVConvergenceError": "gplugins.devsim.results",
    "ConvergenceError": "gplugins.devsim.results",
    "PointInfo": "gplugins.devsim.results",
    # DEVSIM backend
    "DDComponent": "gplugins.devsim.backends.devsim",
    "create_1d_stack": "gplugins.devsim.stack1d",
    "sweep_cv": "gplugins.devsim.cv",
    "PINWaveguide": "gplugins.devsim.get_simulation_xsection",
    "alpha_to_k": "gplugins.devsim.get_simulation_xsection",
    "clear_devsim_cache": "gplugins.devsim.get_simulation_xsection",
    "dalpha_carriers": "gplugins.devsim.get_simulation_xsection",
    "dn_carriers": "gplugins.devsim.get_simulation_xsection",
    "k_to_alpha": "gplugins.devsim.get_simulation_xsection",
}

__all__ = sorted(_NAMES)
__version__ = "0.0.1"

_logged = False


def _import(module: str):
    try:
        mod = importlib.import_module(module)
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
    global _logged
    if not _logged and "devsim" in sys.modules and sys.modules["devsim"] is not None:
        from gdsfactory import logger

        tcad = sys.modules["devsim"]
        logger.info(f"DEVSIM {tcad.__version__!r} installed at {tcad.__path__!r}")
        _logged = True
    return mod


def __getattr__(name: str):
    module = _NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_NAMES))
