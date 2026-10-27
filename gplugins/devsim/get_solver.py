# SPDX-License-Identifier: MIT
"""``gplugins.devsim.get_solver`` is :mod:`gplugins.devsim.backends.devsim` (the DEVSIM backend).

The solver moved behind the backend seam (:mod:`gplugins.devsim.device_description`). This
name is kept so that ``from gplugins.devsim.get_solver import DDComponent`` and
``from gplugins.devsim import get_solver`` keep working. It is an alias of the module itself,
not a copy of its names, so that attribute access, assignment and ``monkeypatch.setattr`` on
``get_solver`` act on the backend module that ``DDComponent`` runs in.
"""

import sys

from gplugins.devsim.backends import devsim as _devsim_backend

sys.modules[__name__] = _devsim_backend
