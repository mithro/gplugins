# SPDX-License-Identifier: MIT
"""pytest collection for gplugins.devsim.

``python_files`` (pyproject.toml) collects every ``gplugins/*.py``. ``get_solver.py`` replaces
itself in ``sys.modules`` with :mod:`gplugins.devsim.backends.devsim` (an alias, see its
docstring), so pytest's import-file check would refuse it; it holds no tests.
"""

collect_ignore = ["get_solver.py"]
