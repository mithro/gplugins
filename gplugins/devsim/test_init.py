# SPDX-License-Identifier: MIT
"""``import gplugins.devsim`` explains a missing devsim/meshwell on Python < 3.13."""

from __future__ import annotations

import importlib
import sys

import pytest


def _forget_gplugins_devsim(monkeypatch) -> None:
    """Make the next ``import gplugins.devsim`` run again; undone after the test."""
    for name in [
        m for m in sys.modules if m == "gplugins.devsim" or m.startswith("gplugins.devsim.")
    ]:
        monkeypatch.delitem(sys.modules, name)


# Only devsim: nothing gplugins.devsim imports needs meshwell yet (the meshwell 2.5 mesher
# comes in a later change), so a missing meshwell cannot be provoked here.
@pytest.mark.parametrize("missing", ["devsim"])
def test_import_hint_on_python_before_313(monkeypatch, missing) -> None:
    _forget_gplugins_devsim(monkeypatch)
    for name in [m for m in sys.modules if m == missing or m.startswith(f"{missing}.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, missing, None)  # `import <missing>` now fails
    monkeypatch.setattr(sys, "version_info", (3, 12, 9, "final", 0))

    with pytest.raises(ImportError, match=r"Python >= 3\.13") as info:
        importlib.import_module("gplugins.devsim")
    assert missing in str(info.value)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_no_hint_on_python_313(monkeypatch) -> None:
    _forget_gplugins_devsim(monkeypatch)
    monkeypatch.setitem(sys.modules, "devsim", None)

    with pytest.raises(ModuleNotFoundError) as info:
        importlib.import_module("gplugins.devsim")
    assert "3.13" not in str(info.value)
