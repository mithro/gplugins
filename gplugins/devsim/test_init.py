# SPDX-License-Identifier: MIT
"""A missing devsim/meshwell is explained on Python < 3.13.

The package is lazy (the backend seam): ``import gplugins.devsim`` needs no DEVSIM, and the
error comes when a name that needs DEVSIM is first accessed.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _ours(name: str) -> bool:
    return name == "gplugins.devsim" or name.startswith("gplugins.devsim.")


@pytest.fixture(autouse=True)
def _drop_reimported_modules():
    """After each test, drop the gplugins.devsim modules the test imported afresh.

    The package is lazy, so ``import gplugins.devsim`` succeeds even with devsim missing and
    imports new module objects; monkeypatch restores the old entries it deleted, and this
    removes the ones that did not exist before, so later tests see one set of classes.
    """
    before = {m for m in sys.modules if _ours(m)}
    yield
    for name in [m for m in sys.modules if _ours(m) and m not in before]:
        del sys.modules[name]


def _forget_gplugins_devsim(monkeypatch) -> None:
    """Make the next ``import gplugins.devsim`` run again; undone after the test.

    Importing a submodule rebinds ``gplugins.devsim`` (the attribute of the ``gplugins``
    package) to the new package object, so that attribute is restored too.
    """
    import gplugins

    if hasattr(gplugins, "devsim"):
        monkeypatch.setattr(gplugins, "devsim", gplugins.devsim)
    for name in [m for m in sys.modules if _ours(m)]:
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

    gd = importlib.import_module("gplugins.devsim")  # lazy: no DEVSIM needed
    with pytest.raises(ImportError, match=r"Python >= 3\.13") as info:
        gd.DDComponent  # noqa: B018
    assert missing in str(info.value)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_no_hint_on_python_313(monkeypatch) -> None:
    _forget_gplugins_devsim(monkeypatch)
    monkeypatch.setitem(sys.modules, "devsim", None)

    gd = importlib.import_module("gplugins.devsim")
    with pytest.raises(ModuleNotFoundError) as info:
        gd.sweep_cv  # noqa: B018
    assert "3.13" not in str(info.value)


def test_neutral_names_need_no_devsim(monkeypatch) -> None:
    """The backend-neutral names load with ``import devsim`` failing."""
    _forget_gplugins_devsim(monkeypatch)
    monkeypatch.setitem(sys.modules, "devsim", None)

    gd = importlib.import_module("gplugins.devsim")
    for name in ("Contact", "InterfaceRef", "LayerRef", "Layer1D", "Mesh1D", "Constants",
                 "CONSTANTS", "CVResult", "PointInfo", "CVConvergenceError", "ConvergenceError"):
        assert getattr(gd, name).__module__.startswith("gplugins.devsim."), name
    with pytest.raises(AttributeError):
        gd.no_such_name  # noqa: B018
