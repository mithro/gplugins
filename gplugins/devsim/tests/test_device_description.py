# SPDX-License-Identifier: MIT
"""The backend seam: the device description imports no backend, and the DEVSIM backend
consumes it with the same results as the API that predates the seam.

The import check runs ``import gplugins.devsim.device_description`` in a fresh interpreter in
which importing ``devsim``, ``meshwell`` or ``gplugins.devsim.backends`` raises. The package
``gplugins/devsim/__init__.py`` imports DEVSIM itself, so the child registers a bare package
module for ``gplugins.devsim`` (its ``__path__`` only) and the check covers exactly the
description and what it imports. The same check run on ``gplugins.devsim.contacts`` (which
imports DEVSIM) must fail, which shows that the blocker detects a backend import.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import devsim
import pytest

from gplugins.devsim import device_description as dd_mod
from gplugins.devsim import doping
from gplugins.devsim.contacts import Contact, InterfaceRef, delete_circuit
from gplugins.devsim.materials import CONSTANTS
from gplugins.devsim.stack1d import Layer1D, Mesh1D, create_1d_stack

UM = 1e-4  # cm
BLOCKED = ("devsim", "meshwell", "gplugins.devsim.backends")

_CHILD = textwrap.dedent(
    """
    import importlib
    import importlib.abc
    import pathlib
    import sys
    import types

    BLOCKED = tuple(sys.argv[2].split(","))


    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if any(name == b or name.startswith(b + ".") for b in BLOCKED):
                raise ImportError(f"backend import of {name!r}")
            return None


    sys.meta_path.insert(0, Block())
    import gplugins

    pkg = types.ModuleType("gplugins.devsim")
    pkg.__path__ = [str(pathlib.Path(gplugins.__file__).parent / "devsim")]
    sys.modules["gplugins.devsim"] = pkg
    importlib.import_module(sys.argv[1])
    loaded = sorted(m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in BLOCKED))
    assert not loaded, loaded
    print("ok", sys.modules[sys.argv[1]].__file__)
    """
)


def _import_without_backends(module: str, tmp_path) -> subprocess.CompletedProcess:
    child = tmp_path / "import_without_backends.py"
    child.write_text(_CHILD, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(child), module, ",".join(BLOCKED)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def clean_devsim():
    """Fail on, and remove, any device a test leaves; delete leftover circuits and meshes;
    restore DEVSIM precision."""
    yield
    left = list(devsim.get_device_list())
    for device in left:
        devsim.delete_device(device=device)
    delete_circuit()
    for mesh in devsim.get_mesh_list():
        devsim.delete_mesh(mesh=mesh)
    for name in ("extended_solver", "extended_model", "extended_equation"):
        devsim.set_parameter(name=name, value=False)
    assert not left, f"devices left behind: {left}"


# --------------------------------------------------------------------------------------------
# The description imports no backend
# --------------------------------------------------------------------------------------------
def test_device_description_imports_no_backend(tmp_path) -> None:
    res = _import_without_backends("gplugins.devsim.device_description", tmp_path)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == f"ok {dd_mod.__file__}"  # this checkout's module


def test_import_blocker_catches_a_backend_import(tmp_path) -> None:
    """The blocker is not vacuous: contacts.py imports DEVSIM, so the same check fails."""
    res = _import_without_backends("gplugins.devsim.contacts", tmp_path)
    assert res.returncode != 0
    assert "backend import of 'devsim'" in res.stderr


# --------------------------------------------------------------------------------------------
# The public API that predates the seam is unchanged
# --------------------------------------------------------------------------------------------
def test_moved_names_are_the_same_objects() -> None:
    from gplugins.devsim import contacts, materials, stack1d

    for name in (
        "Contact",
        "ContactKind",
        "InterfaceKind",
        "InterfaceRef",
        "LayerRef",
        "check_contact_region",
        "interface_kind",
    ):
        assert getattr(contacts, name) is getattr(dd_mod, name), name
    for name in ("OXIDE_MATERIALS", "SILICON_MATERIALS", "RegionClass", "material_class"):
        assert getattr(materials, name) is getattr(dd_mod, name), name
    for name in ("Layer1D", "Mesh1D", "graded_spacings"):
        assert getattr(stack1d, name) is getattr(dd_mod, name), name
    assert stack1d._check_contacts is dd_mod.check_1d_contacts
    assert stack1d.UM_TO_CM == 1e-4


def test_get_solver_is_the_devsim_backend() -> None:
    import gplugins.devsim as gd
    from gplugins.devsim import get_solver
    from gplugins.devsim.backends import devsim as backend
    from gplugins.devsim.get_solver import DDComponent

    assert get_solver is backend
    assert sys.modules["gplugins.devsim.get_solver"] is backend
    assert DDComponent is backend.DDComponent is gd.DDComponent


# --------------------------------------------------------------------------------------------
# Stack1D: the neutral 1D description
# --------------------------------------------------------------------------------------------
def _layers(gate_dopant="donor", body_dopant="acceptor", tox_um=0.0152):
    return (
        Layer1D("gate_poly", "silicon", 0.2, doping.constant(1e20), gate_dopant),
        Layer1D("oxide", "oxide", tox_um, None),
        Layer1D("body", "silicon", 1.0, doping.constant(1.7e17), body_dopant),
    )


CONTACTS = (
    Contact("gate", InterfaceRef("gate_poly", None), "ohmic"),
    Contact("body", InterfaceRef("body", None), "ohmic"),
)
MESH = Mesh1D({"gate_poly": (0.01, 0.0002), "oxide": (0.0002, 0.0002), "body": (0.0002, 0.02)})


def test_stack1d_is_frozen_plain_data() -> None:
    stack = dd_mod.Stack1D("s", list(_layers()), list(CONTACTS), MESH, CONSTANTS)
    assert isinstance(stack.layers, tuple) and isinstance(stack.contacts, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stack.name = "t"
    assert stack.region_classes == {"gate_poly": "silicon", "oxide": "oxide", "body": "silicon"}
    assert stack.interfaces == ["gate_poly___oxide", "oxide___body"]
    assert stack.contact_ends() == {"gate": 0, "body": 3}
    # Doping depth: the poly gate from its face on the oxide (into the gate), the body from
    # its first face (the oxide interface).
    bounds = stack.boundaries_um()
    assert stack.doping_surfaces_um() == {"gate_poly": (bounds[1], -1), "body": (bounds[2], 1)}


def test_stack1d_nodes_are_graded_spacings() -> None:
    stack = dd_mod.Stack1D("s", _layers(), CONTACTS, MESH, CONSTANTS)
    xs, boundary_node = stack.node_positions_um()
    bounds = stack.boundaries_um()
    assert [xs[i] for i in boundary_node] == bounds
    for i, layer in enumerate(stack.layers):
        inside = xs[boundary_node[i] : boundary_node[i + 1] + 1]
        want = dd_mod.graded_spacings(layer.thickness_um, *MESH.spacing_um[layer.name])
        got = [b - a for a, b in zip(inside, inside[1:])]
        assert got == pytest.approx(want, rel=1e-12, abs=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layers": ()}, "at least one layer"),
        ({"mesh": Mesh1D({"gate_poly": (0.01, 0.0002), "oxide": (0.0002, 0.0002)})}, "no spacing"),
        ({"contacts": CONTACTS[:1]}, "exactly two contacts"),
        (
            {"mesh": Mesh1D({"gate_poly": (0.01, 0.01), "oxide": (0.0002, 0.0002), "body": (0.0002, 0.02)})},
            "spacing jumps",
        ),
    ],
)
def test_stack1d_validates_on_construction(kwargs, match) -> None:
    args = {"name": "s", "layers": _layers(), "contacts": CONTACTS, "mesh": MESH} | kwargs
    with pytest.raises(ValueError, match=match):
        dd_mod.Stack1D(**args)


def test_cvsweep_is_plain_data() -> None:
    sweep = dd_mod.CVSweep("gate", [0.0, 0.5], "qs")
    assert sweep.vg == (0.0, 0.5) and sweep.dv_qs == 1e-3 and sweep.freq_hz is None
    with pytest.raises(ValueError, match="method"):
        dd_mod.CVSweep("gate", [0.0], "hf")


# --------------------------------------------------------------------------------------------
# The DEVSIM backend consumes the description
# --------------------------------------------------------------------------------------------
def _device_state(name: str) -> dict:
    out = {}
    for region in devsim.get_region_list(device=name):
        for var in ("x", "NetDoping", "Potential", "Electrons", "Holes"):
            if var in devsim.get_node_model_list(device=name, region=region):
                out[(region, var)] = tuple(
                    devsim.get_node_model_values(device=name, region=region, name=var)
                )
    return out


def test_create_1d_device_matches_create_1d_stack_bit_for_bit(clean_devsim) -> None:
    from gplugins.devsim.backends.devsim import create_1d_device

    a = create_1d_stack("seam_a", list(_layers()), list(CONTACTS), MESH, CONSTANTS)
    try:
        state_a = _device_state("seam_a")
    finally:
        a.delete_device()
    stack = dd_mod.Stack1D(
        "seam_b",
        _layers(),
        CONTACTS,
        MESH,
        CONSTANTS,
        dd_mod.SolverSettings(absolute_error=1e10, relative_error=1e-10, maximum_iterations=100),
    )
    b = create_1d_device(stack)
    try:
        state_b = _device_state("seam_b")
        assert b.region_classes == stack.region_classes
        assert b.contacts == CONTACTS
        assert (b.atol, b.rtol, b.max_iter) == (1e10, 1e-10, 100)
        assert sorted(devsim.get_interface_list(device="seam_b")) == sorted(stack.interfaces)
        xs, _ = stack.node_positions_um()
        got = [v for r in ("gate_poly", "oxide", "body") for v in state_b[(r, "x")]]
        # Each boundary node appears in both of its regions.
        assert sorted(set(got)) == sorted({x * UM for x in xs})
    finally:
        b.delete_device()
    assert state_a == state_b


def test_run_cv_sweep_matches_sweep_cv(clean_devsim) -> None:
    from gplugins.devsim.backends.devsim import run_cv_sweep
    from gplugins.devsim.cv import sweep_cv

    vg = [-1.0, 0.0, 1.0]
    dev = create_1d_stack("seam_cv", list(_layers()), list(CONTACTS), MESH, CONSTANTS)
    try:
        direct = sweep_cv(dev, "gate", vg, method="qs", dv_qs=2e-3, rel_tol=1e-10)
        planned = run_cv_sweep(
            dev, dd_mod.CVSweep("gate", vg, "qs", dv_qs=2e-3), rel_tol=1e-10
        )
    finally:
        dev.delete_device()
    assert planned == direct


def test_from_description_matches_the_constructor() -> None:
    from gplugins.devsim.backends.devsim import DDComponent

    desc = dd_mod.CrossSection2D(
        component=None,
        xsection_bounds=((0, 0), (1, 0)),
        full_layer_stack=object(),
        physical_layer_stack=None,
        doping_info={"unused": None},
        contacts=CONTACTS,
        constants=CONSTANTS,
        solver=dd_mod.SolverSettings(
            absolute_error=1e11, relative_error=1e-9, maximum_iterations=7, extended_precision=False
        ),
        resolutions={"si": {"resolution": 0.1}},
        background_tag="bg",
    )
    a = DDComponent.from_description(desc, devsim_mesh_name="m", devsim_device_name="d")
    b = DDComponent(
        component=None,
        xsection_bounds=((0, 0), (1, 0)),
        full_layer_stack=desc.full_layer_stack,
        physical_layer_stack=None,
        doping_info={"unused": None},
        contacts=list(CONTACTS),
        constants=CONSTANTS,
        atol=1e11,
        rtol=1e-9,
        max_iter=7,
        extended_precision=False,
        resolutions={"si": {"resolution": 0.1}},
        background_tag="bg",
        devsim_mesh_name="m",
        devsim_device_name="d",
    )
    assert vars(a) == vars(b)
    assert a.legacy is False and a.constants is CONSTANTS
