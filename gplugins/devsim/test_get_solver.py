# SPDX-License-Identifier: MIT
"""DEVSIM-only tests of gplugins.devsim.get_solver.DDComponent.

These tests build small devices directly with DEVSIM's own mesher and hand them to DDComponent
in place of its gmsh mesher (``create_2Duz_simulation``, tested in tests/test_mesh.py), so each
checks one solver behaviour on a hand-made mesh.
"""

from __future__ import annotations

import importlib
import warnings

import devsim
import numpy as np
import pytest
from devsim.python_packages import model_create

from gplugins.devsim import get_solver

UM = 1e-4  # cm


def _mesh_1x01um(mesh: str, regions: dict[str, tuple[str, float, float]], contacts=()):
    """A 2D DEVSIM mesh 1 um wide and 0.1 um high, split along x into ``regions``.

    ``regions`` maps a region name to ``(material, x_low_um, x_high_um)``; ``contacts`` are
    ``(name, region, x_um)`` vertical lines.
    """
    devsim.create_2d_mesh(mesh=mesh)
    for pos in (0.0, 0.5 * UM, 1.0 * UM):
        devsim.add_2d_mesh_line(mesh=mesh, dir="x", pos=pos, ps=0.1 * UM)
    for pos in (0.0, 0.1 * UM):
        devsim.add_2d_mesh_line(mesh=mesh, dir="y", pos=pos, ps=0.1 * UM)
    for name, (material, xl, xh) in regions.items():
        devsim.add_2d_region(
            mesh=mesh,
            material=material,
            region=name,
            xl=xl * UM,
            xh=xh * UM,
            yl=0.0,
            yh=0.1 * UM,
        )
    for name, region, x in contacts:
        devsim.add_2d_contact(
            mesh=mesh,
            name=name,
            material="metal",
            region=region,
            xl=x * UM,
            xh=x * UM,
            yl=0.0,
            yh=0.1 * UM,
        )
    names = list(regions)
    for r0, r1 in zip(names, names[1:]):
        devsim.add_2d_interface(
            mesh=mesh,
            name=f"{r0}___{r1}",
            region0=r0,
            region1=r1,
            xl=regions[r0][2] * UM,
            xh=regions[r0][2] * UM,
            yl=0.0,
            yh=0.1 * UM,
        )
    devsim.finalize_mesh(mesh=mesh)


def _ddcomponent(**kwargs) -> get_solver.DDComponent:
    """A DDComponent whose geometry inputs are unused (its mesher is replaced in each test)."""
    return get_solver.DDComponent(
        component=None,
        xsection_bounds=((0, 0), (1, 0)),
        full_layer_stack=object(),
        physical_layer_stack=None,
        doping_info={"unused": None},
        contact_info={},
        extended_precision=False,
        **kwargs,
    )


@pytest.fixture
def clean_devsim():
    """Fail on, and remove, any device or mesh a test leaves behind."""
    yield
    left = list(devsim.get_device_list())
    for device in left:
        devsim.delete_device(device=device)
    assert not left, f"devices left behind: {left}"


def test_get_solver_imports_without_pydantic_deprecations() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(get_solver)
    from_get_solver = [
        str(w.message) for w in caught if w.filename == get_solver.__file__
    ]
    assert not from_get_solver, from_get_solver


def test_delete_device_deletes_the_device_and_its_mesh(clean_devsim) -> None:
    c = _ddcomponent(devsim_mesh_name="m_del", devsim_device_name="d_del")
    _mesh_1x01um("m_del", {"si0": ("si", 0.0, 1.0)})
    devsim.create_device(mesh="m_del", device="d_del")
    c.device = "d_del"  # set by ddsolver

    c.delete_device()

    assert "d_del" not in devsim.get_device_list()
    with pytest.raises(devsim.error):  # the mesh is gone too
        devsim.create_device(mesh="m_del", device="d_del2")


def test_get_refined_mesh_halves_edges_across_a_potential_step(clean_devsim) -> None:
    c = _ddcomponent()
    _mesh_1x01um("m_ref", {"si0": ("si", 0.0, 1.0)})
    devsim.create_device(mesh="m_ref", device="d_ref")
    c.device, c.regions = "d_ref", {"si": ["si0"]}  # set by ddsolver
    model_create.CreateNodeModel("d_ref", "si0", "Potential", "step(x - 5.5e-5)")

    field = c.get_refined_mesh(
        factor=2.0,
        refine_dict={
            "Potential": {"func": lambda x0, x1: abs(x0 - x1), "threshold": 0.05}
        },
        refine_regions=("si",),
    )

    x = np.array(devsim.get_node_model_values(device="d_ref", region="si0", name="x"))
    y = np.array(devsim.get_node_model_values(device="d_ref", region="si0", name="y"))
    n0 = [int(i) for i in devsim.get_edge_model_values(device="d_ref", region="si0", name="node_index@n0")]
    n1 = [int(i) for i in devsim.get_edge_model_values(device="d_ref", region="si0", name="node_index@n1")]
    length = np.array(devsim.get_edge_model_values(device="d_ref", region="si0", name="EdgeLength"))
    crosses = (x[n0] > 5.5e-5) != (x[n1] > 5.5e-5)
    assert crosses.any() and not crosses.all()
    assert field.shape == (len(n0), 4)
    np.testing.assert_allclose(field[:, 0], (x[n0] + x[n1]) / 2 / UM)
    np.testing.assert_allclose(field[:, 1], (y[n0] + y[n1]) / 2 / UM)
    np.testing.assert_array_equal(field[:, 2], 0.0)
    np.testing.assert_allclose(field[:, 3], np.where(crosses, length / 2, length) / UM)
    devsim.delete_device(device="d_ref")
    devsim.delete_mesh(mesh="m_ref")


def test_ddsolver_solves_only_silicon_regions(clean_devsim, tmp_path, monkeypatch) -> None:
    """A device with an oxide region: the silicon models go on the silicon regions only."""
    monkeypatch.chdir(tmp_path)  # ddsolver writes debug.dat
    calls = []

    def fake_create_2Duz_simulation(**kwargs):
        calls.append(kwargs)
        mesh, device = kwargs["devsim_mesh_name"], kwargs["devsim_device_name"]
        _mesh_1x01um(
            mesh,
            {
                "n_si": ("si", 0.0, 0.5),
                "p_si": ("si", 0.5, 0.8),
                "ox": ("Oxide", 0.8, 1.0),
            },
            contacts=(("cathode", "n_si", 0.0), ("anode", "p_si", 0.8)),
        )
        devsim.create_device(mesh=mesh, device=device)
        for region, donors, acceptors in (("n_si", 1e17, 0.0), ("p_si", 0.0, 1e17)):
            for name, value in (("Donors", donors), ("Acceptors", acceptors)):
                model_create.CreateNodeModel(device, region, name, f"{value:g}")
            model_create.CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
        regions = {"si": ["n_si", "p_si"], "Oxide": ["ox"]}
        return device, regions, {}

    monkeypatch.setattr(get_solver, "create_2Duz_simulation", fake_create_2Duz_simulation)
    # Without extended precision (left off so the test changes no global DEVSIM state) the
    # Newton iteration stalls at a relative error of ~2e-7, above DDComponent's default 1e-12.
    c = _ddcomponent(
        devsim_mesh_name="m_dd", devsim_device_name="d_dd", atol=1e8, rtol=1e-5
    )

    c.ddsolver()

    assert calls[0]["devsim_mesh_name"] == "m_dd"
    assert calls[0]["devsim_device_name"] == "d_dd"
    assert c.device == "d_dd"
    for region in ("n_si", "p_si"):
        models = devsim.get_node_model_list(device="d_dd", region=region)
        assert {"Potential", "Electrons", "Holes"} <= set(models), (region, models)
    assert "Potential" not in devsim.get_node_model_list(device="d_dd", region="ox")
    potential = devsim.get_node_model_values(device="d_dd", region="n_si", name="Potential")
    assert all(np.isfinite(potential))
    c.delete_device()


def test_ddsolver_solves_regions_with_netdoping(clean_devsim, tmp_path, monkeypatch) -> None:
    """The silicon models go on every region that has NetDoping, whatever its material.

    create_2Duz_simulation gives the contact regions (material "Aluminum" in the v0.8.0 PN slab)
    NetDoping = 0 (get_simulation.py "Assign doping field contact"), and v0.8.0 solves them as
    intrinsic silicon, coupled to the slab by the Si/Si interface model. That is kept. A region
    without NetDoping (oxide) gets no silicon model, nor do its interfaces.
    """
    monkeypatch.chdir(tmp_path)  # ddsolver writes debug.dat

    def fake_create_2Duz_simulation(**kwargs):
        mesh, device = kwargs["devsim_mesh_name"], kwargs["devsim_device_name"]
        _mesh_1x01um(
            mesh,
            {
                "n_si": ("si", 0.0, 0.5),
                "al": ("Aluminum", 0.5, 0.8),
                "ox": ("Oxide", 0.8, 1.0),
            },
            contacts=(("cathode", "n_si", 0.0),),
        )
        devsim.create_device(mesh=mesh, device=device)
        for name, value in (("Donors", 1e17), ("Acceptors", 0.0)):
            model_create.CreateNodeModel(device, "n_si", name, f"{value:g}")
        model_create.CreateNodeModel(device, "n_si", "NetDoping", "Donors-Acceptors")
        # as create_2Duz_simulation does for a contact region
        n = len(devsim.get_node_model_values(device=device, region="al", name="x"))
        devsim.node_solution(device=device, region="al", name="NetDoping")
        devsim.set_node_values(device=device, region="al", name="NetDoping", values=[0] * n)
        regions = {"si": ["n_si"], "Aluminum": ["al"], "Oxide": ["ox"]}
        return device, regions, {}

    monkeypatch.setattr(get_solver, "create_2Duz_simulation", fake_create_2Duz_simulation)
    c = _ddcomponent(
        devsim_mesh_name="m_nd", devsim_device_name="d_nd", atol=1e8, rtol=1e-5
    )

    c.ddsolver()

    for region in ("n_si", "al"):
        models = devsim.get_node_model_list(device="d_nd", region=region)
        assert {"Potential", "Electrons", "Holes"} <= set(models), (region, models)
    assert "Potential" not in devsim.get_node_model_list(device="d_nd", region="ox")
    assert devsim.get_interface_model_list(device="d_nd", interface="n_si___al")
    assert not devsim.get_interface_model_list(device="d_nd", interface="al___ox")
    c.delete_device()
