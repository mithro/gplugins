# SPDX-License-Identifier: MIT
"""Typed contacts: the dataclasses, the gate work-function offset, the old ``contact_info``
adapter and typed contacts on the 2D (gdsfactory cross-section) builder."""

from __future__ import annotations

import math
import os
import warnings

import devsim
import pytest

from gplugins.devsim import doping
from gplugins.devsim.contacts import (
    Contact,
    InterfaceRef,
    LayerRef,
    contacts_from_contact_info,
    delete_circuit,
    interface_kind,
)
from gplugins.devsim.materials import CONSTANTS
from gplugins.devsim.stack1d import Layer1D, Mesh1D, create_1d_stack

N_A = 1.7e17


@pytest.fixture(autouse=True)
def clean_devsim():
    yield
    left = list(devsim.get_device_list())
    for device in left:
        devsim.delete_device(device=device)
    delete_circuit()
    for name in ("extended_solver", "extended_model", "extended_equation"):
        devsim.set_parameter(name=name, value=False)
    assert not left, f"devices left behind: {left}"


def test_contact_dataclasses() -> None:
    c = Contact("g", InterfaceRef("ox", None), "gate", workfunction_offset_v=0.1)
    assert c.circuit is False and c.location.b is None
    assert LayerRef((105, 0)).layer == (105, 0)
    with pytest.raises(ValueError):
        Contact("g", InterfaceRef("ox", None), "schottky")
    with pytest.raises(ValueError):  # a work-function offset is a metal-gate property
        Contact("b", InterfaceRef("si", None), "ohmic", workfunction_offset_v=0.1)
    with pytest.raises(ValueError):
        Contact("", InterfaceRef("si", None), "ohmic")
    with pytest.raises(TypeError):
        Contact("b", "si", "ohmic")
    with pytest.raises(ValueError):
        InterfaceRef("", None)
    with pytest.raises(ValueError):
        LayerRef((1, 2, 3))


def test_interfaces_by_material_pair() -> None:
    assert interface_kind("silicon", "oxide") == "potential"
    assert interface_kind("oxide", "silicon") == "potential"
    assert interface_kind("oxide", "oxide") == "potential"
    assert interface_kind("silicon", "silicon") == "drift_diffusion"
    assert interface_kind("silicon", None) is None  # no equations on the other side


def _metal_gate_stack(name: str, offset: float):
    layers = [
        Layer1D("oxide", "oxide", 0.0152, None),
        Layer1D("body", "silicon", 1.0, doping.constant(N_A), "acceptor"),
    ]
    contacts = [
        Contact("gate", InterfaceRef("oxide", None), "gate", workfunction_offset_v=offset),
        Contact("body", InterfaceRef("body", None), "ohmic"),
    ]
    mesh = Mesh1D({"oxide": (0.0002, 0.0002), "body": (0.0002, 0.02)})
    return create_1d_stack(name, layers, contacts, mesh, CONSTANTS)


def _metal_gate_vfb(offset: float) -> float:
    from gplugins.devsim.tests.test_stack1d import flatband

    dd = _metal_gate_stack(f"wf{offset:+.2f}", offset)
    psi_b = -CONSTANTS.vt_v * math.log(N_A / CONSTANTS.n_i_cm3)
    vfb = flatband(dd, lambda d: d.contact_charge("gate"), psi_b + offset - 0.05, psi_b + offset)
    dd.delete_device()
    return vfb


def test_workfunction_offset_shifts_vfb() -> None:
    """On a metal-gate/oxide/p-Si stack the flat band moves by exactly the offset.

    With offset 0 the gate's Potential is Vg, so the flat band (zero gate charge) is at the
    bulk potential psi_b = -Vt ln(N_A / n_i); the offset shifts it to psi_b + offset.
    """
    psi_b = -CONSTANTS.vt_v * math.log(N_A / CONSTANTS.n_i_cm3)
    v0, v1 = _metal_gate_vfb(0.0), _metal_gate_vfb(0.3)
    print(f"metal gate: Vfb(0) {v0!r} V, Vfb(0.3) {v1!r} V, psi_b {psi_b!r} V")
    assert abs((v1 - v0) - 0.3) <= 1e-3
    assert abs(v0 - psi_b) <= 1e-3


# --------------------------------------------------------------------------------------------
# The contact_info adapter and typed contacts in 2D
# --------------------------------------------------------------------------------------------
def test_contact_info_adapter() -> None:
    from gdsfactory.gpdk import LAYER_STACK

    info = {
        "anode": {
            "gds_layer": (105, 0),
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        }
    }
    with pytest.warns(DeprecationWarning, match="contact_info"):
        contacts, levels = contacts_from_contact_info(info, LAYER_STACK)
    assert contacts == [Contact("anode", InterfaceRef("anode", "slab90"), "ohmic")]
    via = LAYER_STACK.layers["via_contact"]
    level = levels["anode"]
    assert (level.thickness, level.zmin, level.material) == (
        via.thickness,
        via.zmin,
        via.material,
    )
    assert tuple(level.layer.layer) == (105, 0)


def _pn_inputs():
    from gplugins.devsim.tests.test_mesh import pn_slab_inputs

    return pn_slab_inputs(1.0)


def test_typed_contacts_2d_pn_slab(tmp_path, monkeypatch) -> None:
    """The PN slab with typed ``LayerRef`` contacts and the cited constants.

    The contact regions are added to the stack by the caller (the ``via_contact`` level on the
    anode/cathode GDS layers). Unlike the deprecated ``contact_info`` path, they are electrodes
    with no equations, so the device has the same region/contact/interface names but only the
    silicon is solved. At equilibrium the anode-cathode potential difference is the ohmic
    difference 2 Vt ln(1.1e18 / n_i), now with the cited constants.
    """
    from gdsfactory.technology import LayerLevel

    from gplugins.devsim.get_solver import DDComponent

    monkeypatch.chdir(tmp_path)
    kw = _pn_inputs()
    info = kw.pop("contact_info")
    via = kw["full_layer_stack"].layers["via_contact"]
    for name, item in info.items():
        kw["physical_layer_stack"].layers[name] = LayerLevel(
            layer=item["gds_layer"],
            thickness=via.thickness,
            zmin=via.zmin,
            material=via.material,
            sidewall_angle=via.sidewall_angle,
            info=via.info,
        )
    contacts = [
        Contact("anode", LayerRef((105, 0)), "ohmic"),
        Contact("cathode", LayerRef((106, 0)), "ohmic"),
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*contact_info", category=DeprecationWarning)
        c = DDComponent(**kw, contacts=contacts, constants=CONSTANTS)
        c.ddsolver()
    device = c.device
    assert sorted(devsim.get_region_list(device=device)) == [
        "anode",
        "cathode",
        "core",
        "slab90",
    ]
    assert sorted(devsim.get_contact_list(device=device)) == ["anode", "cathode"]
    assert sorted(devsim.get_interface_list(device=device)) == [
        "core___slab90",
        "slab90___anode",
        "slab90___cathode",
    ]
    assert devsim.get_region_list(device=device, contact="anode") == ("slab90",)
    assert c.region_classes == {
        "anode": None,
        "cathode": None,
        "core": "silicon",
        "slab90": "silicon",
    }
    assert "Potential" not in devsim.get_node_model_list(device=device, region="anode")
    assert devsim.get_parameter(device=device, region="slab90", name="T") == 298.15

    def contact_potential(contact):
        nodes = sorted(
            {
                i
                for e in devsim.get_element_node_list(
                    device=device, region="slab90", contact=contact
                )
                for i in e
            }
        )
        pot = devsim.get_node_model_values(device=device, region="slab90", name="Potential")
        return sum(pot[i] for i in nodes) / len(nodes)

    dv = contact_potential("anode") - contact_potential("cathode")
    expected = 2 * CONSTANTS.vt_v * math.log(1.1e18 / CONSTANTS.n_i_cm3)
    print(f"typed PN slab: anode - cathode {dv!r} V, expected {expected!r} V")
    assert abs(dv - expected) < 1e-9
    c.delete_device()


def test_typed_layerref_needs_one_level(tmp_path, monkeypatch) -> None:
    from gplugins.devsim.get_simulation import create_2Duz_simulation

    monkeypatch.chdir(tmp_path)
    kw = _pn_inputs()
    kw.pop("contact_info")
    with pytest.raises(ValueError, match="no level"):
        create_2Duz_simulation(
            **kw, contact_info=None, contacts=[Contact("anode", LayerRef((105, 0)), "ohmic")]
        )
    assert not os.path.exists("temp.msh2")  # it refuses before meshing


def test_ddcomponent_contact_info_is_deprecated(tmp_path, monkeypatch) -> None:
    from gplugins.devsim.get_solver import DDComponent

    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning, match="contact_info"):
        DDComponent(**_pn_inputs(), extended_precision=False)
    with pytest.raises(ValueError, match="not both"):
        DDComponent(
            **_pn_inputs(),
            contacts=[Contact("anode", LayerRef((105, 0)), "ohmic")],
            extended_precision=False,
        )
