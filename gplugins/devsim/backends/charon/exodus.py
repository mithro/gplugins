# SPDX-License-Identifier: MIT
"""Exodus II meshes for Charon, written from the device description's nodes without moving one.

Charon has no 1D mode and reads only Exodus II (Charon User Manual v2.2, section 3). A
:class:`~gplugins.devsim.device_description.Stack1D` becomes a strip one element wide
(:func:`strip_of`): node ``i`` of the stack (``Stack1D.node_positions_um``) is nodes ``2i``
(y = 0) and ``2i + 1`` (y = width) of the strip, one QUAD4 element per interval, an element
block per layer named as the layer, and a side set per contact at the strip's two ends.
Coordinates are in um, which Charon assumes.

The writer (:func:`write_exodus`) is a thin netCDF classic writer of the Exodus II variables
Charon's SEACAS reader needs. It never reorders nodes, and its QA record carries no date, so two
writes of one mesh are byte-identical.

A side set may not share a name with an element block: Charon 2.2 then aborts with MPI error
code 14 and no message (experiment B1/B2, gf180-tcad ``docs/limitations/charon.md``). The side
sets are therefore named ``<contact>_contact`` when a contact is named like a layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The strip width (um): the C per area does not depend on it (tested).
STRIP_WIDTH_UM = 1.0

#: Exodus name length (32 characters and a terminator).
LEN_NAME = 33


@dataclass(frozen=True)
class Block:
    """An element block: its name, Exodus element type and 0-based connectivity (E, k)."""

    name: str
    elem_type: str
    connectivity: np.ndarray


@dataclass(frozen=True)
class SideSet:
    """A side set: its name and, per side, the 1-based element id and 1-based local side."""

    name: str
    elements: tuple[int, ...]
    sides: tuple[int, ...]


@dataclass(frozen=True)
class Mesh:
    """A 2D Exodus mesh: node coordinates (N, 2) in um, element blocks and side sets."""

    xy_um: np.ndarray
    blocks: tuple[Block, ...]
    side_sets: tuple[SideSet, ...]


def side_set_name(contact: str, regions) -> str:
    """The side set name of ``contact``: its name, or ``<name>_contact`` if a region has it."""
    return f"{contact}_contact" if contact in set(regions) else contact


def strip_of(stack, width_um: float = STRIP_WIDTH_UM) -> Mesh:
    """The strip mesh of a ``Stack1D`` (see the module notes)."""
    if not (width_um > 0 and np.isfinite(width_um)):
        raise ValueError(f"width_um must be > 0, got {width_um!r}")
    xs, bnodes = stack.node_positions_um()
    n = len(xs)
    xy = np.zeros((2 * n, 2))
    xy[0::2, 0] = xs
    xy[1::2, 0] = xs
    xy[1::2, 1] = width_um
    blocks, first_elem = [], {}
    eid = 0
    for i, layer in enumerate(stack.layers):
        js = range(bnodes[i], bnodes[i + 1])
        conn = np.array([(2 * j, 2 * (j + 1), 2 * (j + 1) + 1, 2 * j + 1) for j in js],
                        dtype=np.int64)  # fmt: skip
        first_elem[i] = (eid + 1, eid + len(conn))
        eid += len(conn)
        blocks.append(Block(layer.name, "QUAD4", conn))
    regions = [layer.name for layer in stack.layers]
    side_sets = []
    for contact, end in sorted(stack.contact_ends().items(), key=lambda kv: kv[1]):
        if end == 0:  # the x = x[0] edge of the first element: QUAD4 side 4 (nodes 4-1)
            elem, side = first_elem[0][0], 4
        else:  # the x = x[-1] edge of the last element: QUAD4 side 2 (nodes 2-3)
            elem, side = first_elem[len(stack.layers) - 1][1], 2
        side_sets.append(SideSet(side_set_name(contact, regions), (elem,), (side,)))
    return Mesh(xy, tuple(blocks), tuple(side_sets))


def _chars(names, width: int = LEN_NAME) -> np.ndarray:
    a = np.zeros((len(names), width), dtype="S1")
    for i, name in enumerate(names):
        b = name.encode("ascii")
        if len(b) >= width:
            raise ValueError(f"name {name!r} longer than {width - 1} characters")
        a[i, : len(b)] = np.frombuffer(b, dtype="S1")
    return a


def write_exodus(
    mesh: Mesh, path: Path | str, title: str = "gplugins.devsim charon"
) -> None:
    """Write ``mesh`` as an Exodus II file (netCDF 64-bit offset, 8-byte floats)."""
    import netCDF4

    xy = np.asarray(mesh.xy_um, dtype=np.float64)
    names = [b.name for b in mesh.blocks]
    clash = {s.name for s in mesh.side_sets} & set(names)
    if clash:
        raise ValueError(f"side sets named like element blocks: {sorted(clash)}")
    ds = netCDF4.Dataset(str(path), "w", format="NETCDF3_64BIT_OFFSET")
    try:
        for key, value in (("api_version", np.float32(8.03)), ("version", np.float32(8.03)),
                           ("floating_point_word_size", np.int32(8)),
                           ("file_size", np.int32(1)), ("maximum_name_length", np.int32(32)),
                           ("int64_status", np.int32(0)), ("title", title)):  # fmt: skip
            ds.setncattr(key, value)
        for dim, size in (("len_string", 33), ("len_line", 81), ("four", 4),
                          ("len_name", LEN_NAME), ("time_step", None), ("num_dim", 2),
                          ("num_nodes", len(xy)),
                          ("num_elem", sum(len(b.connectivity) for b in mesh.blocks)),
                          ("num_el_blk", len(mesh.blocks)),
                          ("num_side_sets", len(mesh.side_sets)),
                          ("num_qa_rec", 1)):  # fmt: skip
            ds.createDimension(dim, size)
        ds.createVariable("time_whole", "f8", ("time_step",))
        ds.createVariable("coordx", "f8", ("num_nodes",))[:] = xy[:, 0]
        ds.createVariable("coordy", "f8", ("num_nodes",))[:] = xy[:, 1]
        ds.createVariable("coor_names", "S1", ("num_dim", "len_name"))[:] = _chars(
            ["x", "y"]
        )
        ds.createVariable("eb_status", "i4", ("num_el_blk",))[:] = 1
        v = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        v.setncattr("name", "ID")
        v[:] = np.arange(1, len(mesh.blocks) + 1)
        ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))[:] = _chars(
            names
        )
        for k, b in enumerate(mesh.blocks, 1):
            conn = np.asarray(b.connectivity, dtype=np.int64)
            ds.createDimension(f"num_el_in_blk{k}", conn.shape[0])
            ds.createDimension(f"num_nod_per_el{k}", conn.shape[1])
            v = ds.createVariable(f"connect{k}", "i4",
                                  (f"num_el_in_blk{k}", f"num_nod_per_el{k}"))  # fmt: skip
            v.setncattr("elem_type", b.elem_type)
            v[:] = conn + 1
        if mesh.side_sets:
            ds.createVariable("ss_status", "i4", ("num_side_sets",))[:] = 1
            v = ds.createVariable("ss_prop1", "i4", ("num_side_sets",))
            v.setncattr("name", "ID")
            v[:] = np.arange(1, len(mesh.side_sets) + 1)
            ds.createVariable("ss_names", "S1", ("num_side_sets", "len_name"))[:] = (
                _chars([s.name for s in mesh.side_sets])
            )
            for k, s in enumerate(mesh.side_sets, 1):
                ds.createDimension(f"num_side_ss{k}", len(s.elements))
                ds.createVariable(f"elem_ss{k}", "i4", (f"num_side_ss{k}",))[:] = (
                    s.elements
                )
                ds.createVariable(f"side_ss{k}", "i4", (f"num_side_ss{k}",))[:] = (
                    s.sides
                )
        ds.createVariable("qa_records", "S1", ("num_qa_rec", "four", "len_string"))[
            0
        ] = _chars(["gplugins", "devsim", "", ""], 33)
    finally:
        ds.close()
