# SPDX-License-Identifier: MIT
"""Reading Charon's outputs: the Exodus states, ``currents-loca.dat``, the gate charge and C.

The output Exodus file holds one state per solve (``Unscale Variables is on``: potential in V,
densities in cm^-3); a LOCA sweep writes one per continuation step. ``currents-loca.dat``
holds one row per step with the contact voltages printed to nine significant digits.

**Gate charge.** On the 1D strip the oxide's Laplace equation makes the potential linear across
every oxide interval, so the CVFEM flux through the oxide is ``eps_ox * eps0 * (psi_a - psi_b) /
h`` on every interval; :func:`gate_charge` takes it at the poly face and returns the spread over
the oxide intervals with it. That equals the poly's space charge within 2.8e-13 (nfet) and 8.0e-13
(pfet) relative, and minus the body's within 4.4e-13 and 8.1e-13 (gf180-tcad
``docs/limitations/charon.md``, experiment B6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The voltage resolution (V) of currents-loca.dat: values printed as %.8e of about 5 V.
CURRENTS_RESOLUTION_V = 1e-7


@dataclass(frozen=True)
class States:
    """The states of an Exodus output file: coordinates, times and nodal fields (T, N)."""

    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    nodal: dict[str, np.ndarray]


def _names(var) -> list[str]:
    return ["".join(c.decode() for c in row if c) for row in var[:]]


def read_states(path: Path | str) -> States:
    """The states of the Exodus file ``path`` (needs netCDF4)."""
    import netCDF4

    ds = netCDF4.Dataset(str(path))
    ds.set_auto_mask(False)
    try:
        x = np.array(ds["coordx"][:])
        y = np.array(ds["coordy"][:])
        times = (
            np.array(ds["time_whole"][:])
            if "time_whole" in ds.variables
            else np.zeros(0)
        )
        nodal = {}
        if "name_nod_var" in ds.variables:
            for k, name in enumerate(_names(ds["name_nod_var"]), 1):
                nodal[name] = np.array(ds[f"vals_nod_var{k}"][:])
        return States(x, y, times, nodal)
    finally:
        ds.close()


def read_currents(path: Path | str) -> tuple[list[str], np.ndarray]:
    """``currents-loca.dat``: its column names and rows."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    head = lines[0].split()
    rows = np.array(
        [[float(v) for v in line.split()] for line in lines[1:] if line.strip()]
    )
    return head, rows.reshape(-1, len(head))


def grid_states(voltages: np.ndarray, want: np.ndarray, sub: int) -> np.ndarray:
    """The state indices of the grid ``want`` (V) in a sweep of ``sub`` steps per grid step.

    They are every ``sub``-th state, each within :data:`CURRENTS_RESOLUTION_V` of its grid value.
    One extra state past the last grid point (LOCA's zero-length step onto the bound) is allowed
    when it sits on the last grid value; anything else raises ValueError (a cut or misaligned
    sweep).
    """
    v = np.asarray(voltages, dtype=float)
    want = np.asarray(want, dtype=float)
    idx = np.arange(len(want)) * sub
    if len(v) <= idx[-1]:
        raise ValueError(f"sweep cut: {len(v)} states, the grid needs {idx[-1] + 1}")
    dev = np.abs(v[idx] - want)
    if np.max(dev) > CURRENTS_RESOLUTION_V:
        k = int(np.argmax(dev))
        raise ValueError(f"state {idx[k]} at {v[idx[k]]!r} V, grid point {want[k]!r} V")
    extra = v[idx[-1] + 1 :]
    if len(extra) > 1 or (
        len(extra) == 1 and abs(extra[0] - want[-1]) > CURRENTS_RESOLUTION_V
    ):
        raise ValueError(
            f"unexpected states after the last grid point: {extra.tolist()}"
        )
    return idx


def strip_nodes(states: States) -> np.ndarray:
    """The 1D node indices (the y = 0 nodes 2i) of a strip's states."""
    return np.arange(0, len(states.x), 2)


def gate_charge(
    states: States,
    x_um,
    oxide_nodes: tuple[int, int],
    eps_ox: float,
    eps0_f_per_cm: float,
    index,
) -> tuple[np.ndarray, np.ndarray]:
    """Per state of ``index``: the gate charge per area and the spread of the oxide flux.

    The gate charge (C/cm^2) is the oxide flux at the poly face; the spread is the largest
    relative difference of the flux over the oxide intervals.

    ``oxide_nodes`` are the 1D indices of the oxide's first and last node; ``x_um`` the 1D node
    positions.
    """
    i0, i1 = oxide_nodes
    psi = states.nodal["ELECTRIC_POTENTIAL"][np.asarray(index)][:, 0::2]
    x = np.asarray(x_um, dtype=float) * 1e-4
    flux = (
        eps_ox
        * eps0_f_per_cm
        * (psi[:, i0:i1] - psi[:, i0 + 1 : i1 + 1])
        / np.diff(x[i0 : i1 + 1])
    )
    q = flux[:, 0]
    spread = np.max(np.abs(flux - q[:, None]), axis=1) / np.max(np.abs(q))
    return q, spread


def space_charge(
    states: States, x_um, nodes: tuple[int, int], net_cm3: float, q_c: float, index
) -> np.ndarray:
    """Per state of ``index``: the space charge per area (C/cm^2) of one silicon layer.

    The layer's 1D nodes are ``nodes`` (first, last) and its net doping ``net_cm3`` (N_D - N_A,
    cm^-3); the charge is summed over the strip's control volumes (half of each neighbouring
    interval, the CVFEM dual of a 1D mesh).
    """
    lo, hi = nodes
    sel = np.asarray(index)
    n = states.nodal["ELECTRON_DENSITY"][sel][:, 0::2][:, lo : hi + 1]
    p = states.nodal["HOLE_DENSITY"][sel][:, 0::2][:, lo : hi + 1]
    x = np.asarray(x_um, dtype=float)[lo : hi + 1] * 1e-4
    h = np.diff(x)
    vol = np.zeros(len(x))
    vol[:-1] += h / 2
    vol[1:] += h / 2
    return (q_c * (p - n + net_cm3)) @ vol


def central_difference(
    q_minus: np.ndarray, q_plus: np.ndarray, dv: float
) -> np.ndarray:
    """C = (Q(vg + dv) - Q(vg - dv)) / (2 dv), per unit of Q per V."""
    return (np.asarray(q_plus) - np.asarray(q_minus)) / (2 * dv)
