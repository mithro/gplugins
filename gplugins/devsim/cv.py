# SPDX-License-Identifier: MIT
"""Gate capacitance-voltage sweeps: quasi-static and small-signal AC Cgg (and Cxg).

:func:`sweep_cv` sweeps the gate contact of a solved
:class:`~gplugins.devsim.get_solver.DDComponent` over ``vg`` with every other contact held at
0 V, and returns a :class:`CVResult`. DEVSIM works in cm: Cgg is F/cm^2 on a 1D device and F/cm
on a 2D one (``CVResult.unit``).

Two methods:

``"qs"`` (quasi-static, low frequency)
    Cgg = dQ_gate/dVg by a central difference: a separate solve at ``vg - dv_qs`` and
    ``vg + dv_qs`` for every grid point (neighbouring grid points are never used). Each solve
    is the equilibrium Poisson-Boltzmann problem with flat quasi-Fermi levels: the potential
    is the only unknown, and in each silicon region n = n_i exp((psi - phi)/V_t) and
    p = n_i exp((phi - psi)/V_t), with phi the bias of that region's contacts (0 V for the
    body side, Vg for a poly gate). With every non-gate contact at 0 V and the gate isolated by
    the oxide, this is exactly the zero-current DC steady state of drift-diffusion, so it is
    the LF/QS curve, and it converges in inversion. Q_gate is the gate electrode's charge:
    DEVSIM's gate contact charge plus the space charge of the silicon regions on the gate's
    side of the insulator (none for a metal gate).

``"ac"`` (small-signal AC at ``freq_hz``)
    Every contact gets its own voltage source (the gate the swept one, every other one at 0 V
    DC), the drift-diffusion DC point is solved, and a 1 V AC excitation on the gate source
    alone gives each source's AC current. Cgg = Im(Y_gg)/(2 pi f) and, for every non-gate
    contact x, Cxg = -dQ_x/dVg = -Im(I_x)/(2 pi f) (I_x the AC current into the device at x),
    so that charge conservation reads Cgg = sum over x != gate of Cxg. Drift-diffusion with
    only a body contact does not converge at the onset of inversion when Newton is stepped
    from depletion (gplugins Task 6). The DC point there is still the equilibrium, so each DD
    point starts from the Poisson-Boltzmann solution at the same bias (``Potential`` from it,
    ``Electrons``/``Holes`` from the flat-quasi-Fermi-level densities), which is already the
    DD solution up to round-off. With only a body contact the inversion layer responds in
    about 2 tau_g N_A/n_i (~4e2 s for 1e-5 s, 1.7e17 cm^-3; Nicollian & Brews), so 1D AC is
    the high-frequency curve in inversion at any practical frequency.

Convergence. Every solve uses DEVSIM's relative error ``rel_tol`` and a non-binding solver
absolute error (DEVSIM takes one pair for every equation, and the carriers' absolute update is
in cm^-3). After it converges, the last iteration's per-equation errors are checked: every
equation's relative error <= ``rel_tol`` and every ``PotentialEquation``'s absolute error (the
potential update, V) <= ``abs_tol``; Newton continues within ``max_iterations`` if the
potential update is still larger. They are recorded per point (:class:`PointInfo`). A bias
point that does not converge, or not at these tolerances, raises :class:`CVConvergenceError`
naming every such point, after the whole sweep was attempted; no point is ever dropped,
interpolated or accepted at a relaxed tolerance. To reach a grid point the gate is ramped from
the previous one in steps of at most 0.1 V (halved on a failed step, down to 1/16 of that).

DEVSIM's circuit is global: ``delete_device`` and :func:`~gplugins.devsim.contacts.delete_circuit`
delete the whole circuit, and source and node names are not per device. So the AC sweep names
its sources ``Vcv__<device>__<contact>`` on nodes ``cv__<device>__<contact>``, refuses to run
while any device's contact uses a circuit node, and deletes the circuit when it ends. The
device must have been built with ``circuit=False`` contacts (else ``ValueError``, both
methods). A pre-existing standalone circuit (elements attached to no device) cannot be seen
beforehand (probing with ``get_circuit_node_list`` is unsafe without a circuit), but DEVSIM
refuses to add circuit nodes once an analysis has been performed with a circuit present: the
first source then fails, the AC sweep raises ``ValueError`` naming that likely cause, and
nothing is deleted (the circuit is deleted only if one of the sweep's own sources was added).

The sweep leaves the device as it found it: its equations, contact biases and solution are
restored afterwards (also when it raises), as is DEVSIM's extended precision setting, which
the sweep switches on (``extended_precision=True``, the default) or, for ``"qs"`` only, off
(``False``: double precision, about twice as fast on large 2D meshes, with the same tolerances
checked). The AC solve needs extended precision: in double precision the 1 kHz AC Cgg of the
1D nfet stack near the onset of inversion moves by 1.1e-3 relative although every DC point
converges to rel_tol 1e-10, so ``"ac"`` refuses ``extended_precision=False``.

In double precision the relative criterion can also stall where a potential is near 0 V: on a
2D MOS capacitor with a charge-free dielectric next to the gate, one QS point of 201 stopped at
a relative update of 7e-10 (absolute 1.7e-16 V, round-off) and could not reach 1e-10. With
``precision_fallback=True`` (QS in double precision only) a solve that fails in double precision
is retried, from where it stopped, in extended precision at the same tolerances, then the sweep
returns to double; the grid points that needed it are listed in ``CVResult.extended_fallback``.
Nothing is relaxed: a point that fails in extended precision too still fails.

Round-off near psi = 0. DEVSIM's relative error of an equation is the largest
|update| / (|value| + 1e-10) over its nodes (``Equation.cc``, ``minError`` 1e-10), so where a
potential passes through 0 V in a charge-free dielectric the 1e-10 relative criterion asks for
an update below about 1e-20 V, beyond the linear solve's round-off (about 1e-16 V, in double and
in extended precision alike: on the gf180-tcad 2D MOS capacitor the update stalls at
1.6-1.7e-16 V while the relative error wanders between 1e-10 and 7e-10 for 45 iterations).
``roundoff_accept=True`` (QS only, opt-in) accepts such a non-converged solve, and records it
(``CVResult.roundoff_accepted``), only if, on its final iterations:

(a) every ``PotentialEquation`` absolute update is <= ``abs_tol`` (unchanged);
(b) the largest potential update of each of the last :data:`ROUNDOFF_WINDOW` iterations is
    <= :data:`ROUNDOFF_K` * eps * max|psi| (eps the double machine epsilon, max|psi| over the
    device): round-off of the potentials, not an unconverged update;
(c) the relative error has stagnated: no new minimum in the last :data:`ROUNDOFF_WINDOW`
    iterations (a plateau, not slow convergence);
(d) in every region whose relative error exceeds ``rel_tol`` a node has
    |psi| <= (its largest update) / ``rel_tol``, which is where the definition puts a violating
    node; that region and its smallest |psi| are recorded.

DEVSIM restores the solution from before a non-converged solve (``math/Newton.cc``,
``RestoreSolutions``). So (a)-(c) are checked on the failed solve's iteration history, and the
point is then solved again from the restored state with DEVSIM's absolute error set to the
round-off bound and its relative error non-binding: DEVSIM stops, converged, at the round-off
iterate and keeps that state. The update bound and (d) are checked on that kept state, whose
charge the sweep then uses.

Progress. A long sweep (a fine 2D mesh takes hours) can report each bias point through
``progress(stage, index, vg)``, called after the point is done (converged or not), in sweep
order: stage ``"qs"`` for every point of a QS sweep; ``"pb"`` (its Poisson-Boltzmann point)
and then ``"ac"`` (its drift-diffusion point and AC solve) for every point of an AC sweep.
"""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Callable, Sequence
from types import MappingProxyType
from typing import Literal

import devsim
from devsim.python_packages import model_create

from gplugins.devsim import contacts as contacts_mod
from gplugins.devsim.convergence import solve_dc

# The result and error types are backend-neutral: they live in gplugins.devsim.results (the
# backend seam) and are re-exported here unchanged.
from gplugins.devsim.results import (
    ConvergenceError,  # noqa: F401 (re-exported, as before)
    CVConvergenceError,
    CVResult,
    PointInfo,
    RoundoffPoint,
)

__all__ = ["CVConvergenceError", "CVResult", "PointInfo", "RoundoffPoint", "sweep_cv"]

#: DEVSIM's absolute error for every solve: not binding (the potential update is checked
#: against ``abs_tol`` per equation afterwards; a carrier's update is in cm^-3).
SOLVER_ABSOLUTE_ERROR = 1e30
#: The largest gate step (V) when ramping between grid points, and how often it may be halved.
MAX_STEP_V = 0.1
MAX_HALVINGS = 4

#: roundoff_accept: the k of the round-off bound k * eps * max|psi|. At the 2D point that
#: motivated it, max|psi| is about 0.6 V (n+ source/drain +0.60 V, p body -0.43 V), so the bound
#: is 32 * 2.22e-16 * 0.6 V = 4.3e-15 V: 25 times the observed floor (1.7e-16 V) and 230 times
#: below the default abs_tol (1e-12 V). At max|psi| = 6 V it is 4.3e-14 V, still 23 times below.
ROUNDOFF_K = 32
#: roundoff_accept: the iterations over which the update must be at round-off and the relative
#: error must make no new minimum.
ROUNDOFF_WINDOW = 10
_EPS = sys.float_info.epsilon
#: The relative error of the round-off re-solve: non-binding (its acceptance is checked after).
_NON_BINDING_REL = 1e30
_QFL = "cv_qfl"  # silicon region parameter: the region's flat quasi-Fermi level (V)
_N, _P, _CHARGE = "cv_qs_electrons", "cv_qs_holes", "cv_qs_charge"
_CONTINUITY = ("ElectronContinuityEquation", "HoleContinuityEquation")
_PRECISION = ("extended_solver", "extended_model", "extended_equation")


# --------------------------------------------------------------------------------------------
# Solving with per-equation checks
# --------------------------------------------------------------------------------------------
def _errors(info: dict, device: str) -> tuple[float, float]:
    """(largest relative error, largest PotentialEquation absolute error) of the last iteration.

    Raises :class:`ConvergenceError` if the last iteration has no ``PotentialEquation`` entry for
    ``device``, so the check can never pass vacuously.
    """
    rel, pot_abs, seen = 0.0, 0.0, False
    for dev in info["iterations"][-1]["devices"]:
        if dev["name"] != device:
            continue
        for region in dev["regions"]:
            for eq in region["equations"]:
                rel = max(rel, eq["relative_error"])
                if eq["name"] == "PotentialEquation":
                    pot_abs = max(pot_abs, eq["absolute_error"])
                    seen = True
    if not seen:
        raise ConvergenceError(
            f"the solve info has no PotentialEquation errors for device {device!r}; "
            "the tolerances cannot be checked",
            info,
        )
    return rel, pot_abs


def _solve(device: str, *, rel_tol: float, abs_tol: float, max_iterations: int) -> PointInfo:
    """A DC solve meeting ``rel_tol`` on every equation and ``abs_tol`` V on the potential.

    Raises :class:`ConvergenceError` otherwise. Newton continues (within ``max_iterations`` in
    total) while the potential update is above ``abs_tol``; nothing is relaxed.
    """
    total = 0
    while True:
        info = solve_dc(
            absolute_error=SOLVER_ABSOLUTE_ERROR,
            relative_error=rel_tol,
            maximum_iterations=max_iterations - total,
        )
        total += len(info["iterations"])
        rel, pot_abs = _errors(info, device)
        if rel <= rel_tol and pot_abs <= abs_tol:
            return PointInfo(True, total, rel, pot_abs)
        if total >= max_iterations:
            raise ConvergenceError(
                f"converged by DEVSIM but not at the requested tolerances after {total} "
                f"iterations: relative {rel!r} (<= {rel_tol!r}), potential absolute "
                f"{pot_abs!r} V (<= {abs_tol!r})",
                info,
            )


def _potential_errors(device: str, iteration: dict) -> dict[str, tuple[float, float]] | None:
    """``{region: (rel, abs)}`` of the PotentialEquation of one iteration; None if any other
    equation is above its relative tolerance (not the potential round-off case) or the device
    is missing."""
    devs = [d for d in iteration["devices"] if d["name"] == device]
    if not devs:
        return None
    out = {}
    for region in devs[0]["regions"]:
        for eq in region["equations"]:
            if eq["name"] == "PotentialEquation":
                out[region["name"]] = (eq["relative_error"], eq["absolute_error"])
    return out


def _max_abs_psi(device: str, regions) -> dict[str, list[float]]:
    return {
        r: [abs(v) for v in devsim.get_node_model_values(device=device, region=r, name="Potential")]
        for r in regions
    }


def _roundoff_history(device: str, info: dict | None, *, rel_tol: float, abs_tol: float):
    """The round-off bound (V) if a failed solve's iteration history meets conditions (a)-(c)
    of ``roundoff_accept`` (module notes), else None.

    DEVSIM restores the pre-solve solution of a non-converged solve, so max|psi| here is that
    of the restored state (the previous bias, a few mV away: the same scale); (d) is checked on
    the re-solved state (:meth:`_Session._roundoff_resolve`).
    """
    iterations = (info or {}).get("iterations") or ()
    if len(iterations) <= ROUNDOFF_WINDOW:
        return None
    rels, pots = [], []
    for it in iterations:
        errs = _potential_errors(device, it)
        if not errs:
            return None
        devs = [d for d in it["devices"] if d["name"] == device][0]
        for region in devs["regions"]:
            for eq in region["equations"]:
                if eq["name"] != "PotentialEquation" and eq["relative_error"] > rel_tol:
                    return None  # only potential round-off is accepted (QS has no carriers)
        rels.append(max(r for r, _ in errs.values()))
        pots.append(max(a for _, a in errs.values()))
    last = _potential_errors(device, iterations[-1])
    # (a) every potential update within abs_tol
    if any(a > abs_tol for _, a in last.values()):
        return None
    psi = _max_abs_psi(device, last)
    bound = ROUNDOFF_K * _EPS * max(max(v) for v in psi.values())
    # (b) round-off updates over the whole window
    if any(p > bound for p in pots[-ROUNDOFF_WINDOW:]):
        return None
    # (c) a plateau: no new minimum of the relative error in the window
    if min(rels[-ROUNDOFF_WINDOW:]) < min(rels[:-ROUNDOFF_WINDOW]):
        return None
    return bound, len(last)


def _merge(a: PointInfo, b: PointInfo) -> PointInfo:
    return PointInfo(
        a.converged and b.converged,
        a.iterations + b.iterations,
        max(a.rel_error, b.rel_error),
        max(a.abs_error, b.abs_error),
    )


def _ident(name: str) -> str:
    return re.sub(r"\W", "_", name)


# --------------------------------------------------------------------------------------------
# A sweep session: switches the device between Poisson-Boltzmann and drift-diffusion(+circuit)
# --------------------------------------------------------------------------------------------
class _Session:
    def __init__(
        self,
        dd,
        gate: str,
        *,
        rel_tol: float,
        abs_tol: float,
        max_iterations: int,
        precision_fallback: bool = False,
        roundoff_accept: bool = False,
    ):
        self.dd = dd
        self.fallback = precision_fallback
        self.fallback_used = False
        self.roundoff = roundoff_accept
        self.roundoff_hits: list[RoundoffPoint] = []
        self.index = -1
        self.device = device = dd.device
        self.gate = gate
        self.tol = {"rel_tol": rel_tol, "abs_tol": abs_tol, "max_iterations": max_iterations}
        self.contacts = list(devsim.get_contact_list(device=device))
        if gate not in self.contacts:
            raise ValueError(f"gate {gate!r} is not a contact of {device!r}: {self.contacts}")
        self.others = [c for c in self.contacts if c != gate]
        self.regions = list(devsim.get_region_list(device=device))
        classes = dd.region_classes
        self.silicon = [r for r in self.regions if classes.get(r) == "silicon"]
        self.contact_region = {
            c: devsim.get_region_list(device=device, contact=c)[0] for c in self.contacts
        }

        # The saved device: equations, solution, biases, precision.
        self.eqs = {
            r: {
                e: devsim.get_equation_command(device=device, region=r, name=e)
                for e in devsim.get_equation_list(device=device, region=r)
            }
            for r in self.regions
        }
        self.ceqs = {
            c: {
                e: devsim.get_contact_equation_command(device=device, contact=c, name=e)
                for e in devsim.get_contact_equation_list(device=device, contact=c)
            }
            for c in self.contacts
        }
        self.interfaces = list(devsim.get_interface_list(device=device))
        self.ieqs = {
            i: {
                e: devsim.get_interface_equation_command(device=device, interface=i, name=e)
                for e in devsim.get_interface_equation_list(device=device, interface=i)
            }
            for i in self.interfaces
        }
        for c, eqs in self.ceqs.items():
            for e, cmd in eqs.items():
                if cmd.get("circuit_node"):
                    raise ValueError(
                        f"contact {c!r} of {device!r} is attached to circuit node "
                        f"{cmd['circuit_node']!r}; sweep_cv sources every contact itself, so "
                        "build the device with circuit=False contacts"
                    )
        self.values = {
            (r, v): list(devsim.get_node_model_values(device=device, region=r, name=v))
            for r in self.regions
            for v in ("Potential", "Electrons", "Holes")
            if v in devsim.get_node_model_list(device=device, region=r)
        }
        self.biases = {
            c: devsim.get_parameter(device=device, name=contacts_mod.bias_name(c))
            for c in self.contacts
        }
        self.precision = {}
        for name in _PRECISION:
            try:
                self.precision[name] = devsim.get_parameter(name=name)
            except devsim.error:
                self.precision[name] = False

        self.gate_regions = self._gate_side_regions()
        self.mode = "dd"
        self.circuit = False
        self.added_sources = 0
        self.v_gate = float(self.biases[gate])
        dim = devsim.get_dimension(device=device)
        if dim not in (1, 2):
            raise ValueError(f"sweep_cv supports 1D and 2D devices, {device!r} is {dim}D")
        self.unit = "F/cm^2" if dim == 1 else "F/cm"
        names = {c: _ident(c) for c in self.contacts}
        if len(set(names.values())) != len(names):
            raise ValueError(f"contact names {self.contacts} collide as circuit names")
        dev = _ident(device)
        self.node = {c: f"cv__{dev}__{names[c]}" for c in self.contacts}
        self.source = {c: f"Vcv__{dev}__{names[c]}" for c in self.contacts}

    def _gate_side_regions(self) -> list[str]:
        """The silicon regions connected to the gate contact (its electrode); validates."""
        adj = {r: set() for r in self.silicon}
        for i in self.interfaces:
            r0, r1 = devsim.get_region_list(device=self.device, interface=i)
            if r0 in adj and r1 in adj:
                adj[r0].add(r1)
                adj[r1].add(r0)
        seen: set[str] = set()
        gate_side: list[str] = []
        for start in self.silicon:
            if start in seen:
                continue
            component, todo = [], [start]
            seen.add(start)
            while todo:
                r = todo.pop()
                component.append(r)
                for s in adj[r] - seen:
                    seen.add(s)
                    todo.append(s)
            on = [c for c in self.contacts if self.contact_region[c] in component]
            if not on:
                raise ValueError(
                    f"silicon regions {sorted(component)} of {self.device!r} have no contact: "
                    "their Fermi level is undefined"
                )
            if self.gate in on:
                if len(on) > 1:
                    raise ValueError(
                        f"the gate {self.gate!r} shares silicon {sorted(component)} with "
                        f"contacts {sorted(set(on) - {self.gate})}: no insulator between them"
                    )
                gate_side = component
        return sorted(gate_side)

    # --- state ----------------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            r: list(devsim.get_node_model_values(device=self.device, region=r, name="Potential"))
            for r in self.regions
            if (r, "Potential") in self.values
        }

    def load(self, snap: dict) -> None:
        for r, values in snap.items():
            devsim.set_node_values(device=self.device, region=r, name="Potential", values=values)

    def set_gate(self, v: float) -> None:
        """Gate bias ``v``: the parameter (PB) or the source (DD + circuit), and the gate-side
        silicon's quasi-Fermi level."""
        if self.circuit:
            devsim.circuit_alter(name=self.source[self.gate], value=float(v))
        else:
            devsim.set_parameter(
                device=self.device, name=contacts_mod.bias_name(self.gate), value=float(v)
            )
        for r in self.gate_regions:
            devsim.set_parameter(device=self.device, region=r, name=_QFL, value=float(v))
        self.v_gate = float(v)

    def _roundoff_resolve(self, bound: float, n_regions: int, *, failed_iterations: int):
        """Solve again, from the restored state, to the round-off iterate, and accept it if the
        kept state meets the update bound and condition (d); else None.

        DEVSIM's absolute error is the sum over regions of each equation's largest update, so
        ``absolute_error = bound * n_regions`` with a non-binding relative error makes DEVSIM
        stop, converged, once the potential updates are at round-off, and keep that state
        (a converged solve is not restored). The per-equation update and (d) are then checked
        on that state; a solve that stops elsewhere is rejected.
        """
        rel_tol, abs_tol = self.tol["rel_tol"], self.tol["abs_tol"]
        try:
            info = solve_dc(
                absolute_error=bound * n_regions,
                relative_error=_NON_BINDING_REL,
                maximum_iterations=self.tol["max_iterations"],
            )
        except ConvergenceError:
            return None
        last = _potential_errors(self.device, info["iterations"][-1])
        if not last or any(a > min(abs_tol, bound) for _, a in last.values()):
            return None
        psi = _max_abs_psi(self.device, last)
        # (d) every region above rel_tol has a node with |psi| small enough for the relative
        # error (|update| / (|psi| + 1e-10 V)) to exceed rel_tol at round-off updates
        violating = {r: ra for r, ra in last.items() if ra[0] > rel_tol}
        for r, (_, a) in violating.items():
            if min(psi[r]) > a / rel_tol:
                return None
        rel = max(r for r, _ in last.values())
        pot = max(a for _, a in last.values())
        worst = max(last, key=lambda r: last[r][0])
        self.roundoff_hits.append(
            RoundoffPoint(
                index=self.index,
                vg=self.v_gate,
                rel_error=rel,
                abs_error=pot,
                max_abs_psi=max(max(v) for v in psi.values()),
                roundoff_bound=bound,
                region=worst,
                psi_bound=last[worst][1] / rel_tol,
                min_abs_psi=min(psi[worst]),
            )
        )
        return PointInfo(True, failed_iterations + len(info["iterations"]), rel, pot)

    def solve(self) -> PointInfo:
        try:
            return _solve(self.device, **self.tol)
        except ConvergenceError as e:
            if self.roundoff:
                pre = _roundoff_history(
                    self.device, e.info, rel_tol=self.tol["rel_tol"], abs_tol=self.tol["abs_tol"]
                )
                if pre is not None:
                    point = self._roundoff_resolve(*pre, failed_iterations=len(e.info["iterations"]))
                    if point is not None:
                        return point
            if not self.fallback:
                raise
        # retry from where the double-precision Newton stopped, in extended precision
        for name in _PRECISION:
            devsim.set_parameter(name=name, value=True)
        try:
            info = _solve(self.device, **self.tol)
        finally:
            for name in _PRECISION:
                devsim.set_parameter(name=name, value=False)
        self.fallback_used = True
        return info

    # --- Poisson-Boltzmann (flat quasi-Fermi levels, potential only) ------------------------
    def enter_pb(self) -> None:
        device = self.device
        for c in self.contacts:
            devsim.set_parameter(device=device, name=contacts_mod.bias_name(c), value=0.0)
        for i, eqs in self.ieqs.items():
            for e in eqs:
                if e in _CONTINUITY:
                    devsim.delete_interface_equation(device=device, interface=i, name=e)
        for c, eqs in self.ceqs.items():
            for e in eqs:
                if e in _CONTINUITY:
                    devsim.delete_contact_equation(device=device, contact=c, name=e)
        for r in self.silicon:
            for e in self.eqs[r]:
                if e in _CONTINUITY:
                    devsim.delete_equation(device=device, region=r, name=e)
            devsim.set_parameter(device=device, region=r, name=_QFL, value=0.0)
            for name, expr in (
                (_N, f"n_i*exp((Potential - {_QFL})/V_t)"),
                (_P, f"n_i*exp(({_QFL} - Potential)/V_t)"),
                (_CHARGE, f"-ElectronCharge*kahan3({_P}, -{_N}, NetDoping)"),
            ):
                model_create.CreateNodeModel(device, r, name, expr)
                model_create.CreateNodeModelDerivative(device, r, name, expr, "Potential")
            cmd = dict(self.eqs[r]["PotentialEquation"])
            cmd["node_model"] = _CHARGE
            devsim.equation(**cmd)
        self.mode = "pb"
        self.set_gate(self.v_gate)

    def pb_to(self, target: float) -> PointInfo:
        """Ramp the gate to ``target`` (PB) and return the info of the solve there.

        On failure the device is left at the last converged bias and ConvergenceError raised.
        """
        step, halvings = MAX_STEP_V, 0
        good = self.snapshot()
        while True:
            v = self.v_gate
            nxt = target if abs(target - v) <= step else v + math.copysign(step, target - v)
            self.set_gate(nxt)
            try:
                info = self.solve()
            except ConvergenceError:
                self.load(good)
                self.set_gate(v)
                if halvings >= MAX_HALVINGS:
                    raise
                step, halvings = step / 2, halvings + 1
                continue
            good = self.snapshot()
            if nxt == target:
                return info

    def pb_gate_charge(self) -> float:
        """Gate electrode charge (C/cm^2 in 1D, C/cm in 2D) in the PB state."""
        q = self.dd.constants.q_c
        total = [devsim.get_contact_charge(device=self.device, contact=self.gate, equation="PotentialEquation")]
        for r in self.gate_regions:

            def values(name, r=r):
                return devsim.get_node_model_values(device=self.device, region=r, name=name)

            total.append(
                q
                * math.fsum(
                    v * (p - n + d)
                    for v, n, p, d in zip(values("NodeVolume"), values(_N), values(_P), values("NetDoping"))
                )
            )
        return math.fsum(total)

    # --- drift-diffusion with a source on every contact -------------------------------------
    def enter_dd_circuit(self) -> None:
        device = self.device
        for r in self.silicon:
            for cmd in self.eqs[r].values():
                devsim.equation(**cmd)
        for eqs in self.ieqs.values():
            for cmd in eqs.values():
                devsim.interface_equation(**cmd)
        self.circuit = True  # from here on restore() cleans up, also after a partial failure
        for c in self.contacts:
            try:
                devsim.circuit_element(
                    name=self.source[c], n1=self.node[c], n2="0", value=0.0, acreal=0.0
                )
            except devsim.error as e:
                raise ValueError(
                    f"cannot add the AC sweep's source {self.source[c]!r}: {e}. The likely "
                    "cause is a pre-existing global DEVSIM circuit (e.g. a standalone "
                    "circuit_element) present during an earlier solve; DEVSIM's circuit is "
                    "global and an AC sweep needs it to itself"
                ) from e
            self.added_sources += 1
        for c in self.contacts:
            devsim.set_parameter(device=device, name=contacts_mod.bias_name(c), value=0.0)
            for e, cmd in self.ceqs[c].items():
                cmd = dict(cmd)
                if e == "PotentialEquation":
                    # The contact's own model is Potential - <c>_bias + f (both contact kinds),
                    # with <c>_bias = 0 here: the circuit node takes the bias's place.
                    model = cmd["node_model"]
                    new = f"cv_{model}"
                    expr = f"{model} + {contacts_mod.bias_name(c)} - {self.node[c]}"
                    model_create.CreateContactNodeModel(device, c, new, expr)
                    model_create.CreateContactNodeModelDerivative(device, c, new, expr, "Potential")
                    model_create.CreateContactNodeModel(device, c, f"{new}:{self.node[c]}", "-1")
                    cmd["node_model"] = new
                cmd["circuit_node"] = self.node[c]
                devsim.contact_equation(**cmd)
        self.mode = "dd"
        self.set_gate(self.v_gate)

    def load_dd(self, snap: dict, v: float) -> None:
        """Start a DD point at gate bias ``v`` from its PB solution ``snap``."""
        self.set_gate(v)
        self.load(snap)
        for r in self.silicon:
            for var, model in (("Electrons", _N), ("Holes", _P)):
                devsim.set_node_values(device=self.device, region=r, name=var, init_from=model)

    def ac(self, freq_hz: float) -> dict[str, complex]:
        """AC current into the device at every contact for 1 V AC on the gate source only."""
        devsim.circuit_alter(name=self.source[self.gate], param="acreal", value=1.0)
        try:
            # One linear solve around the converged DC point (no Newton, no convergence info).
            devsim.solve(type="ac", frequency=float(freq_hz))
            out = {}
            for c in self.contacts:
                node = f"{self.source[c]}.I"
                current = complex(
                    devsim.get_circuit_node_value(node=node, solution="ssac_real"),
                    devsim.get_circuit_node_value(node=node, solution="ssac_imag"),
                )
                out[c] = -current  # the source current flows out of n1: the device draws -I
            return out
        finally:
            devsim.circuit_alter(name=self.source[self.gate], param="acreal", value=0.0)

    # --- restore --------------------------------------------------------------------------
    def restore(self) -> None:
        device = self.device
        for r in self.silicon:
            for cmd in self.eqs[r].values():
                devsim.equation(**cmd)
        for eqs in self.ieqs.values():
            for cmd in eqs.values():
                devsim.interface_equation(**cmd)
        for eqs in self.ceqs.values():
            for cmd in eqs.values():
                devsim.contact_equation(**cmd)
        if self.circuit:
            # Nothing may refer to the circuit nodes once the circuit is gone.
            for c, eqs in self.ceqs.items():
                model = f"cv_{eqs['PotentialEquation']['node_model']}"
                for name in (model, f"{model}:Potential", f"{model}:{self.node[c]}"):
                    model_create.CreateContactNodeModel(device, c, name, "0")
            if self.added_sources:
                # Our sources cannot be removed one by one: the (global) circuit goes. If none
                # was added, a pre-existing circuit is left alone.
                contacts_mod.delete_circuit()
            self.circuit = False
            self.added_sources = 0
        for r in self.silicon:
            models = devsim.get_node_model_list(device=device, region=r)
            for name in (_CHARGE, _P, _N):
                for m in (f"{name}:Potential", name):
                    if m in models:
                        devsim.delete_node_model(device=device, region=r, name=m)
        for (r, v), values in self.values.items():
            devsim.set_node_values(device=device, region=r, name=v, values=values)
        for c, value in self.biases.items():
            devsim.set_parameter(device=device, name=contacts_mod.bias_name(c), value=value)
        for name, value in self.precision.items():
            devsim.set_parameter(name=name, value=value)


def _check_global_circuit() -> None:
    """Raise if any device's contact is attached to a (global) DEVSIM circuit node."""
    for device in devsim.get_device_list():
        for c in devsim.get_contact_list(device=device):
            for e in devsim.get_contact_equation_list(device=device, contact=c):
                cmd = devsim.get_contact_equation_command(device=device, contact=c, name=e)
                if cmd.get("circuit_node"):
                    raise ValueError(
                        f"DEVSIM's circuit is global and in use (device {device!r}, contact "
                        f"{c!r}, node {cmd['circuit_node']!r}); an AC sweep needs it to itself"
                    )


# --------------------------------------------------------------------------------------------
# The sweep
# --------------------------------------------------------------------------------------------
def sweep_cv(
    device,
    gate: str,
    vg: Sequence[float],
    *,
    method: Literal["qs", "ac"],
    freq_hz: float | None = None,
    dv_qs: float = 1e-3,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-12,
    max_iterations: int = 50,
    extended_precision: bool = True,
    progress: Callable[[str, int, float], None] | None = None,
    precision_fallback: bool = False,
    roundoff_accept: bool = False,
) -> CVResult:
    """Cgg (and, for ``"ac"``, Cxg of every non-gate contact) of ``device`` over ``vg``.

    Args:
        device: a solved :class:`~gplugins.devsim.get_solver.DDComponent` (1D or 2D) whose
            contacts are not attached to a circuit (``circuit=False``).
        gate: the swept contact; every other contact is held at 0 V.
        vg: gate voltages (V), swept in this order.
        method: ``"qs"`` (quasi-static, equilibrium Poisson-Boltzmann, central difference) or
            ``"ac"`` (small-signal AC on the drift-diffusion DC point); see the module notes.
        freq_hz: the AC frequency (``"ac"`` only, required there).
        dv_qs: the central-difference half step (V) of ``"qs"``.
        rel_tol: DEVSIM's relative error, also checked on every equation of the last
            iteration.
        abs_tol: the largest accepted potential update (V) of the last iteration, checked on
            every ``PotentialEquation``.
        max_iterations: Newton iterations per solve.
        extended_precision: solve with DEVSIM's extended (128-bit) solver, model and equation
            precision (default), or in double precision (``False``, ``"qs"`` only). Restored
            afterwards.
        progress: optional ``progress(stage, index, vg)``, called after each bias point
            (``"qs"``; or ``"pb"`` then ``"ac"``), whether or not it converged.
        precision_fallback: ``"qs"`` with ``extended_precision=False`` only: retry a solve
            that fails in double precision in extended precision, at the same tolerances
            (see the module notes); ``CVResult.extended_fallback`` lists those points.
        roundoff_accept: ``"qs"`` only: accept a solve whose relative error plateaus at
            round-off near psi = 0 under conditions (a)-(d) of the module notes;
            ``CVResult.roundoff_accepted`` records each one.

    The result depends on the device's state at entry (DEVSIM keeps solver state across
    solves): sweep a freshly built device for bit-reproducible results; a second sweep
    of the same device agrees to round-off only.

    Returns:
        A :class:`CVResult`; ``unit`` is F/cm^2 (1D) or F/cm (2D).

    Raises:
        CVConvergenceError: if any bias point did not converge at these tolerances (all
            points are attempted first; the device is restored).
        ValueError: on invalid arguments or a device sweep_cv cannot drive.
    """
    if method not in ("qs", "ac"):
        raise ValueError(f"method must be 'qs' or 'ac', got {method!r}")
    vg = tuple(float(v) for v in vg)
    if not vg or not all(math.isfinite(v) for v in vg):
        raise ValueError(f"vg must be a non-empty sequence of finite voltages, got {vg!r}")
    if method == "ac":
        if freq_hz is None or not (freq_hz > 0 and math.isfinite(freq_hz)):
            raise ValueError(f"method 'ac' needs freq_hz > 0, got {freq_hz!r}")
        freq_hz = float(freq_hz)
    elif freq_hz is not None:
        raise ValueError("freq_hz is an AC parameter; method 'qs' takes none")
    if not (dv_qs > 0 and math.isfinite(dv_qs)):
        raise ValueError(f"dv_qs must be > 0, got {dv_qs!r}")
    if not (rel_tol > 0 and abs_tol > 0):
        raise ValueError("rel_tol and abs_tol must be > 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if precision_fallback and (method != "qs" or extended_precision):
        raise ValueError(
            "precision_fallback applies to a QS sweep in double precision "
            "(method='qs', extended_precision=False)"
        )
    if roundoff_accept and method != "qs":
        raise ValueError("roundoff_accept applies to a QS sweep (method='qs')")
    if method == "ac" and not extended_precision:
        raise ValueError(
            "method 'ac' needs extended precision: the small-signal AC solve loses accuracy in "
            "double precision (see the module notes)"
        )

    session = _Session(
        device,
        gate,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_iterations=max_iterations,
        precision_fallback=precision_fallback,
        roundoff_accept=roundoff_accept,
    )
    fallback_points: list[int] = []
    if method == "ac":
        _check_global_circuit()
    failed: dict[int, str] = {}
    cgg: dict[int, float] = {}
    points: dict[int, PointInfo] = {}
    cgx: dict[str, dict[int, float]] = {c: {} for c in session.others}
    try:
        for name in _PRECISION:
            devsim.set_parameter(name=name, value=bool(extended_precision))
        session.enter_pb()
        pb_states: dict[int, dict] = {}
        for i, v in enumerate(vg):
            session.fallback_used = False
            session.index = i
            try:
                if method == "qs":
                    lo = session.pb_to(v - dv_qs)
                    q_lo = session.pb_gate_charge()
                    hi = session.pb_to(v + dv_qs)
                    q_hi = session.pb_gate_charge()
                    cgg[i] = (q_hi - q_lo) / (2 * dv_qs)
                    points[i] = _merge(lo, hi)
                else:
                    session.pb_to(v)
                    pb_states[i] = session.snapshot()
            except ConvergenceError as e:
                failed[i] = f"Poisson-Boltzmann: {e}"
            if session.fallback_used:
                fallback_points.append(i)
            if progress is not None:
                progress("qs" if method == "qs" else "pb", i, v)
        if method == "ac":
            session.enter_dd_circuit()
            omega = 2 * math.pi * freq_hz
            for i, v in enumerate(vg):
                if i not in pb_states:
                    if progress is not None:
                        progress("ac", i, v)
                    continue
                session.load_dd(pb_states[i], v)
                try:
                    points[i] = session.solve()
                except ConvergenceError as e:
                    failed[i] = f"drift-diffusion from the Poisson-Boltzmann guess: {e}"
                else:
                    current = session.ac(freq_hz)
                    cgg[i] = current[gate].imag / omega
                    for c in session.others:
                        cgx[c][i] = -current[c].imag / omega
                if progress is not None:
                    progress("ac", i, v)
    finally:
        session.restore()
    if failed:
        order = sorted(failed)
        raise CVConvergenceError([vg[i] for i in order], [(i, vg[i], failed[i]) for i in order])
    n = len(vg)
    return CVResult(
        vg=vg,
        cgg=tuple(cgg[i] for i in range(n)),
        unit=session.unit,
        points=tuple(points[i] for i in range(n)),
        cgx=MappingProxyType(
            {c: tuple(v[i] for i in range(n)) for c, v in cgx.items()} if method == "ac" else {}
        ),
        method=method,
        freq_hz=freq_hz,
        dv_qs=dv_qs if method == "qs" else None,
        extended_fallback=tuple(fallback_points),
        roundoff_accepted=tuple(session.roundoff_hits),
    )
