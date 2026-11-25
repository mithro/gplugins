# SPDX-License-Identifier: MIT
"""Charon interpreter decks for a :class:`~gplugins.devsim.device_description.Stack1D`.

Every constant comes from the stack's :class:`~gplugins.devsim.constants.Constants` and is
written explicitly, never left to a Charon default (Charon's silicon defaults: permittivity 11.8,
band gap 1.08 eV; Charon User Manual v2.2, section 6): the relative permittivities, the lattice
temperature, the intrinsic density (``constant intrinsic concentration``), the constant
mobilities and the SRH recombination. Boltzmann statistics are Charon's default; the deck never
enables Fermi-Dirac, band-gap narrowing or incomplete ionisation.

**SRH lifetimes.** The deck writes :data:`SRH_LIFETIME_S` (1e-15 s), not the description's
lifetimes. The quasi-static C-V solves the zero-current steady state, which is each region's
equilibrium and does not depend on the lifetimes; with the description's lifetimes Charon's
double-precision drift-diffusion Newton solve does not pass the onset of inversion, and with
1e-14 s or shorter it does. The C of the two lifetimes differs by 1.1e-9 relative, the solver's
noise (gf180-tcad ``docs/limitations/charon.md``, experiment B2). The value is recorded with the
result.

**Doping.** A layer's profile is written as a Charon ``uniform doping`` block only when it is a
:class:`~gplugins.devsim.doping.Constant` over the whole layer; any other profile raises
:class:`NotComparable` (Charon's analytic forms are not node-exact copies of the plugin's, and a
doping file is interpolated). A uniform block must be named ``Function1``, ``Function2``, ...:
Charon ignores any other name silently.

**Linear solver.** Amesos2 KLU2 (direct), set by ``Direct input`` lines; the interpreter offers
only AztecOO and Belos, and Belos with ILUT does not converge on this problem.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gplugins.devsim import doping
from gplugins.devsim.backends.charon.exodus import side_set_name

#: The SRH lifetime (s) written for both carriers (see the module notes).
SRH_LIFETIME_S = 1e-15

#: The electron affinity (eV) of every silicon block. The C-V depends only on affinity
#: differences between semiconductor regions, and every region here is silicon.
SILICON_AFFINITY_EV = 4.05

_STRATIMIKOS = (
    "Charon->Solution Control->NOX->Direction->Newton->Stratimikos Linear Solver->Stratimikos"
)
KLU2_LINES = (
    f'Direct input "{_STRATIMIKOS},Linear Solver Type,string,Amesos2"',
    f'Direct input "{_STRATIMIKOS},Preconditioner Type,string,None"',
    f'Direct input "{_STRATIMIKOS}->Linear Solver Types->Amesos2,Solver Type,string,KLU2"',
)


class NotComparable(ValueError):
    """The description holds a model or profile Charon cannot express identically."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Tolerances:
    """Newton WRMS tolerances, iterations and the linear-solver tolerance of the deck."""

    rel: float = 1e-10
    abs: float = 1e-12
    max_iterations: int = 50
    linear: float = 1e-12


@dataclass(frozen=True)
class Sweep:
    """A LOCA sweep of the gate (the bias range is the gate BC's ``swept from A to B``).

    ``step`` (V) is the fixed step, or with ``min_step`` the largest step of an adaptive one
    (a ramp to a start bias, whose intermediate states are not used); ``hit_bound`` makes LOCA
    end exactly on the range's end. LOCA takes the direction from the range.
    """

    step: float
    max_steps: int
    min_step: float | None = None
    hit_bound: bool = False
    predictor: str = "secant"

    def lines(self) -> list[str]:
        h = abs(self.step)
        lines = ["start sweep options", f"  initial step size = {h!r}",
                 f"  minimum step size = {abs(self.min_step or h)!r}",
                 f"  maximum step size = {h!r}",
                 f"  maximum number of steps = {self.max_steps}",
                 f"  hit continuation bound = {'true' if self.hit_bound else 'false'}"]
        if self.min_step is None:
            lines.append("  step size aggressiveness = 0.0")
        return [*lines, f"  predictor method is {self.predictor}", "end sweep options"]


def _uniform(layer) -> tuple[float, str]:
    prof = layer.doping
    if not isinstance(prof, doping.Constant) or prof.ymin is not None or prof.ymax is not None:
        raise NotComparable(
            f"doping profile of layer {layer.name!r} ({type(prof).__name__}) has no exact "
            "Charon form (only a Constant over the whole layer)"
        )
    if not prof.conc > 0:
        raise NotComparable(f"layer {layer.name!r}: doping {prof.conc!r} is not positive")
    return prof.conc, layer.dopant


def material_block(name: str, layer, constants) -> list[str]:
    """The material block of ``layer`` (silicon or oxide)."""
    c = constants
    if layer.material == "oxide":
        return [f"start Material Block {name}", "  material name is SiO2",
                f"  relative permittivity = {c.eps_ox!r}", f"end Material Block {name}"]  # fmt: skip
    conc, dopant = _uniform(layer)
    return [
        f"start Material Block {name}",
        "  material name is Silicon",
        f"  relative permittivity = {c.eps_si!r}",
        f"  electron affinity = {SILICON_AFFINITY_EV!r}",
        f"  constant intrinsic concentration = {c.n_i_cm3!r}",
        f"  fixed mobility for electrons is {c.mu_n_cm2_per_vs!r} holes is {c.mu_p_cm2_per_vs!r}",
        "  start Carrier Lifetime Block",
        f"    electron lifetime is constant = {SRH_LIFETIME_S!r}",
        f"    hole lifetime is constant = {SRH_LIFETIME_S!r}",
        "  end Carrier Lifetime Block",
        "  start uniform doping named Function1",
        f"    concentration = {conc!r}",
        f"    type is {dopant}",
        "  end uniform doping named Function1",
        f"end Material Block {name}",
    ]


def write_deck(
    stack,
    *,
    mesh_file: str,
    out_file: str,
    gate: str,
    gate_bc: str,
    problem: str,
    init: str,
    sweep: Sweep | None = None,
    tolerances: Tolerances = Tolerances(),
    import_index: int | None = None,
) -> str:
    """The interpreter deck of ``stack``: ``gate`` held by ``gate_bc`` (``fixed at V`` or
    ``swept from A to B``), every other contact at 0 V; ``problem`` ``steady state`` or
    ``parameter sweep``; ``init`` ``equilibrium`` (Charon's estimate) or ``exodus`` (the
    imported state)."""
    if problem not in ("steady state", "parameter sweep"):
        raise ValueError(f"problem must be 'steady state' or 'parameter sweep', got {problem!r}")
    if init not in ("equilibrium", "exodus"):
        raise ValueError(f"init must be 'equilibrium' or 'exodus', got {init!r}")
    if (problem == "parameter sweep") != (sweep is not None):
        raise ValueError("a parameter sweep needs a Sweep, a steady state none")
    c = stack.constants
    regions = [layer.name for layer in stack.layers]
    ends = stack.contact_ends()
    if gate not in ends:
        raise ValueError(f"gate {gate!r} is not a contact of the stack: {sorted(ends)}")
    lines = [
        f"Import State File {mesh_file}" + (f" at Index {import_index}" if import_index else ""),
        "tpetra is on",
        f"lattice temperature = {c.t_k!r}",
        "start output parameters",
        f"  Output State File {out_file}",
        "  Unscale Variables is on",
        "end output parameters",
    ]
    for layer in stack.layers:
        disc = "drift diffusion cvfem" if layer.material == "silicon" else "laplace cvfem"
        lines += [f"start Physics Block {layer.name}", f"  geometry block is {layer.name}",
                  f"  standard discretization type is {disc}",
                  f"  material model is {layer.name}Material"]  # fmt: skip
        if layer.material == "silicon":
            lines.append("  SRH Recombination is on")
        lines.append(f"end Physics Block {layer.name}")
    for layer in stack.layers:
        lines += material_block(f"{layer.name}Material", layer, c)
    for layer in stack.layers:
        dofs = (("ELECTRIC_POTENTIAL", "ELECTRON_DENSITY", "HOLE_DENSITY")
                if layer.material == "silicon" else ("ELECTRIC_POTENTIAL",))  # fmt: skip
        for dof in dofs:
            if init == "exodus":
                how = "exodus file"
            elif layer.material == "oxide":
                how = "uninitialized"
            else:
                how = "equilibrium potential" if dof == "ELECTRIC_POTENTIAL" else (
                    "equilibrium density")  # fmt: skip
            lines.append(f"initial conditions for {dof} in {layer.name} is {how}")
    index = {0: 0, len(stack.layers): len(stack.layers) - 1}
    for contact, end in sorted(ends.items()):
        region = regions[index[end]]
        ss = side_set_name(contact, regions)
        bc = gate_bc if contact == gate else "fixed at 0.0"
        lines.append(f"BC is ohmic for {ss} on {region} {bc}")
    if sweep is not None:
        lines += sweep.lines()
    lines += list(KLU2_LINES)
    t = tolerances
    lines += [
        "start solver block",
        "  start tpetra block",
        f"    problem type is {problem}",
        "    linear solver is belos",
        f"    linear solver tolerance = {t.linear!r}",
        f"    maximum number nonlinear iterations = {t.max_iterations}",
        "    verbosity level is high",
        "    start nonlinear solver wrms block",
        f"      relative tolerance = {t.rel!r}",
        f"      absolute tolerance = {t.abs!r}",
        "    end nonlinear solver wrms block",
        "  end tpetra block",
        "end solver block",
    ]
    return "\n".join(lines) + "\n"


def contact_regions(stack) -> dict[str, str]:
    """``{contact: region it sits on}`` of a 1D stack."""
    regions = [layer.name for layer in stack.layers]
    index = {0: 0, len(stack.layers): len(stack.layers) - 1}
    return {k: regions[index[e]] for k, e in stack.contact_ends().items()}


def check_comparable(stack) -> None:
    """Raise :class:`NotComparable` for anything of ``stack`` the deck cannot write exactly."""
    for layer in stack.layers:
        if layer.material == "silicon":
            _uniform(layer)
    for contact in stack.contacts:
        if contact.kind != "ohmic":
            raise NotComparable(f"contact {contact.name!r}: only ohmic contacts are written")


__all__: Sequence[str] = (
    "KLU2_LINES", "NotComparable", "SRH_LIFETIME_S", "Sweep", "Tolerances", "check_comparable",
    "contact_regions", "material_block", "write_deck",
)  # fmt: skip
