# SPDX-License-Identifier: MIT
"""Material parameters and bulk equations for gplugins.devsim, from explicit constants.

The constants are :class:`gplugins.devsim.constants.Constants`, re-exported here as
:class:`Constants` (there is one constants table; extend it there). :func:`set_silicon` and
:func:`set_oxide` take every value as a keyword argument and have no defaults:
:func:`silicon_parameters` / :func:`oxide_parameters` produce those arguments from a
``Constants``. DEVSIM's ``simple_physics.SetSiliconParameters`` / ``SetOxideParameters``, with
their hard-coded eps_0 = 8.85e-14 F/cm, eps_si = 11.1, q = 1.6e-19 C, n_i = 1e10 cm^-3,
mu = 400 / 200 cm^2/(V s), are never called.

The bulk equations are DEVSIM's ``simple_physics`` models (Boltzmann statistics, constant
mobility, SRH recombination), which read only the region parameters set here. DEVSIM works
in cm.

Region classes: a region is ``"silicon"`` (Poisson + drift-diffusion; a poly-silicon gate is
silicon too), ``"oxide"`` (Poisson only) or ``None`` (no equations, e.g. a metal electrode).
"""

from __future__ import annotations

from typing import Literal

import devsim
from devsim.python_packages import model_create, simple_physics

from gplugins.devsim.constants import CONSTANTS, Constants

__all__ = [
    "CONSTANTS",
    "OXIDE_MATERIALS",
    "SILICON_MATERIALS",
    "SIMPLE_PHYSICS_V080",
    "Constants",
    "RegionClass",
    "create_oxide_potential_only",
    "create_silicon_drift_diffusion",
    "create_silicon_potential_only",
    "material_class",
    "oxide_parameters",
    "set_oxide",
    "set_silicon",
    "silicon_parameters",
]

RegionClass = Literal["silicon", "oxide"]

#: Material names (as in gdsfactory layer stacks and :class:`~gplugins.devsim.stack1d.Layer1D`)
#: solved as silicon and as oxide.
SILICON_MATERIALS = frozenset({"silicon", "si", "Si", "poly", "polysilicon", "poly_si"})
OXIDE_MATERIALS = frozenset({"oxide", "sio2", "SiO2", "Oxide", "gate_oxide"})

#: DEVSIM ``simple_physics.py``'s values (eps_0, eps_si, q, k, n_i at 300 K, mu_n, mu_p, tau),
#: as a :class:`Constants`. **Only** for the deprecated gplugins v0.8.0 path of
#: :class:`~gplugins.devsim.get_solver.DDComponent` (``contact_info`` without ``constants``),
#: which must reproduce v0.8.0 bit for bit; never for TCAD results. With ``t_k`` = 300 K the
#: n_i scaling is exactly 1, so ``n_i_cm3`` is exactly 1e10.
SIMPLE_PHYSICS_V080 = Constants(
    q_c=1.6e-19,
    k_b_j_per_k=1.3806503e-23,
    eps0_f_per_cm=8.85e-14,
    eps_si=11.1,
    eps_ox=3.9,
    t_k=300.0,
    n_i300_cm3=1.0e10,
    mu_n_cm2_per_vs=400.0,
    mu_p_cm2_per_vs=200.0,
    tau_n_s=1e-5,
    tau_p_s=1e-5,
)


def material_class(material: str | None) -> RegionClass | None:
    """``"silicon"``, ``"oxide"`` or ``None`` for a material name."""
    if material in SILICON_MATERIALS:
        return "silicon"
    if material in OXIDE_MATERIALS:
        return "oxide"
    return None


def silicon_parameters(constants: Constants) -> dict[str, float]:
    """The :func:`set_silicon` keyword arguments for ``constants``."""
    return {
        "T_K": constants.t_k,
        "eps_r": constants.eps_si,
        "n_i_cm3": constants.n_i_cm3,
        "mu_n": constants.mu_n_cm2_per_vs,
        "mu_p": constants.mu_p_cm2_per_vs,
        "tau_n": constants.tau_n_s,
        "tau_p": constants.tau_p_s,
        "eps0_F_cm": constants.eps0_f_per_cm,
        "q_C": constants.q_c,
        "kB_J_K": constants.k_b_j_per_k,
    }


def oxide_parameters(constants: Constants) -> dict[str, float]:
    """The :func:`set_oxide` keyword arguments for ``constants``."""
    return {
        "eps_r": constants.eps_ox,
        "eps0_F_cm": constants.eps0_f_per_cm,
        "q_C": constants.q_c,
    }


def set_silicon(
    device: str,
    region: str,
    *,
    T_K: float,
    eps_r: float,
    n_i_cm3: float,
    mu_n: float,
    mu_p: float,
    tau_n: float,
    tau_p: float,
    eps0_F_cm: float,
    q_C: float,
    kB_J_K: float,
) -> None:
    """Set the silicon region parameters the ``simple_physics`` models read.

    The parameter names are DEVSIM's: ``Permittivity`` (F/cm), ``ElectronCharge`` (C), ``n_i``
    (cm^-3), ``T`` (K), ``kT`` (J), ``V_t`` (V), ``mu_n``/``mu_p`` (cm^2/(V s)), the SRH
    ``taun``/``taup`` (s) and ``n1`` = ``p1`` = ``n_i`` (midgap trap). They are computed in the
    same order as ``simple_physics.SetSiliconParameters``, so the same inputs give
    bit-identical parameters.
    """
    for name, value in (
        ("Permittivity", eps_r * eps0_F_cm),
        ("ElectronCharge", q_C),
        ("n_i", n_i_cm3),
        ("T", T_K),
        ("kT", kB_J_K * T_K),
        ("V_t", kB_J_K * T_K / q_C),
        ("mu_n", mu_n),
        ("mu_p", mu_p),
        ("n1", n_i_cm3),
        ("p1", n_i_cm3),
        ("taun", tau_n),
        ("taup", tau_p),
    ):
        devsim.set_parameter(device=device, region=region, name=name, value=value)


def set_oxide(device: str, region: str, *, eps_r: float, eps0_F_cm: float, q_C: float) -> None:
    """Set the oxide region parameters: ``Permittivity`` (F/cm) and ``ElectronCharge`` (C)."""
    devsim.set_parameter(
        device=device, region=region, name="Permittivity", value=eps_r * eps0_F_cm
    )
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=q_C)


def create_silicon_potential_only(device: str, region: str) -> None:
    """Poisson's equation with equilibrium (intrinsic-referenced Boltzmann) carriers.

    Needs ``NetDoping`` and the :func:`set_silicon` parameters.
    """
    model_create.CreateSolution(device, region, "Potential")
    simple_physics.CreateSiliconPotentialOnly(device, region)


def create_oxide_potential_only(device: str, region: str) -> None:
    """Poisson's equation without charge (Laplace) in an oxide region."""
    model_create.CreateSolution(device, region, "Potential")
    simple_physics.CreateOxidePotentialOnly(device, region, "log_damp")


def create_silicon_drift_diffusion(device: str, region: str) -> None:
    """Electron and hole continuity (constant mobility, SRH), from the equilibrium carriers.

    Call after a converged :func:`create_silicon_potential_only` solve: the carriers start
    from ``IntrinsicElectrons``/``IntrinsicHoles`` of that solution.
    """
    model_create.CreateSolution(device, region, "Electrons")
    model_create.CreateSolution(device, region, "Holes")
    devsim.set_node_values(
        device=device, region=region, name="Electrons", init_from="IntrinsicElectrons"
    )
    devsim.set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    simple_physics.CreateSiliconDriftDiffusion(device, region)
