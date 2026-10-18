# SPDX-License-Identifier: MIT
"""Physical constants for the gplugins.devsim TCAD models.

This is the one table of TCAD constants in the plugin; extend it here, never make a second one.
None of DEVSIM's ``python_packages/simple_physics.py`` defaults (eps_0 = 8.85e-14,
eps_si = 11.1, q = 1.6e-19, n_i = 1e10, T = 300 K) is used.

The values and citations are copied from gf180-tcad's ``Constants``
(https://github.com/mithro/gf180-tcad, ``src/gf180tcad/tcad/process.py``, commit a7dff78), which
follows the constants table of the gf180-tcad Phase 2 plan. Each field has a provenance tag:
``literature`` (a publication) or ``spice-card`` (the GF180MCU public ngspice card,
``sm141064.ngspice`` of gf180mcuD, sha256 ``69a7ab877344a6d477fdd11440457f64e7ad88a624504a737a161da463de8bc8``).

==========================  ==============================  ===========================================
field                       value                           provenance
==========================  ==============================  ===========================================
``q_c``                     1.602176634e-19 C               ``literature``: exact SI (CODATA 2018)
``k_b_j_per_k``             1.380649e-23 J/K                ``literature``: exact SI (CODATA 2018)
``eps0_f_per_cm``           8.8541878128e-14 F/cm           ``literature``: CODATA 2018
``eps_si``                  11.7                            ``literature``: Sze & Ng, *Physics of
                                                            Semiconductor Devices*, 3rd ed., Appendix
                                                            (https://doi.org/10.1002/0470068329); equal
                                                            to the BSIM4 EPSRSUB default (the card sets
                                                            no EPSRSUB). Page number unverified.
``eps_ox``                  3.9                             ``spice-card``: EPSROX,
                                                            ``sm141064.ngspice:37126`` (nfet_06v0.0)
``t_k``                     298.15 K                        ``spice-card``: TNOM 25 °C,
                                                            ``sm141064.ngspice:37308``
``n_i300_cm3``              9.65e9 cm⁻³                     ``literature``: Altermatt et al., J. Appl.
                                                            Phys. 93, 1598 (2003),
                                                            https://doi.org/10.1063/1.1529297
``n_i_cm3`` (property)      8.266e9 cm⁻³ at 298.15 K        ``literature``: n_i300·(T/300)^2.54·
                                                            exp(-6726 K·(1/T - 1/300 K)), the Misiakos &
                                                            Tsamakis T-dependence, J. Appl. Phys. 74,
                                                            3293 (1993),
                                                            https://doi.org/10.1063/1.354551,
                                                            normalised to Altermatt's 300 K value
``mu_n_cm2_per_vs``         1417 cm²/(V·s)                  ``literature``: Masetti et al., IEEE TED 30,
``mu_p_cm2_per_vs``         470.5 cm²/(V·s)                 764 (1983), µ_max,
                                                            https://doi.org/10.1109/T-ED.1983.21207;
                                                            constant (no field or doping dependence)
``tau_n_s``, ``tau_p_s``    1e-5 s                          ``literature``: Sze & Ng ch. 1 (SRH
                                                            recombination, midgap trap n1 = p1 = n_i).
                                                            Page number unverified.
==========================  ==============================  ===========================================

``vt_v`` (k_B·T/q) is 25.6926 mV at 298.15 K.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    """Physical constants of the TCAD models (see the module docstring for the citations)."""

    q_c: float = 1.602176634e-19
    k_b_j_per_k: float = 1.380649e-23
    eps0_f_per_cm: float = 8.8541878128e-14
    eps_si: float = 11.7
    eps_ox: float = 3.9
    t_k: float = 298.15
    n_i300_cm3: float = 9.65e9
    mu_n_cm2_per_vs: float = 1417.0
    mu_p_cm2_per_vs: float = 470.5
    tau_n_s: float = 1e-5
    tau_p_s: float = 1e-5

    @property
    def vt_v(self) -> float:
        """Thermal voltage k_B·T/q (V)."""
        return self.k_b_j_per_k * self.t_k / self.q_c

    @property
    def n_i_cm3(self) -> float:
        """Intrinsic density at ``t_k`` (cm⁻³), Misiakos & Tsamakis scaling of ``n_i300_cm3``."""
        t = self.t_k
        return (
            self.n_i300_cm3
            * (t / 300.0) ** 2.54
            * math.exp(-6726.0 * (1 / t - 1 / 300.0))
        )


# The constants used by every TCAD calculation of the plugin.
CONSTANTS = Constants()
