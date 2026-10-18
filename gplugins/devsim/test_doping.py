# SPDX-License-Identifier: MIT
"""Tests of the analytic doping profiles and of the ``gplugins.process`` implant profiles."""

from __future__ import annotations

import numpy as np
import pytest

from gplugins.devsim.constants import CONSTANTS
from gplugins.process import diffusion, implant_tables

# --- gplugins.process.diffusion ------------------------------------------------------------

# phosphorus 100 keV: Rp = 0.1227 um, straggle = 0.0442 um (implant_tables polynomials)
_IMPLANT = dict(dopant="phosphorus", dose=1e13, E=100.0)
_Z = np.linspace(0.0, 0.6, 6001)  # um


def test_diffused_equals_gaussian_at_t0() -> None:
    """With no diffusion time the diffused profile is the as-implanted Gaussian."""
    diffused = diffusion.silicon_diffused_gaussian_profile(**_IMPLANT, t=0.0, T=1000.0, z=_Z)
    implanted = implant_tables.silicon_gaussian_profile(**_IMPLANT, z=_Z)
    np.testing.assert_allclose(diffused, implanted, rtol=1e-12, atol=0.0)


def test_diffused_peak_at_t0() -> None:
    """The prefactor alone: at z = Rp the exponent is 0, so only the prefactor is compared.

    The prefactor must be dose / (sqrt(2 pi) sigma) with sigma converted from um to cm.
    """
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    sigma_cm = implant_tables.straggle_in_silicon["phosphorus"](100.0) * 1e-4
    (peak,) = diffusion.silicon_diffused_gaussian_profile(
        **_IMPLANT, t=0.0, T=1000.0, z=np.array([rp])
    )
    assert peak == pytest.approx(1e13 / (np.sqrt(2 * np.pi) * sigma_cm), rel=1e-12)


def test_diffused_shape_at_t0() -> None:
    """The exponent alone: C(z)/C(Rp) must be exp(-(z - Rp)^2 / (2 sigma^2))."""
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    sigma = implant_tables.straggle_in_silicon["phosphorus"](100.0)
    z = rp + sigma * np.array([0.0, 1.0, 2.0])
    c = diffusion.silicon_diffused_gaussian_profile(**_IMPLANT, t=0.0, T=1000.0, z=z)
    np.testing.assert_allclose(c / c[0], np.exp(-np.array([0.0, 0.5, 2.0])), rtol=1e-12)


def test_diffused_dose_and_width_after_anneal() -> None:
    """After an anneal the dose is unchanged and the variance grows by 2 D t (um^2)."""
    t, temp = 600.0, 1000.0
    sigma = implant_tables.straggle_in_silicon["phosphorus"](100.0)
    rp = implant_tables.depth_in_silicon["phosphorus"](100.0)
    dt_um2 = diffusion.D("phosphorus", temp) * 1e8 * t
    s = np.sqrt(sigma**2 + 2 * dt_um2)
    z = np.linspace(rp - 12 * s, rp + 12 * s, 20001)  # the whole Gaussian, not just the wafer
    c = diffusion.silicon_diffused_gaussian_profile(**_IMPLANT, t=t, T=temp, z=z)
    dose = np.trapezoid(c, z * 1e-4)
    var = np.trapezoid(c * (z - rp) ** 2, z) / np.trapezoid(c, z)
    assert dose == pytest.approx(1e13, rel=1e-9)
    assert var == pytest.approx(sigma**2 + 2 * dt_um2, rel=1e-9)


# --- gplugins.devsim.constants -------------------------------------------------------------


def test_constants_derived_values() -> None:
    assert CONSTANTS.vt_v == pytest.approx(25.6926e-3, abs=1e-7)
    assert CONSTANTS.n_i_cm3 == pytest.approx(8.266e9, rel=1e-4)
