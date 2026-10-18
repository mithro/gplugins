"""Doping for gplugins.devsim: analytic doping profiles and their DEVSIM node models.

A :class:`DopingProfile` is a callable ``profile(x_um, y_um) -> cm^-3``. ``x_um`` is the lateral
position and ``y_um`` the depth below the semiconductor surface, both in um, with the depth
positive into the semiconductor. The profiles are:

- :func:`constant`: a uniform concentration, optionally limited to a depth window;
- :func:`gaussian`: an implant ``peak * exp(-(y - depth)^2 / (2 straggle^2))``, and
  :func:`gaussian_from_dose`, the same from its dose. A dose-specified implant is renormalised
  by default so that the dose in the material (y >= 0) is the stated dose
  (``truncation="renormalise"``; ``"clip"`` keeps the whole-Gaussian peak);
- :func:`erfc`: a diffused profile ``surface * erfc(y / length)``, with ``length = 2 sqrt(D t)``
  (the brief's ``erfc(surface, xj)`` form is :func:`erfc_from_junction`, which takes the
  junction depth ``xj`` with a ``background`` doping and sets
  ``length = xj / erfcinv(background / surface)``);
- :func:`lateral`: a dimensionless mask-edge factor, the lateral spread of an implant through a
  mask opening ``[xmin, xmax]``, to multiply a vertical profile by.

Profiles add (``p1 + p2``) and multiply (``gaussian(...) * lateral(...)``). Each one also writes
itself as a DEVSIM node-model expression, so :func:`assign_doping` creates the ``Donors``,
``Acceptors`` and ``NetDoping`` node models of a region as expressions of the node positions
rather than as fixed node values. :func:`profile_from_process` turns a gdsfactory
``technology.processes`` step (``ImplantGaussian``, ``ImplantPhysical`` or ``DopingConstant``)
into a ``(kind, profile)`` pair.

The older layer-based helpers (:class:`DopingLayerLevel`, :func:`get_doping_info_generic`,
:func:`step`) used by ``get_simulation`` are kept unchanged at the end of the module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from gdsfactory.gpdk import LAYER
from gdsfactory.typings import Layer
from pydantic import BaseModel, ConfigDict
from scipy import special

UM_TO_CM = 1e-4

DopantKind = Literal["donor", "acceptor"]


def _num(value: float) -> str:
    """A float as a parenthesised DEVSIM literal (shortest round-trip repr)."""
    return f"({float(value)!r})"


class DopingProfile:
    """A doping profile ``(x_um, y_um) -> cm^-3``; ``y_um`` is the depth below the surface.

    Subclasses implement ``__call__`` (numpy, for arrays or scalars) and :meth:`expression`
    (the same function as a DEVSIM expression). Profiles combine with ``+`` and ``*``.
    """

    def __call__(self, x_um, y_um) -> np.ndarray:
        raise NotImplementedError

    def expression(self, x_um: str, y_um: str) -> str:
        """The profile as a DEVSIM expression of the expressions ``x_um`` and ``y_um``."""
        raise NotImplementedError

    def __add__(self, other: DopingProfile) -> DopingProfile:
        return _Sum((self, other))

    def __mul__(self, other: DopingProfile) -> DopingProfile:
        return _Product((self, other))


@dataclass(frozen=True)
class _Sum(DopingProfile):
    terms: tuple[DopingProfile, ...]

    def __call__(self, x_um, y_um) -> np.ndarray:
        return sum(t(x_um, y_um) for t in self.terms)

    def expression(self, x_um: str, y_um: str) -> str:
        return "(" + " + ".join(t.expression(x_um, y_um) for t in self.terms) + ")"


@dataclass(frozen=True)
class _Product(DopingProfile):
    factors: tuple[DopingProfile, ...]

    def __call__(self, x_um, y_um) -> np.ndarray:
        return math.prod(f(x_um, y_um) for f in self.factors)

    def expression(self, x_um: str, y_um: str) -> str:
        return "(" + " * ".join(f.expression(x_um, y_um) for f in self.factors) + ")"


@dataclass(frozen=True)
class Constant(DopingProfile):
    """``conc`` (cm^-3) for ``ymin <= y <= ymax`` (um, ``None`` = unbounded), else 0."""

    conc: float
    ymin: float | None = None
    ymax: float | None = None

    def __call__(self, x_um, y_um) -> np.ndarray:
        y = np.asarray(y_um, dtype=float)
        inside = np.ones_like(y, dtype=bool)
        if self.ymin is not None:
            inside &= y >= self.ymin
        if self.ymax is not None:
            inside &= y <= self.ymax
        return np.where(inside, self.conc, 0.0) + 0.0 * np.asarray(x_um, dtype=float)

    def expression(self, x_um: str, y_um: str) -> str:
        out = _num(self.conc)
        # DEVSIM step(v) is 1 for v >= 0, matching the closed window above
        if self.ymin is not None:
            out += f"*step({y_um} - {_num(self.ymin)})"
        if self.ymax is not None:
            out += f"*step({_num(self.ymax)} - {y_um})"
        return f"({out})"


@dataclass(frozen=True)
class Gaussian(DopingProfile):
    """``peak * exp(-(y - depth)^2 / (2 straggle^2))``: peak in cm^-3, depth and straggle in um."""

    peak: float
    depth: float
    straggle: float

    def __post_init__(self) -> None:
        if not self.straggle > 0:
            raise ValueError(f"straggle must be > 0 um, got {self.straggle!r}")

    @property
    def dose(self) -> float:
        """The dose (cm^-2) of the whole Gaussian, ``peak * sqrt(2 pi) * straggle``."""
        return self.peak * math.sqrt(2 * math.pi) * self.straggle * UM_TO_CM

    @property
    def dose_in_silicon(self) -> float:
        """The dose (cm^-2) at y >= 0: :attr:`dose` times ``erfc(-depth/(sqrt(2) straggle))/2``."""
        return self.dose * _in_silicon_fraction(self.depth, self.straggle)

    def __call__(self, x_um, y_um) -> np.ndarray:
        u = (np.asarray(y_um, dtype=float) - self.depth) / self.straggle
        return self.peak * np.exp(-0.5 * u * u) + 0.0 * np.asarray(x_um, dtype=float)

    def expression(self, x_um: str, y_um: str) -> str:
        # pow() and an explicit -0.5 factor: DEVSIM parses "-(a)^2" as "(-a)^2"
        u = f"(({y_um} - {_num(self.depth)})/{_num(self.straggle)})"
        return f"({_num(self.peak)}*exp((-0.5)*pow({u}, 2)))"


@dataclass(frozen=True)
class Erfc(DopingProfile):
    """``surface * erfc(y / length)``: surface in cm^-3, length (= 2 sqrt(D t)) in um."""

    surface: float
    length: float

    def __post_init__(self) -> None:
        if not self.length > 0:
            raise ValueError(f"length must be > 0 um, got {self.length!r}")

    @property
    def dose(self) -> float:
        """The dose (cm^-2) for y >= 0, ``surface * length / sqrt(pi)``."""
        return self.surface * self.length * UM_TO_CM / math.sqrt(math.pi)

    def __call__(self, x_um, y_um) -> np.ndarray:
        y = np.asarray(y_um, dtype=float)
        return self.surface * special.erfc(y / self.length) + 0.0 * np.asarray(
            x_um, dtype=float
        )

    def expression(self, x_um: str, y_um: str) -> str:
        return f"({_num(self.surface)}*erfc({y_um}/{_num(self.length)}))"


@dataclass(frozen=True)
class Lateral(DopingProfile):
    """The dimensionless lateral factor of an implant through the mask opening [xmin, xmax].

    For ``straggle > 0`` (um) this is the Gaussian lateral spread
    ``(erf((x - xmin)/(sqrt(2) s)) - erf((x - xmax)/(sqrt(2) s))) / 2``; for ``straggle == 0``
    the sharp window ``xmin <= x <= xmax``. ``None`` leaves that side of the opening unbounded.
    """

    straggle: float
    xmin: float | None = None
    xmax: float | None = None

    def __post_init__(self) -> None:
        if not self.straggle >= 0:
            raise ValueError(f"straggle must be >= 0 um, got {self.straggle!r}")

    def __call__(self, x_um, y_um) -> np.ndarray:
        x = np.asarray(x_um, dtype=float) + 0.0 * np.asarray(y_um, dtype=float)
        if self.straggle == 0:
            inside = np.ones_like(x, dtype=bool)
            if self.xmin is not None:
                inside &= x >= self.xmin
            if self.xmax is not None:
                inside &= x <= self.xmax
            return np.where(inside, 1.0, 0.0)
        w = math.sqrt(2) * self.straggle
        lo = 1.0 if self.xmin is None else special.erf((x - self.xmin) / w)
        hi = -1.0 if self.xmax is None else special.erf((x - self.xmax) / w)
        return 0.5 * (lo - hi)

    def expression(self, x_um: str, y_um: str) -> str:
        if self.straggle == 0:
            out = "1"
            if self.xmin is not None:
                out += f"*step({x_um} - {_num(self.xmin)})"
            if self.xmax is not None:
                out += f"*step({_num(self.xmax)} - {x_um})"
            return f"({out})"
        w = _num(math.sqrt(2) * self.straggle)
        lo = "1" if self.xmin is None else f"erf(({x_um} - {_num(self.xmin)})/{w})"
        hi = "(-1)" if self.xmax is None else f"erf(({x_um} - {_num(self.xmax)})/{w})"
        return f"(0.5*({lo} - {hi}))"


def constant(
    conc: float, ymin: float | None = None, ymax: float | None = None
) -> Constant:
    """Uniform ``conc`` (cm^-3), optionally only for depths ``ymin <= y <= ymax`` (um)."""
    return Constant(conc=conc, ymin=ymin, ymax=ymax)


def gaussian(peak: float, depth: float, straggle: float) -> Gaussian:
    """Gaussian implant: ``peak`` (cm^-3) at ``depth`` (um, the projected range), ``straggle`` (um)."""
    return Gaussian(peak=peak, depth=depth, straggle=straggle)


def _in_silicon_fraction(depth: float, straggle: float) -> float:
    """The fraction of a whole Gaussian at y >= 0, ``erfc(-depth / (sqrt(2) straggle)) / 2``."""
    return 0.5 * math.erfc(-depth / (math.sqrt(2) * straggle))


Truncation = Literal["renormalise", "clip"]


def gaussian_from_dose(
    dose: float,
    depth: float,
    straggle: float,
    *,
    truncation: Truncation = "renormalise",
) -> Gaussian:
    """Gaussian implant of ``dose`` (cm^-2) at ``depth`` (um) with ``straggle`` (um).

    The doped material occupies y >= 0, so a whole Gaussian puts the fraction
    ``f = erfc(-depth / (sqrt(2) straggle)) / 2`` of its dose in it; the rest lies above the
    surface. ``truncation`` says what happens to that rest:

    - ``"renormalise"`` (default): the dose in the material is ``dose``,
      ``peak = dose / (sqrt(2 pi) straggle f)``. This is how a stated implant dose is read
      here: the ions are in the wafer.
    - ``"clip"``: the whole Gaussian has ``dose``, ``peak = dose / (sqrt(2 pi) straggle)``, and
      only ``dose * f`` is in the material (the part above the surface is discarded).

    For a deep implant (depth >> straggle) the two agree; for the shallow implants of
    gdsfactory's gpdk process (B 15 keV, P 50 keV) they differ by about 1 %.
    """
    if truncation == "renormalise":
        fraction = _in_silicon_fraction(depth, straggle)
    elif truncation == "clip":
        fraction = 1.0
    else:
        raise ValueError(
            f"truncation must be 'renormalise' or 'clip', got {truncation!r}"
        )
    return Gaussian(
        peak=dose / (math.sqrt(2 * math.pi) * straggle * fraction * UM_TO_CM),
        depth=depth,
        straggle=straggle,
    )


def erfc(surface: float, length: float) -> Erfc:
    """Diffused profile: surface times the complementary error function of y / length (um)."""
    return Erfc(surface=surface, length=length)


def erfc_from_junction(surface: float, xj: float, background: float) -> Erfc:
    """Diffused erfc profile whose metallurgical junction with ``background`` is at ``xj``.

    ``surface * erfc(xj / L) = background`` gives ``L = xj / erfcinv(background / surface)``.
    ``surface`` and ``background`` are in cm^-3 with ``0 < background < surface``; ``xj`` in um.
    """
    if not 0 < background < surface:
        raise ValueError(
            f"background must satisfy 0 < background < surface, got {background!r} and "
            f"surface {surface!r}"
        )
    return Erfc(
        surface=surface, length=xj / float(special.erfcinv(background / surface))
    )


def lateral(
    straggle: float, xmin: float | None = None, xmax: float | None = None
) -> Lateral:
    """Lateral mask-edge factor for the opening [xmin, xmax] (um), lateral ``straggle`` (um)."""
    return Lateral(straggle=straggle, xmin=xmin, xmax=xmax)


# --- gdsfactory technology.processes ---------------------------------------------------------

# ion tag (lower case) -> (kind, gplugins.process.implant_tables name or None)
_IONS: dict[str, tuple[DopantKind, str | None]] = {
    "b": ("acceptor", "boron"),
    "boron": ("acceptor", "boron"),
    "bf2": ("acceptor", None),
    "in": ("acceptor", None),
    "indium": ("acceptor", None),
    "p": ("donor", "phosphorus"),
    "phosphorus": ("donor", "phosphorus"),
    "as": ("donor", "arsenic"),
    "arsenic": ("donor", "arsenic"),
    "sb": ("donor", "antimony"),
    "antimony": ("donor", "antimony"),
}


def dopant_kind(ion: str) -> DopantKind:
    """``"donor"`` or ``"acceptor"`` for a gdsfactory ion tag such as ``"B"`` or ``"arsenic"``."""
    try:
        return _IONS[ion.lower()][0]
    except KeyError:
        raise ValueError(f"unknown ion {ion!r}; known: {sorted(_IONS)}") from None


def profile_from_process(
    step,
    *,
    mask_um: tuple[float | None, float | None] | None = None,
    surface_z_um: float | None = None,
    truncation: Truncation = "renormalise",
) -> tuple[DopantKind, DopingProfile]:
    """``(kind, profile)`` of a gdsfactory ``technology.processes`` doping step.

    - ``ImplantGaussian``: :func:`gaussian` of ``peak_conc``, ``range`` and
      ``vertical_straggle`` (um);
    - ``ImplantPhysical``: :func:`gaussian_from_dose` of ``dose``, with the range and straggle
      of ``ion`` at ``energy`` (keV) from ``gplugins.process.implant_tables``, and
      ``truncation`` as in :func:`gaussian_from_dose` (by default the dose in the material is
      the stated dose). A peak-specified ``ImplantGaussian`` keeps its peak;
    - ``DopingConstant``: :func:`constant` of ``peak_conc``. gdsfactory's frame is z-up and
      ``zmin``/``zmax`` are absolute z coordinates (um), with ``zmax=None`` meaning the top
      of the doped material. ``surface_z_um`` (required) is that top, and the depth window is
      ``depth = surface_z_um - z``: ``ymin = surface_z_um - zmax`` (0 for ``zmax=None``,
      clipped to 0 if the box reaches above the surface) and ``ymax = surface_z_um - zmin``.
      A window that is empty (``zmax <= zmin``) or lies entirely at or above the surface
      (``zmin >= surface_z_um``) raises ``ValueError``, so it never silently gives no doping.

    With ``mask_um = (xmin, xmax)``, the profile is multiplied by :func:`lateral` for that
    opening, with the step's ``lateral_straggle`` (``ImplantGaussian``) or sharp edges.
    """
    from gdsfactory.technology.processes import (
        DopingConstant,
        ImplantGaussian,
        ImplantPhysical,
    )

    kind = dopant_kind(step.ion)
    lateral_straggle = 0.0
    if isinstance(step, ImplantGaussian):
        if step.vertical_straggle is None:
            raise ValueError(f"ImplantGaussian {step.name!r} has no vertical_straggle")
        profile: DopingProfile = gaussian(
            step.peak_conc, step.range, step.vertical_straggle
        )
        lateral_straggle = step.lateral_straggle or 0.0
    elif isinstance(step, ImplantPhysical):
        from gplugins.process.implant_tables import (
            depth_in_silicon,
            straggle_in_silicon,
        )

        table = _IONS[step.ion.lower()][1]
        if table is None:
            raise ValueError(f"no implant table for ion {step.ion!r}")
        profile = gaussian_from_dose(
            step.dose,
            float(depth_in_silicon[table](step.energy)),
            float(straggle_in_silicon[table](step.energy)),
            truncation=truncation,
        )
    elif isinstance(step, DopingConstant):
        profile = _doping_constant_profile(step, surface_z_um)
    else:
        raise TypeError(f"not a doping step: {type(step).__name__}")
    if mask_um is not None:
        profile = profile * lateral(lateral_straggle, *mask_um)
    return kind, profile


def _doping_constant_profile(step, surface_z_um: float | None) -> Constant:
    """The depth window of a gdsfactory ``DopingConstant`` (z-up, absolute z, um)."""
    if surface_z_um is None:
        raise ValueError(
            f"DopingConstant {step.name!r}: pass surface_z_um, the z (um) of the top of the "
            "doped material; zmin/zmax are absolute z-up coordinates"
        )
    zmax = surface_z_um if step.zmax is None else step.zmax
    if not zmax > step.zmin:
        raise ValueError(
            f"DopingConstant {step.name!r}: the window zmin={step.zmin!r}..zmax={zmax!r} is empty"
        )
    if not step.zmin < surface_z_um:
        raise ValueError(
            f"DopingConstant {step.name!r}: the window zmin={step.zmin!r}..zmax={zmax!r} lies "
            f"above the surface z={surface_z_um!r}"
        )
    return constant(
        step.peak_conc,
        ymin=max(0.0, surface_z_um - zmax),
        ymax=surface_z_um - step.zmin,
    )


# --- DEVSIM ----------------------------------------------------------------------------------


def _total(profiles: DopingProfile | Iterable[DopingProfile]) -> DopingProfile | None:
    if isinstance(profiles, DopingProfile):
        return profiles
    profiles = tuple(profiles)
    if not profiles:
        return None
    return profiles[0] if len(profiles) == 1 else _Sum(profiles)


def assign_doping(
    device: str,
    region: str,
    *,
    donors: DopingProfile | Iterable[DopingProfile] = (),
    acceptors: DopingProfile | Iterable[DopingProfile] = (),
    depth_axis: Literal["x", "y"] = "y",
    surface_um: float = 0.0,
    depth_sign: Literal[1, -1] = 1,
) -> None:
    """Create the ``Donors``, ``Acceptors`` and ``NetDoping`` node models of a DEVSIM region.

    ``Donors`` and ``Acceptors`` are the sums of the given profiles, written as node-model
    expressions of the node positions; ``NetDoping = Donors - Acceptors``. DEVSIM positions are
    in cm. The depth of a node is ``depth_sign * (coordinate_um - surface_um)`` along
    ``depth_axis``; the lateral position is the other axis (``y`` is 0 on a 1D mesh, so a 1D
    device uses ``depth_axis="x"``). Existing models of those names are replaced.
    """
    import devsim

    if depth_axis not in ("x", "y"):
        raise ValueError(f"depth_axis must be 'x' or 'y', got {depth_axis!r}")
    if depth_sign not in (1, -1):
        raise ValueError(f"depth_sign must be 1 or -1, got {depth_sign!r}")
    lateral_axis = "y" if depth_axis == "x" else "x"
    um = _num(1 / UM_TO_CM)
    depth = f"({_num(depth_sign)}*({depth_axis}*{um} - {_num(surface_um)}))"
    x_um = f"({lateral_axis}*{um})"
    for name, profiles in (("Donors", donors), ("Acceptors", acceptors)):
        total = _total(profiles)
        equation = "0" if total is None else total.expression(x_um, depth)
        devsim.node_model(device=device, region=region, name=name, equation=equation)
    devsim.node_model(
        device=device, region=region, name="NetDoping", equation="Donors - Acceptors"
    )


# --- layer-based doping used by get_simulation (unchanged) ------------------------------------


class DopingLayerLevel(BaseModel):
    """Level for doping layer.

    Parameters:
        layer: (GDSII Layer number, GDSII datatype).
        type: str: "Acceptor" or "Donor".
        z_profile: callable evaluating doping versus depth.
        xy_profile: callable modulating z_profile at the edge of the layer.

    """

    layer: Layer
    type: str
    z_profile: Callable
    model_config = ConfigDict(frozen=True, extra="forbid")


cm3_to_um3 = 1e-12


def get_doping_info_generic(
    n_conc: float = 1e17,  # * cm3_to_um3,
    p_conc: float = 1e17,  # * cm3_to_um3,
    npp_conc: float = 1e18,  # * cm3_to_um3,
    ppp_conc: float = 1e18,  # * cm3_to_um3,
) -> dict[str, DopingLayerLevel]:
    layermap = LAYER

    return {
        "N": DopingLayerLevel(
            layer=layermap.N,
            type="Donor",
            z_profile=step(n_conc),
        ),
        "P": DopingLayerLevel(
            layer=layermap.P,
            type="Acceptor",
            z_profile=step(p_conc),
        ),
        "NPP": DopingLayerLevel(
            layer=layermap.NPP,
            type="Donor",
            z_profile=step(npp_conc),
        ),
        "PPP": DopingLayerLevel(
            layer=layermap.PPP,
            type="Acceptor",
            z_profile=step(ppp_conc),
        ),
    }


def step(c, zmin=-np.inf, zmax=np.inf):
    """Step function doping of value c, between zmin and zmax."""
    return lambda y: c
