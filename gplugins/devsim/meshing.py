# SPDX-License-Identifier: MIT
"""Deterministic, single-threaded meshwell 2.5 meshing of a component cross-section for DEVSIM.

:func:`mesh_xsection` is meshwell's own ``generate_mesh`` pipeline
(``meshwell/orchestrator.py``: ``cad`` -> ``ModelManager`` -> ``load_occ_entities`` ->
``meshwell.mesh.mesh``), replicated here so that every stage runs on one thread:

- ``generate_mesh`` builds its meshing ``ModelManager`` without ``n_threads``, so that stage
  takes the import-time ``os.cpu_count()`` default and sets ``General.NumThreads``,
  ``Mesh.MaxNumThreads1D/2D/3D`` and ``Geometry.OCCParallel`` from it;
- ``ModelManager`` re-initialises gmsh (``gmsh.finalize()``/``gmsh.initialize()``), which
  resets every gmsh option set before it.

So ``cad`` and ``ModelManager`` get ``n_threads=1`` explicitly, and :func:`_pin_gmsh` runs as
meshwell's ``pre_2d_hook``, after the last re-initialisation and right before
``generate(2)``. It pins ``Mesh.RandomSeed`` and turns every gmsh thread option to 1, so the
mesh does not depend on ``PYTHON_CPU_COUNT`` or the host's core count.

It mirrors meshwell 2.5.2 ``meshwell/orchestrator.py`` lines 219-266 (``generate_mesh``: the
``CADSettings``, ``cad`` call, ``ModelManager`` set-up, ``Geometry.OCCBoundsUseStl``,
``load_occ_entities`` and ``mesh`` call), and imports the semi-internal
``meshwell.occ_xao_writer.default_interface_aabb_tolerance`` as that code does. Re-sync it with
``generate_mesh`` on every meshwell upgrade.

The mesh is written as MSH 2.2, which DEVSIM reads. Physical groups are named by meshwell:
the entity names (``a``), the interfaces between two entities (``a___b``) and the exterior
boundary of an entity (``a___None``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import gmsh
import meshio
import numpy as np
from meshwell.cad_settings import CADSettings
from meshwell.mesh import mesh
from meshwell.model import ModelManager
from meshwell.occ_xao_writer import default_interface_aabb_tolerance
from meshwell.orchestrator import cad
from meshwell.resolution import ConstantInField, ResolutionSpec, ThresholdField

#: ``Mesh.RandomSeed`` used for every mesh (gmsh's own default; pinned so it cannot drift).
GMSH_RANDOM_SEED = 1
#: MSH format version written; DEVSIM's ``create_gmsh_mesh`` reads 2.2.
MSH_VERSION = 2.2
INTERFACE_DELIMITER = "___"
BOUNDARY_DELIMITER = "None"

_THREAD_OPTIONS = (
    "General.NumThreads",
    "Mesh.MaxNumThreads1D",
    "Mesh.MaxNumThreads2D",
    "Mesh.MaxNumThreads3D",
)


def _pin_gmsh(
    *,
    resolution_min: float | None = None,
    size_callback: Callable | None = None,
) -> None:
    """Pin the gmsh options that decide the mesh (run right before ``generate(2)``)."""
    for name in _THREAD_OPTIONS:
        gmsh.option.setNumber(name, 1)
    gmsh.option.setNumber("Geometry.OCCParallel", 0)
    gmsh.option.setNumber("Mesh.RandomSeed", GMSH_RANDOM_SEED)
    if resolution_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", resolution_min)
    if size_callback is not None:
        gmsh.model.mesh.setSizeCallback(size_callback)


def legacy_resolutions(
    resolutions: dict[str, dict[str, float]],
    resolution_max: float,
) -> dict[str, list[ResolutionSpec]]:
    """The meshwell 2.5 form of a pre-meshwell ``{"name": {"resolution": r, "distance": d}}``.

    gplugins <= 0.8 (``gplugins/gmsh/refine.py::surface_interface_refinement``) gave each
    named surface a constant size ``r`` inside it, and a size growing linearly from ``r`` on
    its boundary curves to ``resolution_max`` at distance ``d``. These are meshwell's
    :class:`~meshwell.resolution.ConstantInField` on the surfaces and
    :class:`~meshwell.resolution.ThresholdField` on the curves.
    """
    out = {}
    for name, spec in resolutions.items():
        r, d = spec["resolution"], spec["distance"]
        out[name] = [
            ConstantInField(resolution=r, apply_to="surfaces"),
            ThresholdField(
                sizemin=r,
                sizemax=resolution_max,
                distmin=0,
                distmax=d,
                apply_to="curves",
            ),
        ]
    return out


def mesh_xsection(
    entities: Sequence,
    output_file: Path | str,
    resolution_specs: dict[str, list[ResolutionSpec]] | None = None,
    *,
    resolution_min: float | None = None,
    resolution_max: float = 0.2,
    global_scaling: float = 1e-4,
    point_tolerance: float | None = None,
    global_meshsize_array: np.ndarray | None = None,
    global_meshsize_interpolant_func: Callable | None = None,
    verbosity: int = 0,
) -> meshio.Mesh:
    """Mesh 2D meshwell ``entities`` single-threaded and write ``output_file`` as MSH 2.2.

    Args:
        entities: meshwell ``PolySurface`` (or other 2D) entities; lowest ``mesh_order`` wins
            overlaps, ties go to the first in the list.
        output_file: the mesh file written (any extension gmsh writes as MSH, e.g.
            ``.msh`` or ``.msh2``).
        resolution_specs: meshwell resolution specs by physical name.
        resolution_min: ``Mesh.CharacteristicLengthMin`` (None: gmsh's default).
        resolution_max: ``Mesh.CharacteristicLengthMax``, meshwell's
            ``default_characteristic_length``.
        global_scaling: factor applied to the coordinates on output (1e-4: um to cm).
        point_tolerance: meshwell CAD point tolerance (None: meshwell's default).
        global_meshsize_array: optional ``[x, y, z, size]`` rows (unscaled units). If given,
            the mesh size is ``min(gmsh size, interpolated size)`` through a gmsh size
            callback, as in gplugins <= 0.8.
        global_meshsize_interpolant_func: the interpolator class for
            ``global_meshsize_array`` (e.g. ``scipy.interpolate.NearestNDInterpolator``).
        verbosity: gmsh ``General.Terminal``.

    Returns:
        The mesh, as meshio reads it back from gmsh. gmsh stays initialised, so its
        options can be inspected afterwards.
    """
    settings = CADSettings.from_kwargs(None, point_tolerance=point_tolerance)
    tolerance = settings.point_tolerance

    occ_entities = cad(
        entities=list(entities),
        cad_settings=settings,
        n_threads=1,
        interface_delimiter=INTERFACE_DELIMITER,
        boundary_delimiter=BOUNDARY_DELIMITER,
    )

    mm = ModelManager(n_threads=1, point_tolerance=tolerance)
    mm.cad_settings = settings
    mm.ensure_initialized(str(mm.filename))
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    mm.load_occ_entities(
        occ_entities,
        remove_all_duplicates=False,
        interface_delimiter=INTERFACE_DELIMITER,
        boundary_delimiter=BOUNDARY_DELIMITER,
        interface_aabb_tolerance=default_interface_aabb_tolerance(tolerance),
    )

    size_callback = None
    if global_meshsize_array is not None:
        if global_meshsize_interpolant_func is None:
            from scipy.interpolate import NearestNDInterpolator

            global_meshsize_interpolant_func = NearestNDInterpolator
        array = np.asarray(global_meshsize_array, dtype=np.float64)
        interpolant = global_meshsize_interpolant_func(array[:, 0:3], array[:, 3])

        def size_callback(dim, tag, x, y, z, lc):
            return min(lc, float(interpolant(x, y, z)))

    def pre_2d_hook() -> None:
        _pin_gmsh(resolution_min=resolution_min, size_callback=size_callback)

    result = mesh(
        dim=2,
        model=mm,
        default_characteristic_length=resolution_max,
        resolution_specs=resolution_specs or {},
        global_scaling=global_scaling,
        gmsh_version=MSH_VERSION,
        n_threads=1,
        verbosity=verbosity,
        interface_delimiter=INTERFACE_DELIMITER,
        boundary_delimiter=BOUNDARY_DELIMITER,
        pre_2d_hook=pre_2d_hook,
    )
    # Written here rather than through mesh(output_file=...), which forces a ".msh" suffix.
    gmsh.write(str(output_file))
    return result
