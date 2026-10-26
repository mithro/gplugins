# SPDX-License-Identifier: MIT
"""A 1D layer stack (e.g. poly gate / oxide / silicon body) built directly as a DEVSIM device.

:func:`create_1d_stack` lays the :class:`Layer1D` layers out along x, in order, from x = 0,
meshes them with ``devsim.create_1d_mesh`` (node spacing from :class:`Mesh1D`, geometrically
graded inside each layer), assigns each silicon layer's doping (:mod:`gplugins.devsim.doping`),
sets the material parameters from explicit :class:`~gplugins.devsim.materials.Constants`,
adds the material-pair interfaces and the two typed contacts, and solves the equilibrium
(0 V on every contact). It returns a :class:`~gplugins.devsim.get_solver.DDComponent`.

Names: the regions are the layer names, the interface between layers ``a`` and ``b`` is
``a___b``, and the contacts are the :class:`~gplugins.devsim.contacts.Contact` names. A 1D
DEVSIM mesh has exactly two contacts, one at each end; each is ``InterfaceRef(<end layer>,
None)``. DEVSIM works in cm: positions are cm, a charge is C/cm^2 and a capacitance F/cm^2.

Statistics are Boltzmann (:mod:`gplugins.devsim.materials`). In a degenerate poly gate
(1e20 n+, 3.6e19 p+) the Fermi level lies past the band edge, so Boltzmann is approximate
there: the gf180-tcad Phase 2 plan estimates about 40 mV on Vfb, a documented gf180-tcad
limitation (https://github.com/mithro/gf180-tcad/tree/main/docs/limitations, Task 13a).
"""

from __future__ import annotations

from collections.abc import Sequence

# The stack's data types and its node layout are backend-neutral: they live in the device
# description (the backend seam) and are re-exported here unchanged. The DEVSIM device is built
# by the DEVSIM backend (gplugins.devsim.backends.devsim.create_1d_device).
from gplugins.devsim.constants import CONSTANTS, Constants
from gplugins.devsim.device_description import (
    Contact,
    Layer1D,
    Mesh1D,
    SolverSettings,
    Stack1D,
    graded_spacings,
)
from gplugins.devsim.device_description import (  # the old private name
    check_1d_contacts as _check_contacts,
)

__all__ = ["UM_TO_CM", "Layer1D", "Mesh1D", "create_1d_stack", "graded_spacings"]

UM_TO_CM = 1e-4


def create_1d_stack(
    name: str,
    layers: Sequence[Layer1D],
    contacts: Sequence[Contact],
    mesh: Mesh1D,
    constants: Constants = CONSTANTS,
    *,
    relative_error: float = 1e-10,
    absolute_error: float = 1e10,
    maximum_iterations: int = 100,
    extended_precision: bool = True,
):
    """Build, and solve at equilibrium, the 1D DEVSIM device ``name`` (mesh ``<name>_mesh``).

    The arguments form a :class:`~gplugins.devsim.device_description.Stack1D` (validated
    there), which the DEVSIM backend builds
    (:func:`gplugins.devsim.backends.devsim.create_1d_device`).

    Args:
        layers: the stack in order along x (x = 0 at the first face of ``layers[0]``).
        contacts: two contacts, ``InterfaceRef(layers[0].name, None)`` and
            ``InterfaceRef(layers[-1].name, None)``; ``ohmic`` on silicon, ``gate`` on oxide.
            A :class:`~gplugins.devsim.contacts.LayerRef` or an interior end raises.
        mesh: the node spacing of every layer.
        constants: the material constants (never DEVSIM ``simple_physics``' defaults).
        relative_error, absolute_error, maximum_iterations: DEVSIM's DC tolerances, also used
            by the returned component's :meth:`~DDComponent.solve_dc`. A solve that does not
            converge raises :class:`~gplugins.devsim.convergence.ConvergenceError`.
        extended_precision: set DEVSIM's (global) extended solver/model/equation precision.

    Returns:
        A :class:`~gplugins.devsim.get_solver.DDComponent` holding the solved device.
    """
    from gplugins.devsim.backends.devsim import create_1d_device

    stack = Stack1D(
        name=name,
        layers=tuple(layers),
        contacts=tuple(contacts),
        mesh=mesh,
        constants=constants,
        solver=SolverSettings(
            absolute_error=absolute_error,
            relative_error=relative_error,
            maximum_iterations=maximum_iterations,
            extended_precision=extended_precision,
        ),
    )
    return create_1d_device(stack)
