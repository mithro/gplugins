# SPDX-License-Identifier: MIT
"""The DEVSIM backend of gplugins.devsim: everything that drives DEVSIM for a device.

This is the DEVSIM side of the backend seam (:mod:`gplugins.devsim.device_description`, which
states the contract). It turns a backend-neutral description into a solved DEVSIM device:

- :func:`create_1d_device` builds and solves a :class:`~gplugins.devsim.device_description.Stack1D`
  (what :func:`gplugins.devsim.stack1d.create_1d_stack` calls);
- :func:`create_2d_device` meshes and solves a
  :class:`~gplugins.devsim.device_description.CrossSection2D`
  (:meth:`DDComponent.from_description`, then :meth:`DDComponent.ddsolver`);
- :meth:`DDComponent.from_device` wraps a DEVSIM device built elsewhere;
- :func:`run_cv_sweep` runs a :class:`~gplugins.devsim.device_description.CVSweep` with
  :func:`gplugins.devsim.cv.sweep_cv`.

Before the seam this module was ``gplugins/devsim/get_solver.py``; ``gplugins.devsim.get_solver``
is still importable and is this module.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence

import devsim
import numpy as np
from devsim import (
    circuit_alter,
    circuit_element,
    delete_device,
    delete_mesh,
    edge_average_model,
    edge_from_node_model,
    get_circuit_node_value,
    get_contact_charge,
    get_contact_list,
    get_edge_model_list,
    get_edge_model_values,
    get_interface_list,
    get_node_model_list,
    get_node_model_values,
    get_region_list,
    set_node_values,
    set_parameter,
    write_devices,
)
from devsim import solve as devsim_solve
from devsim.python_packages import model_create, simple_physics
from gdsfactory.component import Component
from gdsfactory.pdk import get_layer_stack
from gdsfactory.technology import LayerStack

from gplugins.devsim import contacts as contacts_mod
from gplugins.devsim import materials
from gplugins.devsim.contacts import Contact
from gplugins.devsim.convergence import solve_dc
from gplugins.devsim.cv import CVResult, sweep_cv
from gplugins.devsim.device_description import CrossSection2D, CVSweep, Stack1D
from gplugins.devsim.doping import (
    DopingLayerLevel,
    assign_doping,
    get_doping_info_generic,
)
from gplugins.devsim.get_simulation import create_2Duz_simulation
from gplugins.devsim.materials import CONSTANTS, SIMPLE_PHYSICS_V080, Constants

UM_TO_CM = 1e-4  # DEVSIM works in cm


def set_universal_parameters(device, region) -> None:
    """Deprecated (unused): DEVSIM simple_physics' q, k and eps_0; see gplugins.devsim.materials."""
    universal = {
        "q": 1.6e-19,  # , 'coul'),
        "k": 1.3806503e-23,  # , 'J/K'),
        "Permittivity_0": 8.85e-14,  # , 'F/cm^2')
    }
    for k, v in universal.items():
        set_parameter(device=device, region=region, name=k, value=v)


class DDComponent:
    def __init__(
        self,
        component: Component,
        xsection_bounds: tuple[tuple[float, float], tuple[float, float]],
        full_layer_stack: LayerStack,
        physical_layer_stack: LayerStack,
        doping_info: dict[str, DopingLayerLevel],
        contact_info: dict | None = None,
        resolutions: dict[str, dict] | None = None,
        mesh_scaling_factor: float = 1e-4,
        background_tag: str | None = None,
        temp_file_name="temp.msh2",
        devsim_mesh_name="temp",
        devsim_device_name="temp",
        devsim_simulation_filename="devsim.dat",
        atol: float = 1e12,
        rtol: float = 1e-12,
        max_iter: int = 200,
        extended_precision: bool = True,
        contacts: Sequence[Contact] | None = None,
        constants: Constants | None = None,
    ) -> None:
        """Drift-diffusion solver on a Component cross-section.

        Arguments:
            # Device parameters
            component: component
            xsection_bounds: line defined by [x1,y1], [x2,y2] to use for the meshing
            full_layer_stack: complete layer_stack associated with component
            physical_layer_stack: layer_stack subset of full_layer_stack that is used to create the physical mesh
            doping_info: dict relating some full_layer_stack layers and doping profiles (see doping submodule), e.g.

                doping_info = {
                    "N": DopingLayerLevel(          # key is a name (not used currently)
                        layer=layermap.N,           # which layermap layer to
                        type="Donor",               # DEVSIM dopant label, current must be either "Donor" or "Acceptor"
                        z_profile=step(n_conc),     # callable mapping dopant density to z-value
                    ),
                    "P": DopingLayerLevel(
                        layer=layermap.P,
                        type="Acceptor",
                        z_profile=step(p_conc),
                    ),

            contacts: typed contacts (:class:`gplugins.devsim.contacts.Contact`), e.g.
                ``[Contact("anode", LayerRef((105, 0)), "ohmic"), ...]``. The stack must hold
                the contact levels. Solved with ``constants`` (default: the cited
                :data:`gplugins.devsim.materials.CONSTANTS`), with Poisson in the oxide
                regions, material-pair interfaces and the contacts' own boundary conditions.
            constants: material constants (:class:`gplugins.devsim.materials.Constants`).
            contact_info: **deprecated** (use ``contacts``): the gplugins <= 0.8 dict relating
                contact names and gds layers, e.g.

                contact_info = {
                    "anode": {                                              # key is contact name
                        "gds_layer": anode_layer,                           # gds layer to use for this contact
                        "physical_layerlevel_to_replace": "via_contact",    # first physical layer to use to model the contact
                        "physical_layerlevel_to_contact": "slab90",         # second physical layer, the interface with ^ defining the contact
                    },
                    "cathode": {
                        "gds_layer": cathode_layer,
                        "physical_layerlevel_to_replace": "via_contact",
                        "physical_layerlevel_to_contact": "slab90",
                    },
                }

                Without ``contacts`` the device is solved as gplugins v0.8.0 did, bit for bit:
                only the regions with ``NetDoping`` (silicon and the contact regions, as
                intrinsic silicon), with ohmic contacts, and by default with DEVSIM
                ``simple_physics``' values (:data:`gplugins.devsim.materials.SIMPLE_PHYSICS_V080`:
                eps_si 11.1, 300 K, n_i 1e10), unless ``constants`` is given.

            # Mesh parameters
            resolutions: dict of mesh resolutions (see meshing module)
            mesh_scaling_factor: scales mesh dimensions (see meshing module). Defaults to um --> cm conversion
            background_tag: label of background polygon (see meshing module).

            # Filesystem parameters
            temp_file_name = temp_file_name
            devsim_mesh_name = devsim_mesh_name
            devsim_device_name = devsim_device_name
            devsim_simulation_filename = (devsim_simulation_filename,)

            # Solver parameters
            atol: absolute tolerance threshold for self-consistent solve
            rtol: relative tolerance threshold for self-consistent solve
            max_iter : maximum number of iterations for self-consistent solve
            extended_precision: if True (default), sets double precision

        A solve that does not converge raises
        :class:`gplugins.devsim.convergence.ConvergenceError`.
        """
        if contacts is not None and contact_info:
            raise ValueError("pass contacts or contact_info, not both")
        if contacts is None and contact_info is None:
            raise ValueError(
                "pass contacts=[gplugins.devsim.contacts.Contact(...)] (or the deprecated "
                "contact_info dict); neither was given"
            )
        if contacts is None:  # contact_info given: the deprecated gplugins v0.8.0 path
            warnings.warn(
                "DDComponent(contact_info=...) is deprecated and solves with gplugins "
                "v0.8.0's physics; pass contacts=[gplugins.devsim.contacts.Contact(...)] "
                "and constants=gplugins.devsim.materials.Constants()",
                DeprecationWarning,
                stacklevel=2,
            )
            self.legacy = True
            default_constants = SIMPLE_PHYSICS_V080
        else:
            self.legacy = False
            default_constants = CONSTANTS
        # Set attributes
        self.component = component
        self.xsection_bounds = xsection_bounds
        self.full_layer_stack = full_layer_stack or get_layer_stack()
        self.physical_layer_stack = physical_layer_stack
        self.doping_info = doping_info or get_doping_info_generic()
        self.contact_info = contact_info or {}
        self.contacts = tuple(contacts or ())
        self.constants = constants or default_constants
        self.resolutions = resolutions or {}
        self.mesh_scaling_factor = mesh_scaling_factor
        self.background_tag = background_tag
        self.temp_file_name = temp_file_name
        self.devsim_mesh_name = devsim_mesh_name
        self.devsim_device_name = devsim_device_name
        self.devsim_simulation_filename = devsim_simulation_filename
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.region_classes: dict[str, str | None] = {}
        self.last_solve_info: dict | None = None

        # Set precision
        if extended_precision:
            self.set_extended_precision()

    @classmethod
    def from_device(
        cls,
        *,
        device: str,
        mesh: str,
        region_classes: Mapping[str, str | None],
        contacts: Sequence[Contact],
        constants: Constants,
        atol: float,
        rtol: float,
        max_iter: int,
        extended_precision: bool = True,
    ) -> DDComponent:
        """A DDComponent for a DEVSIM device built elsewhere (e.g. by ``create_1d_stack``).

        ``device`` must have its mesh, regions, contacts, interfaces and (on silicon regions)
        ``NetDoping``. ``region_classes`` maps every region to ``"silicon"``, ``"oxide"`` or
        ``None``. The physics are set up and the equilibrium is solved.
        """
        self = cls.__new__(cls)
        self.legacy = False
        self.component = None
        self.xsection_bounds = None
        self.full_layer_stack = None
        self.physical_layer_stack = None
        self.doping_info = {}
        self.contact_info = {}
        self.contacts = tuple(contacts)
        self.constants = constants
        self.resolutions = {}
        self.mesh_scaling_factor = 1e-4
        self.background_tag = None
        self.temp_file_name = None
        self.devsim_mesh_name = mesh
        self.devsim_device_name = device
        self.devsim_simulation_filename = None
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        self.last_solve_info = None
        self.device = device
        self.regions = {}
        for region, region_class in region_classes.items():
            self.regions.setdefault(region_class, []).append(region)
        self.interfaces = {}
        self.region_classes = dict(region_classes)
        if extended_precision:
            self.set_extended_precision()
        self.setup_physics()
        return self

    @classmethod
    def from_description(
        cls,
        description: CrossSection2D,
        *,
        temp_file_name: str = "temp.msh2",
        devsim_mesh_name: str = "temp",
        devsim_device_name: str = "temp",
        devsim_simulation_filename: str | None = "devsim.dat",
    ) -> DDComponent:
        """A DDComponent for the 2D cross-section ``description``; :meth:`ddsolver` solves it.

        The keyword arguments are DEVSIM-backend choices (the mesh file and DEVSIM's names),
        not part of the description. The result is the same as passing the description's
        fields to the constructor (``contacts``, ``constants``, the tolerances of
        ``description.solver`` as ``atol``/``rtol``/``max_iter``/``extended_precision``).
        """
        solver = description.solver
        return cls(
            component=description.component,
            xsection_bounds=description.xsection_bounds,
            full_layer_stack=description.full_layer_stack,
            physical_layer_stack=description.physical_layer_stack,
            doping_info=dict(description.doping_info),
            resolutions=None if description.resolutions is None else dict(description.resolutions),
            mesh_scaling_factor=description.mesh_scaling_factor,
            background_tag=description.background_tag,
            temp_file_name=temp_file_name,
            devsim_mesh_name=devsim_mesh_name,
            devsim_device_name=devsim_device_name,
            devsim_simulation_filename=devsim_simulation_filename,
            atol=solver.absolute_error,
            rtol=solver.relative_error,
            max_iter=solver.maximum_iterations,
            extended_precision=solver.extended_precision,
            contacts=description.contacts,
            constants=description.constants,
        )

    def set_extended_precision(self) -> None:
        set_parameter(name="extended_solver", value=True)
        set_parameter(name="extended_model", value=True)
        set_parameter(name="extended_equation", value=True)

    # --- solving ------------------------------------------------------------------------------
    def solve_dc(
        self,
        *,
        absolute_error: float | None = None,
        relative_error: float | None = None,
        maximum_iterations: int | None = None,
    ) -> dict:
        """DC solve at this component's tolerances (or the given ones); raises unless converged.

        Returns DEVSIM's solve info, also kept as ``last_solve_info``.
        """
        self.last_solve_info = solve_dc(
            absolute_error=self.atol if absolute_error is None else absolute_error,
            relative_error=self.rtol if relative_error is None else relative_error,
            maximum_iterations=self.max_iter
            if maximum_iterations is None
            else maximum_iterations,
        )
        return self.last_solve_info

    def _contact(self, name: str) -> Contact | None:
        for c in self.contacts:
            if c.name == name:
                return c
        return None

    def set_bias(self, contact: str, value: float) -> None:
        """Set the DC bias (V) of ``contact``: its source if ``circuit=True``, else its parameter."""
        typed = self._contact(contact)
        if typed is not None and typed.circuit:
            circuit_alter(name=contacts_mod.source_name(contact), value=float(value))
        else:
            set_parameter(
                device=self.device,
                name=contacts_mod.bias_name(contact),
                value=float(value),
            )

    def contact_charge(self, contact: str) -> float:
        """DEVSIM's contact charge of ``contact`` (C/cm^2 in 1D, C/cm in 2D)."""
        return get_contact_charge(
            device=self.device, contact=contact, equation="PotentialEquation"
        )

    def region_charge(self, region: str) -> float:
        """Space charge q (p - n + NetDoping) integrated over ``region`` (C/cm^2 in 1D).

        0 for a region without ``NetDoping`` (oxide). The carriers are the drift-diffusion
        solution, or the equilibrium ones before it exists.
        """
        models = get_node_model_list(device=self.device, region=region)
        if "NetDoping" not in models:
            return 0.0
        n, p = ("Electrons", "Holes") if "Electrons" in models else (
            "IntrinsicElectrons",
            "IntrinsicHoles",
        )

        def values(name):
            return get_node_model_values(device=self.device, region=region, name=name)

        q = self.constants.q_c
        return q * math.fsum(
            v * (pi - ni + di)
            for v, ni, pi, di in zip(values("NodeVolume"), values(n), values(p), values("NetDoping"))
        )

    def electrode_charge(self, contact: str, regions: Sequence[str]) -> float:
        """The charge (C/cm^2 in 1D) of the electrode at ``contact``.

        DEVSIM's contact charge of ``contact`` plus the space charge of ``regions``, the
        semiconductor regions on that electrode's side of the insulator (e.g. the poly gate).
        For a metal gate on oxide ``regions`` is empty. In a neutral device, by Gauss, the gate
        electrode's charge equals the oxide D field and minus the body electrode's charge.
        """
        return self.contact_charge(contact) + math.fsum(self.region_charge(r) for r in regions)

    def ac_admittance(self, contact: str, frequency_hz: float) -> complex:
        """Small-signal admittance Y = I/V (A/cm^2 per V in 1D) of a ``circuit=True`` contact.

        A 1 V AC excitation is applied on this contact's source only, around the present DC
        solution; I is the current into the device (Y = j w C for a capacitor).
        """
        typed = self._contact(contact)
        if typed is None or not typed.circuit:
            raise ValueError(f"contact {contact!r} is not a circuit=True contact")
        for c in self.contacts:
            if c.circuit:
                circuit_alter(
                    name=contacts_mod.source_name(c.name),
                    param="acreal",
                    value=1.0 if c.name == contact else 0.0,
                )
        try:
            # The small-signal AC solve is a single linear solve around the converged DC point
            # (DEVSIM has no Newton iteration and no convergence info for it), so it is called
            # directly rather than through convergence.solve_dc.
            devsim_solve(type="ac", frequency=float(frequency_hz))
            node = f"{contacts_mod.source_name(contact)}.I"
            current = complex(
                get_circuit_node_value(node=node, solution="ssac_real"),
                get_circuit_node_value(node=node, solution="ssac_imag"),
            )
        finally:
            circuit_alter(name=contacts_mod.source_name(contact), param="acreal", value=0.0)
        # DEVSIM's source current flows from n1 through the source: the device draws -I.
        return -current

    # --- typed physics --------------------------------------------------------------------------
    def setup_physics(self) -> None:
        """Materials, equations, interfaces and contacts of the typed path; solve equilibrium.

        Stage 1 (Poisson, equilibrium carriers) as DEVSIM ``testing/mos_2d.py``: silicon and
        oxide regions, every contact, continuous potential on every interface between two
        regions with equations; solve. Stage 2: drift-diffusion on silicon, the ohmic contacts'
        carrier conditions, carrier continuity on silicon/silicon interfaces; solve.
        """
        device = self.device
        classes = self.region_classes
        for region, region_class in classes.items():
            if region_class == "silicon":
                materials.set_silicon(
                    device, region, **materials.silicon_parameters(self.constants)
                )
            elif region_class == "oxide":
                materials.set_oxide(device, region, **materials.oxide_parameters(self.constants))
        placed = {}
        for contact in self.contacts:
            (region,) = get_region_list(device=device, contact=contact.name)
            contacts_mod.check_contact_region(contact, region, classes.get(region))
            placed[contact.name] = region
            if contact.circuit:
                circuit_element(
                    name=contacts_mod.source_name(contact.name),
                    n1=contacts_mod.bias_name(contact.name),
                    n2="0",
                    value=0.0,
                    acreal=0.0,
                )
            else:
                set_parameter(
                    device=device, name=contacts_mod.bias_name(contact.name), value=0.0
                )
        for region, region_class in classes.items():
            if region_class == "silicon":
                materials.create_silicon_potential_only(device, region)
            elif region_class == "oxide":
                materials.create_oxide_potential_only(device, region)
        for contact in self.contacts:
            region = placed[contact.name]
            contacts_mod.create_contact_potential_only(device, region, contact, classes[region])
        interface_kinds = {}
        for interface in get_interface_list(device=device):
            r0, r1 = get_region_list(device=device, interface=interface)
            kind = contacts_mod.interface_kind(classes.get(r0), classes.get(r1))
            interface_kinds[interface] = kind
            if kind is not None:
                contacts_mod.create_interface_potential(device, interface)
        self.solve_dc()

        silicon = [r for r, k in classes.items() if k == "silicon"]
        if not silicon:
            return
        for region in silicon:
            materials.create_silicon_drift_diffusion(device, region)
        for contact in self.contacts:
            region = placed[contact.name]
            contacts_mod.create_contact_drift_diffusion(device, region, contact, classes[region])
        for interface, kind in interface_kinds.items():
            if kind == "drift_diffusion":
                contacts_mod.create_interface_carriers(device, interface)
        self.solve_dc()

    # --- gplugins v0.8.0 physics (contact_info) -------------------------------------------------
    def set_parameters(self, region, device) -> None:
        """Silicon parameters from ``self.constants`` (v0.8.0: simple_physics' values, 300 K)."""
        materials.set_silicon(device, region, **materials.silicon_parameters(self.constants))

    def initial_solution(self, device, region, circuit_contacts=None) -> None:
        # Create Potential, Potential@n0, Potential@n1
        model_create.CreateSolution(device, region, "Potential")

        # Create potential only physical models
        simple_physics.CreateSiliconPotentialOnly(device, region)

        # Set up the contacts applying a bias
        for i in get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconPotentialOnlyContact(
                    device, region, i, True
                )
            else:
                # it is more correct for the bias to be 0, and it looks like there is side effects
                set_parameter(
                    device=device, name=simple_physics.GetContactBiasName(i), value=0.0
                )
                simple_physics.CreateSiliconPotentialOnlyContact(device, region, i)

    def drift_diffusion_initial_solution(
        self, device, region, circuit_contacts=None
    ) -> None:
        # drift diffusion solution variables
        model_create.CreateSolution(device, region, "Electrons")
        model_create.CreateSolution(device, region, "Holes")

        # create initial guess from dc only solution
        set_node_values(
            device=device,
            region=region,
            name="Electrons",
            init_from="IntrinsicElectrons",
        )
        set_node_values(
            device=device, region=region, name="Holes", init_from="IntrinsicHoles"
        )

        # Set up equations
        simple_physics.CreateSiliconDriftDiffusion(device, region)
        for i in get_contact_list(device=device):
            if circuit_contacts and i in circuit_contacts:
                simple_physics.CreateSiliconDriftDiffusionAtContact(
                    device, region, i, True
                )
            else:
                simple_physics.CreateSiliconDriftDiffusionAtContact(device, region, i)

    def ddsolver(self, global_meshsize_array=None) -> None:
        """Initialize mesh and solver."""
        self.device, self.regions, self.interfaces = create_2Duz_simulation(
            component=self.component,
            xsection_bounds=self.xsection_bounds,
            full_layer_stack=self.full_layer_stack,
            physical_layer_stack=self.physical_layer_stack,
            doping_info=self.doping_info,
            contact_info=self.contact_info if self.legacy else None,
            contacts=None if self.legacy else self.contacts,
            resolutions=self.resolutions,
            mesh_scaling_factor=self.mesh_scaling_factor,
            background_tag=self.background_tag,
            temp_file_name=self.temp_file_name,
            devsim_mesh_name=self.devsim_mesh_name,
            devsim_device_name=self.devsim_device_name,
            global_meshsize_array=global_meshsize_array,
        )
        device = self.device

        if not self.legacy:
            # Silicon: the regions with NetDoping; oxide: by material; the rest (electrodes,
            # unknown materials) get no equations.
            material_of = {r: m for m, rs in self.regions.items() for r in rs}
            self.region_classes = {
                r: "silicon"
                if "NetDoping" in get_node_model_list(device=device, region=r)
                else (
                    "oxide"
                    if materials.material_class(material_of.get(r)) == "oxide"
                    else None
                )
                for r in get_region_list(device=device)
            }
            self.setup_physics()
            return

        # gplugins v0.8.0 (contact_info): the silicon models need NetDoping, so they apply
        # only to the regions that have it, and the silicon/silicon interface model only to
        # interfaces between two of them. create_2Duz_simulation sets NetDoping on the "si"
        # regions and (to 0) on the contact regions, which are solved as intrinsic silicon as
        # in v0.8.0; other regions (e.g. oxide) have none and get no silicon model.
        silicon = [
            r
            for r in get_region_list(device=device)
            if "NetDoping" in get_node_model_list(device=device, region=r)
        ]
        self.region_classes = {
            r: "silicon" if r in silicon else None for r in get_region_list(device=device)
        }

        for region in silicon:
            self.set_parameters(device=device, region=region)

        for region in silicon:
            self.initial_solution(device=device, region=region)

        for region in silicon:
            self.drift_diffusion_initial_solution(device=device, region=region)

        for interface in get_interface_list(device=device):
            if all(
                r in silicon
                for r in get_region_list(device=device, interface=interface)
            ):
                simple_physics.CreateSiliconSiliconInterface(
                    device=device, interface=interface
                )

        self.save_device("debug.dat")

        self.solve_dc()

    def ramp_voltage(
        self, Vfinal: float, Vstep: float, Vinit: float = 0.0, contact_name="cathode"
    ) -> None:
        """Ramps the solution from Vi to Vf."""
        V = Vinit
        while np.abs(V) <= np.abs(Vfinal):
            self.set_bias(contact_name, V)
            self.solve_dc()
            V += Vstep

    def get_node_index(self, region_name):
        """Maps head and tail nodes of from their edge index.

        From https://github.com/devsim/devsim_misc/blob/9a3c7056e0e3e7fc49e17031a706573350292d4d/refinement/refinement2.py#L45
        """
        if "node_index@n0" not in get_edge_model_list(
            device=self.device, region=region_name
        ):
            edge_from_node_model(
                node_model="node_index", device=self.device, region=region_name
            )
        return list(
            zip(
                [
                    int(x)
                    for x in get_edge_model_values(
                        device=self.device,
                        region=region_name,
                        name="node_index@n0",
                    )
                ],
                [
                    int(x)
                    for x in get_edge_model_values(
                        device=self.device,
                        region=region_name,
                        name="node_index@n1",
                    )
                ],
            )
        )

    def get_node_field(self, region_name, field_name="Electrons"):
        return get_node_model_values(
            device=self.device, region=region_name, name=field_name
        )

    def get_mean_edge_from_node_field(self, region_name, node_field="x"):
        edge_average_model(
            device=self.device,
            region=region_name,
            node_model=node_field,
            edge_model=f"{node_field}@mean",
            average_type="arithmetic",
        )
        return get_edge_model_values(
            device=self.device, region=region_name, name=f"{node_field}@mean"
        )

    def get_edge_field(self, region_name, field_name="EdgeLength"):
        return get_edge_model_values(
            device=self.device, region=region_name, name=field_name
        )

    def get_regions(self):
        return get_region_list(device=self.device)

    def save_device(self, filepath) -> None:
        """Save Device to a tecplot filepath that you can open with Paraview."""
        write_devices(file=filepath, type="tecplot")

    def delete_device(self) -> None:
        """Delete this devsim device and mesh, and the (global) circuit if it has circuit contacts."""
        delete_mesh(mesh=self.devsim_mesh_name)
        delete_device(device=self.device)
        if any(c.circuit for c in self.contacts):
            contacts_mod.delete_circuit()

    def get_refined_mesh(
        self,
        factor: float = 2.0,
        refine_dict: dict[str, dict] | None = None,
        refine_regions: tuple[str, ...] = ("si",),
    ):
        """Refines the mesh based on simulation result.

        Currently only remeshes in silicon regions.

        Arguments:
            factor: number to scale the mesh characteristic length down where refine_dict is True.
            refine_dict: Dict of fields:conditions to determine where to remesh, e.g.
                refine_dict = {
                    "Potential": {                                  # top-level key is field name
                        "func": lambda x0, x1: abs(x0 - x1),        # callable to apply to the local field difference before checking threshold
                        "threshold": 0.05,                          # will remesh if func(field) > threshold
                    },
                    "Electrons": {
                        "func": lambda x0, x1: np.abs(np.log10(x0) - np.log10(x1)),
                        "threshold": 1,
                    },
                }
            refine_regions: region keys where to refine.
        """
        xs = {}
        ys = {}
        lcs = {}
        refine = {}

        # Refined regions
        for regiontype, region_names in self.regions.items():
            for regionname in region_names:
                if regiontype in refine_regions:
                    xs[regionname] = np.array(
                        self.get_mean_edge_from_node_field(regionname, "x")
                    )
                    ys[regionname] = np.array(
                        self.get_mean_edge_from_node_field(regionname, "y")
                    )
                    lcs[regionname] = self.get_edge_field(
                        regionname, field_name="EdgeLength"
                    )
                    node_index = self.get_node_index(region_name=regionname)

                    refinements = []
                    for field_name, refinement in refine_dict.items():
                        field = np.array(self.get_node_field(regionname, field_name))
                        refinements.append(
                            np.array(
                                [
                                    refinement["func"](field[x[0]], field[x[1]])
                                    > refinement["threshold"]
                                    for x in node_index
                                ]
                            )
                        )

                    refine[regionname] = refinements[0]
                    for index in range(1, len(refinements)):
                        refine[regionname] = np.logical_or(
                            refinements[index], refinements[index - 1]
                        )

        um_to_cm = 1e-4
        xs = np.hstack([np.array(x) for x in xs.values()], dtype=np.float64) / um_to_cm
        ys = np.hstack([np.array(y) for y in ys.values()], dtype=np.float64) / um_to_cm
        lcs = (
            np.hstack([np.array(lc) for lc in lcs.values()], dtype=np.float64)
            / um_to_cm
        )
        refine = np.hstack([np.array(x) for x in refine.values()], dtype=bool)

        lcs_after = np.where(refine, lcs / factor, lcs)

        N = len(xs)
        meshing_field = np.zeros([N, 4])
        meshing_field[:, 0] = xs
        meshing_field[:, 1] = ys
        meshing_field[:, 2] = np.zeros(N)
        meshing_field[:, 3] = lcs_after
        return meshing_field


def create_1d_device(stack: Stack1D) -> DDComponent:
    """Build, and solve at equilibrium, the 1D DEVSIM device ``stack.name`` (mesh ``<name>_mesh``).

    The nodes are :meth:`Stack1D.node_positions_um` (cm in DEVSIM), each layer a region, each
    layer boundary an interface ``a___b`` and the two contacts the stack's ends. Each silicon
    layer's doping is assigned from :meth:`Stack1D.doping_surfaces_um`. The physics are set up
    from ``stack.constants`` and the equilibrium (0 V on every contact) is solved with
    ``stack.solver``'s tolerances (:meth:`DDComponent.from_device`). On failure the device, its
    mesh and (with a ``circuit=True`` contact) DEVSIM's circuit are deleted.
    """
    name, layers, contacts = stack.name, stack.layers, stack.contacts
    contact_end = stack.contact_ends()
    xs, boundary_node = stack.node_positions_um()

    mesh_name = f"{name}_mesh"
    devsim.create_1d_mesh(mesh=mesh_name)
    tags = {node: f"b{i}" for i, node in enumerate(boundary_node)}
    for k, x in enumerate(xs):
        right = xs[k + 1] - x if k + 1 < len(xs) else x - xs[k - 1]
        left = x - xs[k - 1] if k > 0 else right
        kw = {"tag": tags[k]} if k in tags else {}
        devsim.add_1d_mesh_line(
            mesh=mesh_name, pos=x * UM_TO_CM, ps=right * UM_TO_CM, ns=left * UM_TO_CM, **kw
        )
    for i, layer in enumerate(layers):
        devsim.add_1d_region(
            mesh=mesh_name,
            material=layer.material,
            region=layer.name,
            tag1=f"b{i}",
            tag2=f"b{i + 1}",
        )
    for i, interface in enumerate(stack.interfaces, start=1):
        devsim.add_1d_interface(mesh=mesh_name, name=interface, tag=f"b{i}")
    for contact in contacts:
        devsim.add_1d_contact(
            mesh=mesh_name, name=contact.name, tag=f"b{contact_end[contact.name]}", material="metal"
        )
    devsim.finalize_mesh(mesh=mesh_name)
    devsim.create_device(mesh=mesh_name, device=name)

    try:
        surfaces = stack.doping_surfaces_um()
        for layer in layers:
            if layer.material != "silicon":
                continue
            surface, sign = surfaces[layer.name]
            profile = {"donors" if layer.dopant == "donor" else "acceptors": layer.doping}
            assign_doping(
                name, layer.name, depth_axis="x", surface_um=surface, depth_sign=sign, **profile
            )
        solver = stack.solver
        return DDComponent.from_device(
            device=name,
            mesh=mesh_name,
            region_classes=stack.region_classes,
            contacts=list(contacts),
            constants=stack.constants,
            rtol=solver.relative_error,
            atol=solver.absolute_error,
            max_iter=solver.maximum_iterations,
            extended_precision=solver.extended_precision,
        )
    except BaseException:
        if name in devsim.get_device_list():
            devsim.delete_device(device=name)
        devsim.delete_mesh(mesh=mesh_name)
        if any(c.circuit for c in contacts):
            contacts_mod.delete_circuit()
        raise


def create_2d_device(cross_section: CrossSection2D, **options) -> DDComponent:
    """Mesh ``cross_section``, build it as a DEVSIM device and solve its equilibrium.

    :meth:`DDComponent.from_description` then :meth:`DDComponent.ddsolver`. ``options`` are
    from_description's DEVSIM choices (``temp_file_name``, ``devsim_mesh_name``,
    ``devsim_device_name``, ``devsim_simulation_filename``).
    """
    device = DDComponent.from_description(cross_section, **options)
    device.ddsolver()
    return device


def run_cv_sweep(device: DDComponent, sweep: CVSweep, **options) -> CVResult:
    """:func:`gplugins.devsim.cv.sweep_cv` of ``device`` for the bias plan ``sweep``.

    ``options`` are sweep_cv's solver options (``rel_tol``, ``abs_tol``, ``max_iterations``,
    ``extended_precision``, ``progress``, ``precision_fallback``, ``roundoff_accept``).
    """
    return sweep_cv(
        device,
        sweep.gate,
        sweep.vg,
        method=sweep.method,
        freq_hz=sweep.freq_hz,
        dv_qs=sweep.dv_qs,
        **options,
    )


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.gpdk import LAYER, PDK

    PDK.activate()

    # We choose a representative subdomain of the component
    waveguide = gf.Component()
    waveguide.add_ref(
        gf.functions.trim(
            component=gf.components.straight_pn(length=10, taper=None).copy(),
            domain=[[3, -4], [3, 4], [5, 4], [5, -4]],
        )
    )

    # We will restrict the physical mesh to a subset of layers:
    layermap = LAYER
    physical_layer_stack = LayerStack(
        layers={
            k: get_layer_stack().layers[k]
            for k in (
                "slab90",
                "core",
            )
        }
    )

    # Boundary conditions such as contacts are also defined with GDS polygons.
    # In DEVSIM, contacts must be N-1D, where N is the simulation dimensionality.
    # While complete dummy layers + stack definitions could be used, often we are interested in using existing layers as contact (for instance vias).
    # Since this means different contacts could correspond to the same physical layers, this requires more care in the meshing.

    # Simulation layers for contacts
    anode_layer = (105, 0)
    cathode_layer = (106, 0)

    # We will generate contacts where the vias intersect heavy doping
    via_contact_locations = waveguide.extract([layermap.VIAC])
    NPP_location = waveguide.extract([layermap.NPP])
    PPP_location = waveguide.extract([layermap.PPP])
    waveguide.add_ref(
        gf.boolean(
            via_contact_locations,
            NPP_location,
            operation="and",
            layer=anode_layer,
            layer1=layermap.VIAC,
            layer2=layermap.NPP,
        )
    )
    waveguide.add_ref(
        gf.boolean(
            via_contact_locations,
            PPP_location,
            operation="and",
            layer=cathode_layer,
            layer1=layermap.VIAC,
            layer2=layermap.PPP,
        )
    )
    contact_info = {
        "anode": {
            "gds_layer": anode_layer,
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        },
        "cathode": {
            "gds_layer": cathode_layer,
            "physical_layerlevel_to_replace": "via_contact",
            "physical_layerlevel_to_contact": "slab90",
        },
    }

    # Initial meshing
    resolutions = {
        "core": {"resolution": 0.02, "distance": 1},
        "slab90": {"resolution": 0.02, "distance": 1},
    }

    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layer_stack=get_layer_stack(),
        physical_layer_stack=physical_layer_stack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag=None,
    )

    c.ddsolver()
    c.save_device("test1.dat")

    meshing_field = c.get_refined_mesh(
        factor=2.0,
        refine_dict={
            "Potential": {
                "func": lambda x0, x1: abs(x0 - x1),
                "threshold": 0.05,
            },
            "Electrons": {
                "func": lambda x0, x1: np.abs(np.log10(x0) - np.log10(x1)),
                "threshold": 1,
            },
        },
        refine_regions=["si"],
    )

    c.delete_device()
    c = DDComponent(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layer_stack=get_layer_stack(),
        physical_layer_stack=physical_layer_stack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag=None,
    )

    c.ddsolver(global_meshsize_array=meshing_field)
    c.save_device("test2.dat")
