from __future__ import annotations

from itertools import combinations

import numpy as np
from devsim import (
    add_gmsh_contact,
    add_gmsh_interface,
    add_gmsh_region,
    create_device,
    create_gmsh_mesh,
    finalize_mesh,
    get_node_model_values,
    node_solution,
    set_node_values,
    write_devices,
)
from gdsfactory import Component
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from meshwell.polysurface import PolySurface
from meshwell.resolution import ResolutionSpec
from scipy.interpolate import NearestNDInterpolator
from shapely.geometry import LineString, Polygon

from gplugins.common.base_models.component import fuse_polygons
from gplugins.devsim.doping import get_doping_info_generic
from gplugins.devsim.meshing import legacy_resolutions, mesh_xsection
from gplugins.meshwell import get_meshwell_cross_section, get_u_bounds_polygons

um_to_cm = 1e-4


def create_2Duz_simulation(
    component: Component,
    xsection_bounds: tuple[tuple[float, float], tuple[float, float]],
    full_layer_stack: LayerStack,
    physical_layer_stack: LayerStack,
    doping_info,  # Dict[str, DopingLayerLevel],
    contact_info,
    resolutions: dict[str, list[ResolutionSpec]] | dict[str, dict] | None = None,
    mesh_scaling_factor: float = um_to_cm,
    default_resolution_min: float = 0.001,
    default_resolution_max: float = 0.2,
    background_tag: str | None = None,
    background_padding: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
    temp_file_name: str = "temp.msh2",
    devsim_mesh_name: str = "temp",
    devsim_device_name: str = "temp",
    devsim_simulation_filename: str = "devsim.dat",
    global_meshsize_array: np.array | None = None,
    global_meshsize_interpolant_func: callable | None = NearestNDInterpolator,
):
    """Mesh the u-z cross-section of ``component`` along ``xsection_bounds`` into DEVSIM.

    The mesh comes from meshwell 2.5 through :func:`gplugins.devsim.meshing.mesh_xsection`
    (single-threaded, pinned gmsh seed, MSH 2.2), in units of ``mesh_scaling_factor`` (cm).

    Args:
        resolutions: meshwell resolution specs by layer name, e.g.
            ``{"core": [ConstantInField(resolution=0.02, apply_to="surfaces")]}``. The
            gplugins <= 0.8 form ``{"core": {"resolution": 0.02, "distance": 1}}`` is also
            accepted (see :func:`gplugins.devsim.meshing.legacy_resolutions`).
        default_resolution_min, default_resolution_max: gmsh's smallest and largest element
            size (um).
        background_tag: if given, a background region of that name and material fills the
            bounding box of the stack, padded by ``background_padding`` (u-left, z-down,
            u-right, z-up, um).
        global_meshsize_array: ``[x, y, z, size]`` rows (um) that replace ``resolutions``.

    Naming: DEVSIM regions are the layer names of ``physical_layer_stack`` plus one region
    per contact (named after the contact). gmsh physical groups between two regions are
    ``a___b`` and on the outer boundary ``a___None`` (meshwell's delimiters); a contact is
    the ``<contact>___<layer>`` (or ``<layer>___<contact>``) group, and every ``a___b`` group
    between two regions is added as a DEVSIM interface of that name.

    Returns:
        ``(device name, {material: [region names]}, {(material, material): [interfaces]})``.
    """
    # Replace relevant physical entities by contacts
    simulation_layertack = physical_layer_stack
    for contact_name, contact_dict in contact_info.items():
        contact_layer = full_layer_stack.layers[
            contact_dict["physical_layerlevel_to_replace"]
        ]
        layerlevel = LayerLevel(
            layer=contact_dict["gds_layer"],
            thickness=contact_layer.thickness,
            zmin=contact_layer.zmin,
            material=contact_layer.material,
            sidewall_angle=contact_layer.sidewall_angle,
            info=contact_layer.info,
        )
        simulation_layertack.layers[contact_name] = layerlevel

    # Get structural mesh (meshwell 2.5, single-threaded; see gplugins.devsim.meshing)
    surfaces = xsection_entities(
        component,
        xsection_bounds,
        simulation_layertack,
        background_tag=background_tag,
        background_padding=background_padding,
    )
    if global_meshsize_array is not None:
        # As in gplugins <= 0.8: a global size map replaces the resolutions.
        resolution_specs = {}
    elif resolutions and all(isinstance(v, dict) for v in resolutions.values()):
        resolution_specs = legacy_resolutions(resolutions, default_resolution_max)
    else:
        resolution_specs = resolutions or {}
    mesh = mesh_xsection(
        surfaces,
        temp_file_name,
        resolution_specs,
        resolution_min=default_resolution_min,
        resolution_max=default_resolution_max,
        global_scaling=mesh_scaling_factor,
        global_meshsize_array=global_meshsize_array,
        global_meshsize_interpolant_func=global_meshsize_interpolant_func,
    )

    # The gmsh physical group names. meshio has no cell sets for MSH 2.2 (which DEVSIM reads),
    # so they come from its field_data (name -> [tag, dim]).
    physical_names = set(mesh.field_data) | set(mesh.cell_sets_dict)

    # Get doping layer bounds
    doping_polygons = {}
    for layername, layer_obj in doping_info.items():
        fused_polygons = fuse_polygons(component, LogicalLayer(layer=layer_obj.layer))
        bounds = get_u_bounds_polygons(fused_polygons, xsection_bounds)
        doping_polygons[layername] = [np.array(bound) * um_to_cm for bound in bounds]

    # Regions, tagged by material
    regions = {}
    create_gmsh_mesh(file=temp_file_name, mesh=devsim_mesh_name)
    simulation_layer_stack_dict = simulation_layertack.to_dict()
    for name, values in simulation_layer_stack_dict.items():
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=name,
            region=name,
            material=values["material"],
        )
        if values["material"] in regions:
            regions[values["material"]].append(name)
        else:
            regions[values["material"]] = [name]
    if background_tag:
        simulation_layer_stack_dict[background_tag] = {"material": background_tag}
        add_gmsh_region(
            mesh=devsim_mesh_name,
            gmsh_name=background_tag,
            region=background_tag,
            material=background_tag,
        )
        regions[background_tag] = [background_tag]

    # Contacts
    contacts = {}
    for contact_name, contact in contact_info.items():
        layer1 = contact_name
        layer2 = contact["physical_layerlevel_to_contact"]
        interface = f"{layer1}___{layer2}"
        if interface not in physical_names:
            interface = f"{layer2}___{layer1}"
        add_gmsh_contact(
            gmsh_name=interface,
            material=simulation_layertack.layers[contact_name].material,
            mesh=devsim_mesh_name,
            name=contact_name,
            region=layer2,
        )
        contacts[contact_name] = interface

    # Interfaces (that are not contacts), labeled by material-material
    interfaces = {}
    for (name1, values1), (name2, values2) in combinations(
        simulation_layer_stack_dict.items(), 2
    ):
        interface = f"{name1}___{name2}"
        if interface not in physical_names:
            interface = f"{name2}___{name1}"
            if interface not in physical_names:
                continue
        # if interface not in contacts.values():
        add_gmsh_interface(
            gmsh_name=interface,
            mesh=devsim_mesh_name,
            name=interface,
            region0=name1,
            region1=name2,
        )
        interfaces.setdefault((values1["material"], values2["material"]), []).append(
            interface
        )
    finalize_mesh(mesh=devsim_mesh_name)
    create_device(mesh=devsim_mesh_name, device=devsim_device_name)

    # Assign doping fields silicon
    for region_name in regions["si"]:
        xpos = get_node_model_values(
            device=devsim_device_name, region=region_name, name="x"
        )
        # ypos = get_node_model_values(device=devsim_device_name, region="si", name="y")
        acceptor = np.zeros_like(xpos)
        donor = np.zeros_like(xpos)
        for layername, bounds in doping_polygons.items():
            print(layername, bounds)
            for bound in bounds:
                print(bound)
                node_inds = np.intersect1d(
                    np.where(xpos >= bound[0] * np.ones_like(xpos)),
                    np.where(xpos <= bound[1] * np.ones_like(xpos)),
                )
                # Assume step doping for now (does not depend on y, or z)
                if doping_info[layername].type == "Acceptor":
                    acceptor[node_inds] += doping_info[layername].z_profile(0)
                elif doping_info[layername].type == "Donor":
                    donor[node_inds] += doping_info[layername].z_profile(0)
                else:
                    raise ValueError(
                        f'Doping type "{doping_info[layername].type}" not supported.'
                    )
        net_doping = donor - acceptor

        node_solution(device=devsim_device_name, region=region_name, name="Acceptors")
        set_node_values(
            device=devsim_device_name,
            region=region_name,
            name="Acceptors",
            values=acceptor,
        )
        node_solution(device=devsim_device_name, region=region_name, name="Donors")
        set_node_values(
            device=devsim_device_name, region=region_name, name="Donors", values=donor
        )
        node_solution(device=devsim_device_name, region=region_name, name="NetDoping")
        set_node_values(
            device=devsim_device_name,
            region=region_name,
            name="NetDoping",
            values=net_doping,
        )

    # Assign doping field contact
    for contact_name in contacts:
        xpos = get_node_model_values(
            device=devsim_device_name, region=contact_name, name="x"
        )
        node_solution(device=devsim_device_name, region=contact_name, name="NetDoping")
        set_node_values(
            device=devsim_device_name,
            region=contact_name,
            name="NetDoping",
            values=[0] * len(xpos),
        )

    # Cast to cm
    # for region_name in get_region_list(device=devsim_device_name):
    #     xpos = get_node_model_values(
    #         device=devsim_device_name, region=region_name, name="x"
    #     )
    #     set_node_values(
    #         device=devsim_device_name, region=region_name, name="x", values=np.array(xpos)*um_to_cm
    #     )
    #     ypos = get_node_model_values(
    #         device=devsim_device_name, region=region_name, name="y"
    #     )
    #     set_node_values(
    #         device=devsim_device_name, region=region_name, name="y", values=np.array(ypos)*um_to_cm
    #     )

    if devsim_simulation_filename:
        write_devices(file=devsim_simulation_filename, type="tecplot")

    return devsim_device_name, regions, interfaces


def xsection_entities(
    component: Component,
    xsection_bounds: tuple[tuple[float, float], tuple[float, float]],
    layer_stack: LayerStack,
    background_tag: str | None = None,
    background_padding: tuple[float, float, float, float] = (2.0, 2.0, 2.0, 2.0),
) -> list[PolySurface]:
    """The meshwell surfaces of the u-z cross-section of ``layer_stack`` along the line.

    u is the distance along ``xsection_bounds`` from its first point. Each layer with shapes
    on the line is one surface named after the layer. Overlaps go to the lowest
    ``mesh_order`` and ties to the first layer name in alphabetical order, the order of
    gplugins <= 0.8 (``order_layer_stack`` in ``uz_xsection_mesh``); meshwell itself breaks
    ties by list order, so the list is sorted that way.
    """
    surfaces = get_meshwell_cross_section(
        component=component,
        line=LineString(xsection_bounds),
        layer_stack=layer_stack,
        wafer_layer=None,
        wafer_padding=None,
        name_by="layer",
    )
    surfaces.sort(key=lambda s: (s.mesh_order, s.physical_name))
    if background_tag:
        surfaces.append(
            _background_surface(layer_stack, xsection_bounds, background_tag, background_padding)
        )
    return surfaces


def _background_surface(layer_stack, xsection_bounds, tag, padding) -> PolySurface:
    """A ``tag`` rectangle around the stack's z-range and the cross-section line, padded.

    As gplugins <= 0.8 ``uz_xsection_mesh``: it has the highest mesh_order, so it only fills
    what no layer claims.
    """
    levels = layer_stack.layers.values()
    zs = [z for lv in levels for z in (lv.zmin, lv.zmin + lv.thickness)]
    length = float(np.linalg.norm(np.array(xsection_bounds[1]) - np.array(xsection_bounds[0])))
    left, down, right, up = padding
    zmin, zmax = min(zs), max(zs)
    box = Polygon(
        [
            (-left, zmin - down),
            (-left, zmax + up),
            (length + right, zmax + up),
            (length + right, zmin - down),
        ]
    )
    order = max(lv.mesh_order for lv in levels) + 1
    return PolySurface(polygons=box, physical_name=tag, mesh_order=order)


if __name__ == "__main__":
    import gdsfactory as gf
    from gdsfactory.gpdk import LAYER, LAYER_STACK, PDK

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
            k: LAYER_STACK.layers[k]
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

    # Contact info contains (could be improved)
    # gds_layer to use for this contact
    # physical_layerlevel_1: physical to replace by the contact
    # physical_layerlevel_2: interface to physical_layerlevel_1 to define as the contact
    # two layers defining an interface, and the dummy layer for position refinement
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

    resolutions = {
        "core": {"resolution": 0.01, "distance": 2},
        "slab90": {"resolution": 0.03, "distance": 1},
    }

    # resolutions["via_contact"] = {"resolution": 0.1, "distance": 1}

    device_name, regions, interfaces = create_2Duz_simulation(
        component=waveguide,
        xsection_bounds=[(4, -4), (4, 4)],
        full_layer_stack=LAYER_STACK,
        physical_layer_stack=physical_layer_stack,
        doping_info=get_doping_info_generic(),
        contact_info=contact_info,
        resolutions=resolutions,
        mesh_scaling_factor=1e-4,
        background_tag="sio2",
    )

    print(device_name)
    print(regions)
    print(interfaces)
