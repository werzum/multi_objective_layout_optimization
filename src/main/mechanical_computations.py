import math
import numpy as np
import geopandas as gpd
import plotly.graph_objects as go

from shapely.geometry import LineString, Point

from src.main import (
    plotting_2d,
    geometry_utilities,
    geometry_operations,
    classes_geometry_objects,
    classes_cable_road_computation,
)


# high level functions
def check_if_no_collisions_cable_road(
    this_cable_road: "classes_cable_road_computation.Cable_Road",
):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points elevation plus the support height) which are less than min_height away from the line.
    Returns the cable_road object, and sets the no_collisions property correspondingly

    Args:
        this_cable_road (classes.Cable_Road): The cable_road object to check
        plot_possible_lines (bool): Whether to plot the lines
        pos (list | None): The pos object for plotting

    Returns:
        Nothing, just modifies the cable_road object
    """
    print("checking if no collisions overall line")
    min_height = 3

    # exit the process if we have unrealistically low rope length
    if this_cable_road.c_rope_length < 5:
        this_cable_road.no_collisions = False

    # 1. Test if the CR touches the ground in its loaded state
    this_cable_road.calculate_cr_deflections(loaded=True)

    lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

    # check if the line is above the ground and set it to false if we have a collision
    this_cable_road.no_collisions = lowest_point_height > min_height


def check_if_support_withstands_tension(
    current_segment: "classes_cable_road_computation.SupportedSegment",
    next_segment: "classes_cable_road_computation.SupportedSegment",
) -> bool:
    """
    This function calculates the exerted force on a support tree, based on the tension in a loaded cable road
    and the angle between it and an empty cable road.
    The calculation uses trigonometry and the sine function to determine the force on the support.
    The maximum force that the support can bear is then determined using a Euler buckling calculation.
    The function returns True if the support can handle more force than is being exerted on it, and False otherwise.

    Args:
        diameter_at_height (float): The diameter of the support tree at the height of the support
        attached_at_height (int): The height at which the support is attached to the cable road
        left_cable_road (classes.Cable_Road): The cable road left of the support
        right_cable_road (classes.Cable_Road): The cable road right of the support
    Returns:
        bool: True if the support can handle more force than is being exerted on it, and False otherwise.

    """

    print("checking if support withstands tension")
    scaling_factor = 10000

    # fig, (ax) = plt.subplots(1, 1, figsize=(9, 6))
    ### Calculate the force on the support for the both cable roads
    force_on_support_left = compute_tension_loaded_vs_unloaded_cableroad(
        current_segment.cable_road,
        next_segment.cable_road,
        scaling_factor,
        reverse_direction=False,
    )
    force_on_support_right = compute_tension_loaded_vs_unloaded_cableroad(
        next_segment.cable_road,
        current_segment.cable_road,
        scaling_factor,
        reverse_direction=True,
    )

    print("force on support left", force_on_support_left)
    print("force on support right", force_on_support_right)
    print(
        "max supported force",
        current_segment.end_support.max_supported_force_at_attachment_height,
    )
    # return true if the support can bear more than the exerted force
    return current_segment.end_support.max_supported_force_at_attachment_height > max(
        force_on_support_left, force_on_support_right
    )


def get_line_3d_from_cr_startpoint_to_centroid(
    cable_road: "classes_cable_road_computation.Cable_Road",
    move_towards_start_point: bool,
    sloped: bool,
    index: int,
) -> classes_geometry_objects.LineString_3D:
    """
    Compute the centroid of the cable road and the line that connects the centroid to the start point

    Args:
        cable_road (classes.Cable_Road): The cable road object
        xz_start_point (Point): The xz start point of the cable road
        move_towards_start_point (bool): Whether to move left or right
        sloped (bool): Whether to use the sloped or unloaded line
        index (int): The index of the point along the line
    Returns:
        LineString: The line that connects the centroid to the start point
    """

    # get the 3d start point
    start_point_3d = (
        cable_road.end_support.xyz_location
        if move_towards_start_point
        else cable_road.start_support.xyz_location
    )

    # start from the back of the array if we move towards the start point
    index_swap = -1 if move_towards_start_point else 1

    # get the height by selecting a point along the road
    if sloped:
        centroid_height = cable_road.absolute_loaded_line_height[index * index_swap]
    else:
        centroid_height = cable_road.absolute_unloaded_line_height[index * index_swap]

    # get the xy point along the line
    xy_point_along_line = cable_road.points_along_line[index * index_swap]

    # construct a 3d point along the line with the given height
    centroid_xyz = classes_geometry_objects.Point_3D(
        xy_point_along_line.x, xy_point_along_line.y, centroid_height
    )

    return classes_geometry_objects.LineString_3D(start_point_3d, centroid_xyz)


def compute_resulting_force_on_cable(
    straight_line: classes_geometry_objects.LineString_3D,
    sloped_line: classes_geometry_objects.LineString_3D,
    tension: float,
    scaling_factor: int,
) -> float:
    """
    This function calculates the force on a support tree, based on the tension in a loaded cable road by interpolating
    the force along the straight line and the sloped line and calculating the distance between them.

    Args:
        straight_line (LineString): The straight line from the start point to the centroid
        sloped_line (LineString): The sloped line from the start point to the centroid
        tension (float): The tension in the cable road
        scaling_factor (int): The scaling factor to convert the distance to a force
    Returns:
        float: The force on the support tree
    """

    # interpolate the force along both lines
    force_applied_straight = straight_line.interpolate(tension)
    force_applied_sloped = sloped_line.interpolate(tension)

    # get the distance between both, which represents the force on the cable
    return force_applied_straight.distance(force_applied_sloped) * scaling_factor


def compute_tension_loaded_vs_unloaded_cableroad(
    loaded_cable_road: "classes_cable_road_computation.Cable_Road",
    unloaded_cable_road: "classes_cable_road_computation.Cable_Road",
    scaling_factor: int,
    reverse_direction: bool = False,
    fig: go.Figure = None,
) -> float:
    """
    This function calculates the force on a support tree, based on the tension in a loaded cable road.
    First we get the centroid of the CR, then we calculate the angle between the centroid and the end point.
    Then we interpolate these lines with the tension in the CR.
    Finally, we get the force on the cable road by the distance between the interpolated points.

    The first CR is interpreted as the loaded one, the second one is the unloaded one

    Args:
        loaded_cable_road (classes.Cable_Road): The loaded cable road
        unloaded_cable_road (classes.Cable_Road): The unloaded cable road
        center_point_xz (Point): The central support of the cable road
        scaling_factor (int): The scaling factor to convert the distance to a force
        return_lines (bool): Whether to return the lines or not

    Returns:
        float: The force on the support in Newton, scaled back
    """
    # get the tension that we want to apply on the CR
    tension = (
        loaded_cable_road.s_current_tension / scaling_factor
    )  # scaling to dekanewton

    # we construct this so that the loaded CR is always left and the unloaded always right
    # construct to xz points at the middle of the CR
    loaded_index = len(loaded_cable_road.points_along_line) // 2
    unloaded_index = len(unloaded_cable_road.points_along_line) // 2

    # get the centroid, lines and angles of the two CRs, once tensioned, once empty
    loaded_line_sp_centroid = get_line_3d_from_cr_startpoint_to_centroid(
        loaded_cable_road,
        move_towards_start_point=not reverse_direction,
        sloped=True,
        index=loaded_index,
    )

    unloaded_line_sp_centroid = get_line_3d_from_cr_startpoint_to_centroid(
        unloaded_cable_road,
        move_towards_start_point=reverse_direction,
        sloped=False,
        index=unloaded_index,
    )

    # get the angle between the loaded and the unloaded cable road
    angle_loaded_unloaded_cr = 180 - geometry_utilities.angle_between_3d_lines(
        unloaded_line_sp_centroid, loaded_line_sp_centroid
    )

    # rotate the loaded cable by this angle to be able to compare the distance
    loaded_line_rotated = geometry_utilities.rotate_3d_line_in_z_direction(
        loaded_line_sp_centroid, angle_loaded_unloaded_cr
    )

    if fig:
        fig.data = []  # reset the figure
        print("Angle between lines", angle_loaded_unloaded_cr)
        loaded_line_x, loaded_line_y, loaded_line_z = plotting_2d.get_x_y_z_points(
            loaded_cable_road
        )
        (
            unloaded_line_x,
            unloaded_line_y,
            unloaded_line_z,
        ) = plotting_2d.get_x_y_z_points(unloaded_cable_road)
        fig.add_trace(
            go.Scatter3d(
                x=loaded_line_x,
                y=loaded_line_y,
                z=loaded_line_z,
                mode="lines",
                line=dict(color="red", width=1),
                name="loaded",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=unloaded_line_x,
                y=unloaded_line_y,
                z=unloaded_line_z,
                mode="lines",
                line=dict(color="blue", width=1),
                name="unloaded",
            )
        )

        linestring_dict = {
            "loaded": loaded_line_sp_centroid,
            "unloaded": unloaded_line_sp_centroid,
            "loaded rotated": loaded_line_rotated,
        }
        for line_name, linestring in linestring_dict.items():
            fig = plotting_2d.plot_Linestring_3D(linestring, fig, line_name)

        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            width=1000,
            height=800,
            title="Relief Map with possible Cable Roads",
        )
        fig.show("notebook_connected")
        print(
            "Resulting force:",
            compute_resulting_force_on_cable(
                loaded_line_sp_centroid, loaded_line_rotated, tension, scaling_factor
            ),
        )
        print("Tension:", tension)
    # get the distance between the rotated line and the unloaded line as per the force-interpolated points
    return compute_resulting_force_on_cable(
        loaded_line_sp_centroid, loaded_line_rotated, tension, scaling_factor
    )


def compute_angle_between_lines(
    line1: LineString, line2: LineString, height_gdf: gpd.GeoDataFrame
) -> float:
    """Computes the angle between two lines.

    Args:
        line1 (LineString): The first line.
        line2 (LineString): The second line.
        height_gdf (GeoDataFrame): The GeoDataFrame containing the height data.

    Returns:
        angle (Float): The angle in degrees
    """
    start_point_xy, end_point_xy = Point(line1.coords[0]), Point(line2.coords[1])

    max_deviation = 0.1
    start_point_xy_height = geometry_operations.fetch_point_elevation(
        start_point_xy, height_gdf, max_deviation
    )
    end_point_xy_height = geometry_operations.fetch_point_elevation(
        end_point_xy, height_gdf, max_deviation
    )

    # piece together the triple from the xy coordinates and the z (height)
    start_point_xyz = (
        start_point_xy.coords[0][0],
        start_point_xy.coords[0][1],
        start_point_xy_height,
    )
    end_point_xyz = (
        end_point_xy.coords[0][0],
        end_point_xy.coords[0][1],
        end_point_xy_height,
    )

    return geometry_utilities.angle_between_3d(start_point_xyz, end_point_xyz)


def compute_angle_between_supports(
    possible_line: LineString,
    height_gdf: gpd.GeoDataFrame,
):
    """Compute the angle between the start and end support of a cable road.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_:  the angle between two points in degrees
    """
    start_point_xy, end_point_xy = Point(possible_line.coords[0]), Point(
        possible_line.coords[1]
    )
    max_deviation = 0.1
    start_point_xy_height = geometry_operations.fetch_point_elevation(
        start_point_xy, height_gdf, max_deviation
    )
    end_point_xy_height = geometry_operations.fetch_point_elevation(
        end_point_xy, height_gdf, max_deviation
    )

    # piece together the triple from the xy coordinates and the z (height)
    start_point_xyz = (
        start_point_xy.coords[0][0],
        start_point_xy.coords[0][1],
        start_point_xy_height,
    )
    end_point_xyz = (
        end_point_xy.coords[0][0],
        end_point_xy.coords[0][1],
        end_point_xy_height,
    )

    # a line from the start point to the end point
    vector_start_end = np.subtract(start_point_xyz, end_point_xyz)
    # a line from the start point to the end point on the same height as the start point
    vector_line_floor = np.subtract(
        start_point_xyz, (end_point_xyz[0], end_point_xyz[1], start_point_xyz[2])
    )
    # compute the angle between the line and the x-axis
    return geometry_utilities.angle_between_3d(vector_start_end, vector_line_floor)


def parallelverschiebung(force: float, angle: float) -> float:
    """Compute the force that is exerted on the tower and anchor depending on the angle between the tower and the anchor.

    Args:
        force (float): The force that is exerted on the tower.
        angle (float): The angle between the tower and the anchor.
    Returns:
        resulting_force (float): The force that is exerted on the tower and anchor.
    """

    return (force * np.sin(np.deg2rad(0.5 * angle))) * 2


def check_if_tower_and_anchor_trees_hold(
    this_cable_road: "classes_cable_road_computation.Cable_Road",
    max_holding_force: list[float],
    anchor_triplets: list,
    height_gdf: gpd.GeoDataFrame,
    fig: go.Figure = None,
) -> bool:
    """Check if the tower and its anchors support the exerted forces. First we generate a sideways view of the configuration,
    and then check for every anchor triplet what force is applied to the tower and anchor.
    If both factors are within allowable limits, set the successful anchor triplet to the cable road and exit, else try with the rest of the triplets.

    Args:
        this_cable_road (classes.Cable_Road): The cable road that is checked.
        max_holding_force (list[float]): The maximum force that the tower and anchor can support.
        anchor_triplets (list): The anchor triplets that are checked.
        height_gdf (gpd.GeoDataFrame): The height gdf that is used to fetch the height of the tower and anchor.
    Returns:
        anchors_hold (bool): True if the anchors hold, False if not.

    """
    print("checking if tower and anchor trees hold")
    # get force at last support
    exerted_force = this_cable_road.s_current_tension
    maximum_tower_force = 300000
    scaling_factor = 10000  # unit length = 1m = 10kn of tension

    # start point of the cr tower
    tower_xz_point = this_cable_road.start_support.xyz_location

    # the S_Max point of the tower, by shifting the tower point by the exerted force to the left and then getting the sloped height
    index = min(
        int(exerted_force // scaling_factor), len(this_cable_road.points_along_line) - 1
    )
    loaded_cr_interpolated_tension_point = classes_geometry_objects.Point_3D(
        this_cable_road.points_along_line[index].x,
        this_cable_road.points_along_line[index].y,
        this_cable_road.absolute_loaded_line_height[index],
    )

    for index in range(len(anchor_triplets)):
        # set the central anchor point as line
        this_anchor_line = anchor_triplets[index][0]
        anchor_start_point_xy = Point(this_anchor_line.coords[0])

        # construct the anchor tangent from the anchor point to the tower
        anchor_point_height = geometry_operations.fetch_point_elevation(
            anchor_start_point_xy, height_gdf, 1
        )
        anchor_start_point_xyz = classes_geometry_objects.Point_3D(
            anchor_start_point_xy.x, anchor_start_point_xy.y, anchor_point_height
        )
        anchor_start_point_distance = (
            this_cable_road.start_support.xyz_location.distance(anchor_start_point_xyz)
        )
        print("exerted force", exerted_force)

        force_on_anchor, force_on_tower = construct_tower_force_parallelogram(
            this_cable_road.start_support.xyz_location,
            loaded_cr_interpolated_tension_point,
            anchor_start_point_xyz,
            scaling_factor,
            fig,
        )

        if force_on_tower < maximum_tower_force:
            if force_on_anchor < max_holding_force[index]:
                # do I need to build up a list?
                this_cable_road.anchor_triplets = anchor_triplets[index]
                print("found anchor tree that holds")
                return True
            else:
                print("did not find anchor tree that holds - iterating")

    return False


from varname import nameof


def construct_tower_force_parallelogram(
    tower_point: classes_geometry_objects.Point_3D,
    s_max_point: classes_geometry_objects.Point_3D,
    s_a_point_real: classes_geometry_objects.Point_3D,
    scaling_factor: int,
    fig: go.Figure = None,
) -> tuple[float, float]:
    """Constructs a parallelogram with the anchor point as its base, the force on the anchor as its height and the angle between the anchor tangent and the cr tangent as its angle.
    Based on Stampfer Forstmaschinen und Holzbringung Heft P. 17

    Args:
        tower_point (_type_): the central sideways-viewed top of the anchor
        s_max_point (_type_): the sloped point of the cable road with the force applied in xz view
        s_a_point_real (_type_): the real anchor point (not the point with the force applied)
        force (float): the force applied to the cable road
        scaling_factor (int): the scaling factor to convert the force to a distance
        ax (plt.Axes, optional): the axis to plot the parallelogram on. Defaults to None.

    Returns:
        float: the force applied to the anchor
        float: the force applied the tower
    """
    s_max_to_anchor_dist = s_max_point.distance(tower_point)
    s_max_to_anchor_height = tower_point.z - s_max_point.z

    tower_s_max_x_point = classes_geometry_objects.Point_3D(
        tower_point.x, tower_point.y, s_max_point.z
    )
    tower_s_max_x_point_distance = s_max_point.distance(tower_s_max_x_point)

    # get the point along the line which is the force distance away from the tower point
    s_a_point_interpolated = classes_geometry_objects.LineString_3D(
        tower_point, s_a_point_real
    ).interpolate(tower_s_max_x_point_distance)

    # # update the tower s max x point with the height of the s_a point
    # tower_s_max_x_point = Point(
    #     tower_point.coords[0][0],
    #     s_max_point.coords[0][1]
    #     + (s_a_point_interpolated.coords[0][1] - tower_point.coords[0][1]),
    # )

    # get the z distance from anchor to sa point
    s_a_interpolated_length = s_a_point_interpolated.distance(tower_point)

    # and the central point along the tower xz line with the coordinates of sa
    tower_s_a_radius = classes_geometry_objects.Point_3D(
        tower_point.x,
        tower_point.y,
        tower_point.z - s_a_interpolated_length,
    )

    tower_s_max_radius = classes_geometry_objects.Point_3D(
        tower_point.x,
        tower_point.y,
        tower_point.z - s_max_to_anchor_height,
    )

    # shifting s_max z down by s_a distance to get a_3
    a_3_point = classes_geometry_objects.Point_3D(
        s_max_point.x, s_max_point.y, tower_s_max_radius.z - s_max_to_anchor_height
    )

    # shifting s_a down by s_a_distance
    a_4_point = classes_geometry_objects.Point_3D(
        s_a_point_interpolated.x,
        s_a_point_interpolated.y,
        s_a_point_interpolated.z - s_a_interpolated_length,
    )

    # z distance of anchor to a_4
    z_distance_anchor_to_a_3 = tower_point.z - a_3_point.z
    z_distance_anchor_to_a_4 = tower_point.z - a_4_point.z
    z_distance_anchor_a5 = z_distance_anchor_to_a_3 + z_distance_anchor_to_a_4
    # and now shifting the tower point down by this distance
    a_5_point = classes_geometry_objects.Point_3D(
        tower_point.x,
        tower_point.y,
        tower_point.z - z_distance_anchor_a5,
    )

    # determine the force on the anchor
    force_on_anchor = s_a_point_interpolated.distance(tower_point) * scaling_factor
    # now resulting force = distance from anchor to a_5*scaling factor
    force_on_tower = z_distance_anchor_a5 * scaling_factor

    if fig:
        point_dict = {
            "tower_point": tower_point,
            "s_max_point": s_max_point,
            "s_a_point_real": s_a_point_real,
            "tower_s_max_x_point": tower_s_max_x_point,
            "s_a_point_interpolated": s_a_point_interpolated,
            "tower_s_a_radius": tower_s_a_radius,
            "tower_s_max_radius": tower_s_max_radius,
            "a_3_point": a_3_point,
            "a_4_point": a_4_point,
            "a_5_point": a_5_point,
        }

        for name, point in point_dict.items():
            fig.add_trace(
                go.Scatter3d(
                    x=[point.x],
                    y=[point.y],
                    z=[point.z],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    text=name,
                )
            )
            print(name, point.xyz)

        fig.update_traces(marker={"size": 3})
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            width=1000,
            height=800,
            title="Relief Map with possible Cable Roads",
        )

        print("force on anchor", force_on_anchor)
        print("force on tower", force_on_tower)

        fig.show("notebook_connected")

    return force_on_anchor, force_on_tower


def euler_knicklast(middle_diameter: float, height_of_attachment: float) -> float:
    """Calculates the euler case 2 knicklast of a tree
    Args:
        middle_diameter (float): the diameter at the middle of the tree in cm
        height_of_attachment (float): the height of the attachment in meters
    Returns:
        float: the force the tree can withstand in Newton
    """
    if not height_of_attachment:
        height_of_attachment = 1

    height_of_attachment = height_of_attachment * 10  # convert to cm

    E_module_wood = 80000  # in N/cm^2
    security_factor = 10

    return ((math.pi**2) * E_module_wood * math.pi * (middle_diameter**4)) / (
        (height_of_attachment**2) * 64 * security_factor
    )
