import math
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import vispy.scene
import geopandas as gpd
import matplotlib.pyplot as plt

from src import geometry_utilities, geometry_operations, classes, plotting

# high level functions


def check_if_no_collisions_overall_line(
    this_cable_road: classes.Cable_Road,
    plot_possible_lines: bool,
    view: vispy.scene.ViewBox,
    pos: list,
    current_supports: int,
    anchor_triplets: list,
    max_supported_force: list[float],
    pre_tension: None | float,
    height_gdf: gpd.GeoDataFrame,
):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points elevation plus the support height) which are less than min_height away from the line.
    Returns the cable_road object, and sets the no_collisions property correspondingly

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    print("checking if no collisions overall line")
    min_height = 3

    # exit the process if we have unrealistically low rope length
    if this_cable_road.c_rope_length < 5:
        this_cable_road.no_collisions = False

    # Process of updating the tension and checking if we touch ground and anchors hold
    if not pre_tension:
        while (
            this_cable_road.s_current_tension < this_cable_road.s_max_maximalspannkraft
        ):
            # 1. do the anchors hold? break the loop - this configuration doesnt work
            if check_if_anchor_trees_hold(
                this_cable_road, max_supported_force, anchor_triplets, height_gdf
            ):
                this_cable_road.anchors_hold = True
            else:
                this_cable_road.anchors_hold = False
                break

            calculate_sloped_line_to_floor_distances(this_cable_road)

            lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

            # check if the line is above the ground and set it to false if we have a collision
            if lowest_point_height > min_height:
                # we found no collisions and exit the loop
                this_cable_road.no_collisions = True
                break
            else:
                this_cable_road.no_collisions = False
                # break the loop before we go over the maximalspannkraft
                if (
                    this_cable_road.s_current_tension + 1000
                    < this_cable_road.s_max_maximalspannkraft
                ):
                    this_cable_road.s_current_tension += 1000
                else:
                    break

    # if we skipped over the computation because the tension is pre-set
    if pre_tension:
        calculate_sloped_line_to_floor_distances(this_cable_road)
        lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

        if lowest_point_height > min_height:
            # we found no collisions and exit the loop
            this_cable_road.no_collisions = True
        else:
            this_cable_road.no_collisions = False

    # plot the lines if we have a successful candidate
    if (
        plot_possible_lines
        and this_cable_road.floor_points
        and this_cable_road.floor_height_below_line_points
        and this_cable_road.anchors_hold
        and this_cable_road.no_collisions
    ):
        print("plotting", this_cable_road)
        plotting.plot_lines(this_cable_road, pos)


def check_if_no_collisions_segments(this_cable_road: classes.Cable_Road):
    # Process of updating the tension and checking if we touch ground and anchors hold
    this_cable_road.no_collisions = False

    # 1. calculate current deflections with a given tension
    y_x_deflections = np.asarray(
        [
            pestal_load_path(this_cable_road, point)
            for point in this_cable_road.points_along_line
        ],
        dtype=np.float32,
    )

    #  check the distances between each floor point and the ldh point
    this_cable_road.sloped_line_to_floor_distances = (
        this_cable_road.line_to_floor_distances - y_x_deflections
    )

    lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

    # check if the line is above the ground and set it to false if we have a collision
    if lowest_point_height < this_cable_road.min_height:
        this_cable_road.no_collisions = False
    else:
        # we found no collisions and exit the loop
        this_cable_road.no_collisions = True


def check_if_support_withstands_tension(
    diameter_at_height: float,
    attached_at_height: int,
    loaded_cable_road: classes.Cable_Road,
    empty_cable_road: classes.Cable_Road,
    current_tension: float,
):
    """
    This function calculates the exerted force on a support tree, based on the tension in a loaded cable road and the angle between it and an empty cable road. The calculation uses trigonometry and the sine function to determine the force on the support. The maximum force that the support can bear is then determined using a Euler buckling calculation. The function returns True if the support can handle more force than is being exerted on it, and False otherwise.
    """

    # 1. construct tangents - from the left middle of the loaded cr to its endpoint
    full_tangent = LineString(
        [loaded_cable_road.line.centroid, loaded_cable_road.end_point]
    )
    empt_tangent = LineString(
        [empty_cable_road.start_point, empty_cable_road.line.centroid]
    )

    # get their angles
    angle_tangents = geometry_utilities.angle_between(full_tangent, empt_tangent)

    # compute the exerted force with trigonometry
    # gegenkathete = hpotenuse*sin(angle/2)
    force_on_support = parallelverschiebung(current_tension, angle_tangents)

    # get the supported force of the support tree
    # TBD this can also be done in advance - attached at height+2 to accomodate stütze itself
    max_force_of_support = euler_knicklast(diameter_at_height, attached_at_height + 2)

    # return true if the support can bear more than the exerted force
    return max_force_of_support > force_on_support


def initialize_line_tension(this_cable_road: classes.Cable_Road, current_supports: int):
    print("initialize_line_tension")
    # set tension of the cable_road
    s_br_mindestbruchlast = 170000  # in newton
    this_cable_road.s_max_maximalspannkraft = s_br_mindestbruchlast / 3
    this_cable_road.s_current_tension = this_cable_road.s_max_maximalspannkraft * (
        current_supports + 1 / (current_supports + 2)
    )


def calculate_sloped_line_to_floor_distances(this_cable_road: classes.Cable_Road):
    # 1. calculate current deflections with a given tension
    y_x_deflections = np.asarray(
        [
            pestal_load_path(this_cable_road, point)
            for point in this_cable_road.points_along_line
        ],
        dtype=np.float32,
    )

    #  check the distances between each floor point and the ldh point
    this_cable_road.sloped_line_to_floor_distances = (
        this_cable_road.line_to_floor_distances - y_x_deflections
    )


def compute_angle_between_supports(
    possible_line: LineString, height_gdf: gpd.GeoDataFrame
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

    # and finally compute the angle
    return geometry_utilities.angle_between_3d(start_point_xyz, end_point_xyz)


def parallelverschiebung(force, angle):
    # resulting_force = force * math.sin(0.5 * angle)
    # note - the angle is converted to radians for the np.sin function
    resulting_force = (force * np.sin(np.deg2rad(0.5 * angle))) * 2
    # print(resulting_force)
    return resulting_force


def check_if_anchor_trees_hold(
    this_cable_road: classes.Cable_Road,
    max_supported_force: list[float],
    anchor_triplets: list,
    height_gdf: gpd.GeoDataFrame,
) -> bool:
    # get force at last support
    exerted_force = this_cable_road.s_current_tension

    # get the xz centroid of the cable road based on the x of the centroid and the height of the middle point
    centroid_xz = Point(
        [
            this_cable_road.line.centroid.coords[0][0],
            geometry_operations.fetch_point_elevation(
                this_cable_road.line.centroid, height_gdf, 1
            ),
        ]
    )

    # start point of the cr tower
    start_point_xz = Point(
        [this_cable_road.start_point.coords[0][0], this_cable_road.start_point_height]
    )

    cr_loaded_tangent = LineString([centroid_xz, start_point_xz])

    for index in range(len(anchor_triplets)):
        this_anchor_line = anchor_triplets[index][1]
        anchor_start_point = Point(this_anchor_line.coords[0])

        # 1. construct tangents - from the left middle of the loaded cr to its endpoint
        anchor_xz_point = Point(
            anchor_start_point.coords[0][0],
            geometry_operations.fetch_point_elevation(
                anchor_start_point, height_gdf, 1
            ),
        )

        anchor_tangent = LineString([anchor_xz_point, start_point_xz])
        # get their angles
        angle_tangents = 180 - geometry_utilities.angle_between(
            cr_loaded_tangent, anchor_tangent
        )

        # compute the exerted force with trigonometry
        # gegenkathete = hpotenuse*sin(angle/2)
        force_on_support = parallelverschiebung(exerted_force, angle_tangents)
        force_on_anchor = exerted_force - force_on_support

        if max_supported_force[index] > force_on_anchor:
            # do I need to build up a list?
            this_cable_road.anchor_triplets = anchor_triplets[index]
            return True
    return False


def check_if_anchor_trees_hold_and_plot(
    this_cable_road: classes.Cable_Road,
    max_supported_force: list[float],
    anchor_triplets: list,
    height_gdf: gpd.GeoDataFrame,
    ax: plt.Axes,
    ax2: plt.Axes,
    ax3: plt.Axes,
) -> bool:
    # get force at last support
    exerted_force = this_cable_road.s_current_tension

    # get the xz centroid of the cable road based on the x of the centroid and the height of the middle point
    centroid_xy_distance = this_cable_road.line.centroid.distance(
        this_cable_road.start_point
    )
    # and rotate the centroid at our sideways-x-axis relative to the start point
    centroid_xz = Point(
        [
            this_cable_road.start_point.coords[0][0] - centroid_xy_distance,
            geometry_operations.fetch_point_elevation(
                this_cable_road.line.centroid, height_gdf, 1
            ),
        ]
    )

    # start point of the cr tower
    tower_xz_point = Point(
        [this_cable_road.start_point.coords[0][0], this_cable_road.start_point_height]
    )

    cr_loaded_tangent = LineString([centroid_xz, tower_xz_point])

    ax.clear()
    ax.set_ylim(-60, 20)
    ax.set_xlim(-100, 20)
    ax.plot(*cr_loaded_tangent.xy, color="red")

    for index in [0]:
        this_anchor_line = anchor_triplets[index][1]
        anchor_start_point_xy_distance = tower_xz_point.distance(
            Point(this_anchor_line.coords[0])
        )
        anchor_start_point = Point(this_anchor_line.coords[0])

        # 1. construct tangents - from the left middle of the loaded cr to its endpoint
        anchor_xz_point = Point(
            tower_xz_point.coords[0][0] + anchor_start_point_xy_distance,
            geometry_operations.fetch_point_elevation(
                anchor_start_point, height_gdf, 1
            ),
        )

        anchor_tangent = LineString([anchor_xz_point, tower_xz_point])
        ax.plot(*anchor_tangent.xy)

        # get their angles
        angle_tangents = 180 - geometry_utilities.angle_between(
            cr_loaded_tangent, anchor_tangent
        )

        # compute the exerted force with trigonometry
        # gegenkathete = hpotenuse*sin(angle/2)
        force_on_support = parallelverschiebung(exerted_force, angle_tangents)

        force_on_anchor = construct_tower_force_parallelogram(
            tower_xz_point, exerted_force, angle_tangents, ax=ax3
        )

        ax2.clear()
        ax2_container = ax2.bar(
            ["Exerted Force", "Force on Anchor", "Force on Support"],
            [exerted_force, force_on_anchor, force_on_support],
        )

        ax2.bar_label(ax2_container)
        ax2.set_ylim(0, 10000)
        if max_supported_force[index] > force_on_anchor:
            # do I need to build up a list?
            this_cable_road.anchor_triplets = anchor_triplets[index]
            return True
    return False


def construct_tower_force_parallelogram(
    tower_xz_point: Point,
    exerted_force: float,
    angle_tangents: float,
    ax: plt.Axes = None,
):
    """Constructs a parallelogram with the anchor point as its base, the force on the anchor as its height and the angle between the anchor tangent and the cr tangent as its angle. Based on Stampfer Forstmaschinen und Holzbringung Heft P. 17

    Args:
        tower_xz_point (_type_): the central sideways-viewed top of the anchor
        exerted_force (_type_): _description_
        angle_tangents (_type_): _description_
    """

    # direction of force from point and with angle
    s_max_point = Point(
        [
            tower_xz_point.coords[0][0]
            - exerted_force * np.cos(np.deg2rad(angle_tangents)),
            tower_xz_point.coords[0][1]
            - exerted_force * np.sin(np.deg2rad(angle_tangents)),
        ]
    )
    # x distance from s_max to anchor
    s_max_to_anchor = abs(s_max_point.coords[0][0] - tower_xz_point.coords[0][0])
    # shifting anchor by this distance to the right to get s_a
    s_a_point = Point(
        [
            tower_xz_point.coords[0][0] + s_max_to_anchor,
            tower_xz_point.coords[0][1],
        ]
    )

    # shifting s_max y down by s_max distance to get a_3
    s_max_length = s_max_point.distance(tower_xz_point)
    a_3_point = Point(
        [s_max_point.coords[0][0], s_max_point.coords[0][1] - s_max_length]
    )

    # shifting s_a down by s_a_distance
    s_a_length = s_a_point.distance(tower_xz_point)
    a_4_point = Point([s_a_point.coords[0][0], s_a_point.coords[0][1] - s_a_length])

    # generating s_5 by getting y distance of anchor to a_3
    y_distance_anchor_to_a_3 = abs(tower_xz_point.coords[0][1] - a_3_point.coords[0][1])
    # and shifting a_4 down by this difference
    a_5_point = Point(
        [
            tower_xz_point.coords[0][0],
            a_4_point.coords[0][1] - y_distance_anchor_to_a_3,
        ]
    )

    # now resulting force = distance from anchor to a_5#
    force_on_anchor = tower_xz_point.distance(a_5_point)

    if ax:
        ax.clear()
        ax.set_xlim(-10000, 10000)
        ax.set_ylim(-15000, 1000)

        # plot the points
        ax.plot(*s_max_point.xy, "o", color="black")
        ax.plot(*s_a_point.xy, "o", color="green")
        ax.plot(*a_3_point.xy, "o", color="red")
        ax.plot(*a_4_point.xy, "o", color="red")
        ax.plot(*a_5_point.xy, "o", color="blue")
        # ax.plot(*anchor_xz_point.xy, "o", color="pink")

    return force_on_anchor


def pestal_load_path(cable_road, point):
    # H_t_horizontal_force_tragseil = (Tensile_force_at_center*(horizontal_width/lenght_of_cable_road)
    # os ht the correct force though?

    T_0_basic_tensile_force = cable_road.s_current_tension
    q_s_rope_weight = 1.6
    q_vertical_force = 15000

    h_height_difference = abs(
        cable_road.end_point_height - cable_road.start_point_height
    )

    T_bar_tensile_force_at_center_span = T_0_basic_tensile_force + q_s_rope_weight * (
        (h_height_difference / 2)
    )

    H_t_horizontal_force_tragseil = T_bar_tensile_force_at_center_span * (
        cable_road.b_length_whole_section / cable_road.c_rope_length
    )  # improvised value - need to do the parallelverchiebung here

    x = cable_road.start_point.distance(point)

    y_deflection_at_point = (
        (x * (cable_road.b_length_whole_section - x))
        / (H_t_horizontal_force_tragseil * cable_road.b_length_whole_section)
    ) * (q_vertical_force + ((cable_road.c_rope_length * q_s_rope_weight) / 2))

    return y_deflection_at_point


def euler_knicklast(tree_diameter, height_of_attachment):
    E_module_wood = 80000
    securit_factor = 5
    withstood_force = (math.pi**2 * E_module_wood * math.pi * tree_diameter**4) / (
        height_of_attachment**2 * 64 * securit_factor
    )

    return withstood_force
