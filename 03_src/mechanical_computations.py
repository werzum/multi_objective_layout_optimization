import math
from shapely.geometry import LineString, Point, Polygon
import numpy as np
import vispy.scene
import geopandas as gpd

import geometry_utilities, geometry_operations, classes

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
):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points elevation plus the support height) which are less than min_height away from the line.
    Returns the cable_road object, and sets the no_collisions property correspondingly

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    min_height = 3

    # exit the process if we have unrealistically low rope length
    if this_cable_road.c_rope_length < 5:
        this_cable_road.no_collisions = False

    # Remove the Zweifel computation for now and reli on the old-fashioned
    # Zweifel Schritt 1 - length of skyline without load
    # mechanical_computations.calculate_length_unloaded_skyline(this_cable_road)

    # Zweifel Schritt 2 - length of skyline with load
    # mechanical_computations.calculate_length_loaded_skyline(this_cable_road)

    # increase tension by predefined amount
    # this_cable_road.t_v_j_bar_tensile_force_at_center_span = (
    #    this_cable_road.t_v_j_bar_tensile_force_at_center_span + 10
    # )

    # Zweifel Schritt 3 - calculate properties of skyline under load (ie deflection)
    # y_x_deflections = mechanical_computations.calculate_sloped_line_to_floor_distances(this_cable_road)

    # Process of updating the tension and checking if we touch ground and anchors hold
    if not pre_tension:
        while (
            this_cable_road.s_current_tension < this_cable_road.s_max_maximalspannkraft
        ):
            # 1. do the anchors hold? break the loop - this configuration doesnt work
            if not check_if_anchor_trees_hold(
                this_cable_road, max_supported_force, anchor_triplets
            ):
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

    # plot the lines if true
    if (
        plot_possible_lines
        and this_cable_road.floor_points
        and this_cable_road.floor_height_below_line_points
        and this_cable_road.anchors_hold
        and this_cable_road.no_collisions
    ):
        plotting.plot_lines(
            this_cable_road.floor_points,
            this_cable_road.floor_height_below_line_points,
            this_cable_road.sloped_line_to_floor_distances,
            view,
            pos,
        )


def check_if_no_collisions_segments(this_cable_road: classes.Cable_Road):
    # Process of updating the tension and checking if we touch ground and anchors hold
    this_cable_road.no_collisions = False

    # 1. calculate current deflections with a given tension
    y_x_deflections = np.asarray(
        [
            lastdurchhang_at_point(
                this_cable_road, point, this_cable_road.s_current_tension
            )
            for point in this_cable_road.points_along_line
        ]
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

    # compute the exerted force with trigonometr"""  """
    # gegenkathete = hpotenuse*sin(angle/2)
    # doppeltes Dreieck - gegenkathete*2
    force_on_support = (current_tension * math.sin(angle_tangents / 2)) * 2

    # get the supported force of the support tree
    # TBD this can also be done in advance
    max_force_of_support = euler_knicklast(diameter_at_height, attached_at_height)

    # return true if the support can bear more than the exerted force
    return max_force_of_support > force_on_support


def initialize_line_tension(this_cable_road: classes.Cable_Road, current_supports: int):
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
            lastdurchhang_at_point(
                this_cable_road, point, this_cable_road.s_current_tension
            )
            for point in this_cable_road.points_along_line
        ]
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


def check_if_anchor_trees_hold(
    this_cable_road: classes.Cable_Road,
    max_supported_force: list[float],
    anchor_triplets: list,
) -> bool:
    # get force at last support
    exerted_force = this_cable_road.s_current_tension
    # this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support
    # todo Parallelverschiebung to get actual force
    force_on_anchor = exerted_force / 10  # for now

    # check if the supported tension is greater than the exerted force
    sufficient_anchors = [
        anchor_triplets[i]
        for i in range(len(anchor_triplets))
        if max_supported_force[i] > force_on_anchor
    ]

    if sufficient_anchors:
        this_cable_road.anchor_triplets = sufficient_anchors
        return True
    else:
        return False


# Lower level functions
def calculate_length_unloaded_skyline(this_cable_road):
    # calculate basic length
    this_cable_road.z_mi_height_center_span = (
        this_cable_road.start_point_height + this_cable_road.end_point_height
    ) / 2  # need to adjust this so that first point is reference with z=0)
    this_cable_road.z_mi_height_support = this_cable_road.end_point_height

    # tension at support and center span, 4.6 and 4.7
    this_cable_road.t_i_bar_tensile_force_at_support = (
        this_cable_road.current_tension
        + this_cable_road.z_mi_height_support
        * this_cable_road.q_s_self_weight_center_span
    )

    this_cable_road.t_i_bar_tensile_force_center_span = (
        this_cable_road.current_tension
        + this_cable_road.z_mi_height_center_span
        * this_cable_road.q_s_self_weight_center_span
    )

    h_i_bar_horizontal_tensile_force = (
        this_cable_road.t_i_bar_tensile_force_center_span
        * (this_cable_road.b_length_whole_section / this_cable_road.c_rope_length)
    )

    # 4.10 overlength unloaded skyline
    this_cable_road.delta_s_overlength = (
        (this_cable_road.b_length_whole_section**4)
        * (this_cable_road.q_s_self_weight_center_span**2)
    ) / (24 * this_cable_road.c_rope_length * (h_i_bar_horizontal_tensile_force**2))

    # 4.12 total length of unloaded skyline
    this_cable_road.u_l_total_length = (
        this_cable_road.c_rope_length + this_cable_road.delta_s_overlength
    )


def calculate_length_loaded_skyline(this_cable_road):
    this_cable_road.t_v_j_bar_tensile_force_at_center_span = this_cable_road.tension + (
        this_cable_road.z_mi_height_center_span
        * this_cable_road.q_s_self_weight_center_span
    )
    # deflection as per 4.14
    this_cable_road.y_mi_deflection_at_center_span = (
        this_cable_road.c_rope_length
        / 4
        * this_cable_road.t_v_j_bar_tensile_force_at_center_span
    ) * (
        this_cable_road.q_load
        + (
            this_cable_road.c_rope_length
            * this_cable_road.q_s_self_weight_center_span
            / 2
        )
    )
    # overlength of chords 4.20
    c_delta_chord_length = (
        (2 * this_cable_road.b_length_whole_section**2)
        / (this_cable_road.c_rope_length**3)
    ) * this_cable_road.y_mi_deflection_at_center_span**2
    # and span of chords 4.23
    s_delta_span = (
        (this_cable_road.b_length_whole_section**2)
        * this_cable_road.c_rope_length
        * (this_cable_road.q_s_self_weight_center_span**2)
        / 96
        * this_cable_road.t_v_j_bar_tensile_force_at_center_span
    )

    # sum the different deltas together for overall length of loaded skyline 4.27
    this_cable_road.u_vj_length_loaded_skyline = (
        this_cable_road.c_rope_length + c_delta_chord_length + s_delta_span
    )


def horizontal_force_at_point(this_cable_road, point):
    # extract the x-coords from the point? TBD
    x = point.coords[0]
    # 4.37
    horizontal_force_at_x = (
        this_cable_road.h_mj_horizontal_force_under_load_at_center_span
        * math.sqrt(
            1
            - (
                1
                - (
                    this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support
                    / this_cable_road.h_mj_horizontal_force_under_load_at_center_span
                )
                ** 2
            )
            * (
                1
                - (
                    2
                    * (x / this_cable_road.this_cable_road.b_length_whole_section) ** 2
                )
            )
        )
    )

    return horizontal_force_at_x


def deflection_by_force_and_position(this_cable_road, point, force_at_point):
    x = point.coords[0]
    # 4.36 deflection
    y_x_deflection_at_x = (
        this_cable_road.y_mi_deflection_at_center_span
        * (
            this_cable_road.h_mj_horizontal_force_under_load_at_center_span
            / force_at_point
        )
        * (1 - (1 - (2 * x / this_cable_road.b_length_whole_section) ** 2))
    )

    return y_x_deflection_at_x


# def calculate_sloped_line_to_floor_distances(this_cable_road):
#     """Calculate array of deflections for each point in the skyline according to overlength

#     Args:
#         this_cable_road (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     # horizontal components of force as per 4.34 and 4.35
#     this_cable_road.h_mj_horizontal_force_under_load_at_center_span = (
#         this_cable_road.b_length_whole_section / this_cable_road.c_rope_length
#     ) * this_cable_road.t_v_j_bar_tensile_force_at_center_span

#     this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support = (
#         this_cable_road.b_length_whole_section / this_cable_road.c_rope_length
#     ) * this_cable_road.t_i_bar_tensile_force_at_support

#     # are we getting x right?
#     horizontal_forces = [
#         horizontal_force_at_point(this_cable_road, point)
#         for point in this_cable_road.points_along_line
#     ]
#     # calculate the deflections as per force and position along the CR with 4.36
#     y_x_deflections = [
#         deflection_by_force_and_position(this_cable_road, point, force)
#         for point, force in zip(this_cable_road.points_along_line, horizontal_forces)
#     ]

#     return y_x_deflections


def lastdurchhang_at_point(this_cable_road, point, s_current_tension):
    """
    Calculates the lastdurchhang value at a given point.

    Args:
    point (Point): The point at which the lastdurchhang value is to be calculated.
    start_point (Point): The start point of the section.
    end_point (Point): The end point of the section.
    b_whole_section (float): The length of the whole section.
    H_t_horizontal_force_tragseil (float): The horizontal force of the tragseil.
    q_vertical_force (float): The vertical force.
    c_rope_length (float): The length of the rope.
    q_bar_rope_weight (float): The weight of the rope.
    q_delta_weight_difference_pull_rope_weight (float): The difference in weight between the pull rope and the tragseil.

    Returns:
    float: The lastdurchhang value at the given point.
    """
    H_t_horizontal_force_tragseil = (
        s_current_tension  # improvised value - need to do the parallelverchiebung here
    )
    q_vertical_force = 15000  # improvised value 30kn?
    q_bar_rope_weight = 1.6  # improvised value 2?
    q_delta_weight_difference_pull_rope_weight = 0.6  # improvised value
    # compute distances and create the corresponding points

    b1_section_1 = (
        this_cable_road.start_point.distance(point) + 0.1
    )  # added a little padding to prevent div by zero
    b2_section_2 = this_cable_road.end_point.distance(point) + 0.1

    lastdurchhang = (
        b1_section_1
        * b2_section_2
        / (this_cable_road.b_length_whole_section * H_t_horizontal_force_tragseil)
    ) * (
        q_vertical_force
        + (this_cable_road.c_rope_length * q_bar_rope_weight / 2)
        + (
            this_cable_road.c_rope_length
            * q_delta_weight_difference_pull_rope_weight
            / (4 * this_cable_road.b_length_whole_section)
        )
        * (b2_section_2 - b1_section_1)
    )
    return lastdurchhang


def euler_knicklast(tree_diameter, height_of_attachment):
    E_module_wood = 80000
    securit_factor = 3
    withstood_force = (math.pi**2 * E_module_wood * math.pi * tree_diameter**4) / (
        height_of_attachment**2 * 64 * securit_factor
    )

    return withstood_force
