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
    pos: list | None,
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

    # 1. Test if the CR touches the ground
    calculate_sloped_line_to_floor_distances(this_cable_road)

    lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

    # check if the line is above the ground and set it to false if we have a collision
    this_cable_road.no_collisions = lowest_point_height > min_height

    # plot the lines if we have a successful candidate
    if (
        plot_possible_lines
        and this_cable_road.floor_points
        and this_cable_road.floor_height_below_line_points
        and this_cable_road.anchors_hold
        and this_cable_road.no_collisions
        and pos
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
    max_supported_force: float,
    left_cable_road: classes.Cable_Road,
    right_cable_road: classes.Cable_Road,
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
    scaling_factor = 10000
    ### Calculate the force on the support for the both cable roads
    force_on_support_left = compute_tension_sloped_vs_empty_cableroad(
        left_cable_road, scaling_factor, return_points=False
    )
    force_on_support_right = compute_tension_sloped_vs_empty_cableroad(
        right_cable_road, scaling_factor, return_points=False
    )

    print("forces on lr support", force_on_support_left, force_on_support_right)
    # return true if the support can bear more than the exerted force
    return max_supported_force > max(force_on_support_left, force_on_support_right)


def compute_tension_sloped_vs_empty_cableroad(
    cable_road: classes.Cable_Road,
    scaling_factor: int,
    ax: plt.Axes = None,
    return_points: bool = False,
) -> float | tuple[float, Point, Point]:
    """
    This function calculates the force on a support tree, based on the tension in a loaded cable road.
    First we get the centroid of the CR, then we calculate the angle between the centroid and the end point.
    Then we interpolate these lines with the tension in the CR.
    Finally, we get the force on the cable road by the distance between the interpolated points.

    Args:
        cable_road (classes.Cable_Road): The cable road for which we want to calculate the force on the support
        scaling_factor (int): The scaling factor for the tension in the CR
        ax (plt.Axes): The axes on which we want to plot the lines
        return_points (bool): If we want to return the points on the lines
    Returns:
        float: The force on the support in Newton, scaled back
        Point: The interpolated point on the straight line in xz view
        Point: The interpolated point on the sloped line in xz view
    """

    # get the tension that we want to apply on the CR
    tension = cable_road.s_current_tension // scaling_factor  # scaling to dekanewton

    # our start point
    start_point = Point(
        [
            cable_road.start_point.coords[0][0],
            cable_road.start_point_height,
        ]
    )

    # construct to xz points at the middle of the CR
    index = len(cable_road.points_along_line) // 2
    centroid_straight_height = (
        cable_road.floor_height_below_line_points[index]
        + cable_road.line_to_floor_distances[index]
    )
    # shift the x coordinate by half the length of the CR to get the middle
    centroid_x_sideways = (
        cable_road.start_point.coords[0][0] - cable_road.c_rope_length // 2
    )
    centroid_straight = Point([centroid_x_sideways, centroid_straight_height])

    centroid_sloped_height = (
        cable_road.floor_height_below_line_points[index]
        + cable_road.sloped_line_to_floor_distances[index]
    )

    centroid_sloped = Point([centroid_x_sideways, centroid_sloped_height])

    # get the angle between them
    line_sp_centroid_straight = LineString([start_point, centroid_straight])
    line_sp_centroid_sloped = LineString([start_point, centroid_sloped])
    angle_centroids = geometry_utilities.angle_between(
        line_sp_centroid_straight, line_sp_centroid_sloped
    )

    # interpolate the force along both lines
    force_applied_straight = line_sp_centroid_straight.interpolate(tension)
    force_applied_sloped = line_sp_centroid_sloped.interpolate(tension)

    # get the distance between both, which represents the force on the cable
    force_on_cable = (
        force_applied_straight.distance(force_applied_sloped) * scaling_factor
    )

    print("angle centroids", angle_centroids)
    print("force on cable", force_on_cable)

    if ax:
        ax.plot(*start_point.xy, "o", color="black")
        ax.plot(*centroid_straight.xy, "o", color="green")
        ax.plot(*centroid_sloped.xy, "o", color="red")
        ax.plot(*force_applied_straight.xy, "o", color="blue")
        ax.plot(*force_applied_sloped.xy, "o", color="blue")

        for lines in [
            [start_point, centroid_straight],
            [start_point, centroid_sloped],
        ]:
            ax.plot(*LineString(lines).xy, color="black")

        ax.set_xlim(-120, 0)
        ax.set_ylim(-60, 20)

        ax.annotate(
            "Start point",
            xy=start_point.coords[0],
            xytext=(-3, -15),
            textcoords="offset points",
        )
        ax.annotate(
            "Center point unloaded",
            xy=centroid_straight.coords[0],
            xytext=(-100, 10),
            textcoords="offset points",
        )
        ax.annotate(
            "Center point loaded",
            xy=centroid_sloped.coords[0],
            xytext=(0, -15),
            textcoords="offset points",
        )

    if return_points:
        return force_on_cable, force_applied_straight, force_applied_sloped
    else:
        return force_on_cable


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


def initialize_line_tension(
    this_cable_road: classes.Cable_Road, current_supports: int, pre_tension: int = 0
):
    print("initialize_line_tension")
    # set tension of the cable_road
    s_br_mindestbruchlast = 170000  # in newton
    this_cable_road.s_max_maximalspannkraft = s_br_mindestbruchlast / 2
    if pre_tension:
        this_cable_road.s_current_tension = pre_tension
    else:
        this_cable_road.s_current_tension = this_cable_road.s_max_maximalspannkraft
    # this_cable_road.s_current_tension = this_cable_road.s_max_maximalspannkraft * (
    #     current_supports + 1 / (current_supports + 2)
    # )


def calculate_sloped_line_to_floor_distances(this_cable_road: classes.Cable_Road):
    """Calculate the distances between the line and the floor based on pointwise deflections.

    Args:
        this_cable_road (classes.Cable_Road): This cable road.

    Returns: Nothing, just updates the cable road object.
    """
    # Calculate current deflections with a given tension
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
    # resulting_force = force * math.sin(0.5 * angle)
    # note - the angle is converted to radians for the np.sin function
    resulting_force = (force * np.sin(np.deg2rad(0.5 * angle))) * 2
    # print(resulting_force)
    return resulting_force


def check_if_tower_and_anchor_trees_hold(
    this_cable_road: classes.Cable_Road,
    max_holding_force: list[float],
    anchor_triplets: list,
    height_gdf: gpd.GeoDataFrame,
    ax: plt.Axes = None,
    ax2: plt.Axes = None,
    ax3: plt.Axes = None,
) -> bool:
    """Check if the tower and its anchors support the exerted forces. First we generate a sideways view of the configuration,
    and then check for every anchor triplet what force is applied to the tower and anchor.
    If both factors are within allowable limits, set the successful anchor triplet to the cable road and exit, else try with the rest of the triplets.

    Args:
        this_cable_road (classes.Cable_Road): The cable road that is checked.
        max_holding_force (list[float]): The maximum force that the tower and anchor can support.
        anchor_triplets (list): The anchor triplets that are checked.
        height_gdf (gpd.GeoDataFrame): The height gdf that is used to fetch the height of the tower and anchor.
        ax (plt.Axes, optional): The axes that is used to plot the sideways view. Defaults to None.
        ax2 (plt.Axes, optional): The axes that is used to plot the force on the tower and anchor. Defaults to None.
        ax3 (plt.Axes, optional): The axes that is used to plot the force on the tower and anchor. Defaults to None.
    Returns:
        anchors_hold (bool): True if the anchors hold, False if not.

    """
    # get force at last support
    exerted_force = this_cable_road.s_current_tension
    maximum_tower_force = 200000
    scaling_factor = 10000  # unit length = 1m = 10kn of tension

    # do the parallelverschiebung to get acting force on the tower (s_max)
    (
        force_on_vertical,
        angle_point_xz,
        angle_point_sloped_xz,
    ) = compute_tension_sloped_vs_empty_cableroad(
        this_cable_road, scaling_factor, return_points=True
    )

    # start point of the cr tower
    tower_xz_point = Point(
        [this_cable_road.start_point.coords[0][0], this_cable_road.start_point_height]
    )

    cr_loaded_tangent = LineString([angle_point_sloped_xz, tower_xz_point])

    if ax:
        ax.clear()
        ax.set_ylim(-60, 20)
        ax.set_xlim(-100, 20)
        ax.plot(*cr_loaded_tangent.xy, color="red")

    for index in range(len(anchor_triplets)):
        # need to figure out which anchor is the right one (ie the central one)
        this_anchor_line = anchor_triplets[index][0]
        anchor_start_point = Point(this_anchor_line.coords[0])

        # construct the anchor tangent
        anchor_point_height = geometry_operations.fetch_point_elevation(
            anchor_start_point, height_gdf, 1
        )
        anchor_start_point_distance = this_cable_road.start_point.distance(
            anchor_start_point
        )

        # anchor point on the xz plane
        anchor_xz_point = Point(
            [
                this_cable_road.start_point.coords[0][0] + anchor_start_point_distance,
                anchor_point_height,
            ]
        )

        force_on_anchor, force_on_tower = construct_tower_force_parallelogram(
            tower_xz_point,
            angle_point_sloped_xz,
            anchor_xz_point,
            this_cable_road.s_current_tension,
            scaling_factor,
            angle_point_xz,
            angle_point_sloped_xz,
            this_cable_road.floor_points,
            ax=ax3,
        )

        if ax:
            ax2.clear()
            ax2_container = ax2.bar(
                ["Exerted Force", "Force on Tower", "Force on Support"],
                [exerted_force, force_on_tower, force_on_anchor],
            )

            ax2.bar_label(ax2_container)
            ax2.set_ylim(0, 150000)

        print("force on anchor", force_on_anchor)
        print("force on twoer", force_on_tower)
        print("max supported force by anchor", max_holding_force[index])
        print(maximum_tower_force)
        if force_on_tower < maximum_tower_force:
            if force_on_anchor < max_holding_force[index]:
                # do I need to build up a list?
                this_cable_road.anchor_triplets = anchor_triplets[index]
                return True

    return False


def construct_tower_force_parallelogram(
    tower_xz_point: Point,
    s_max_point: Point,
    s_a_point_real: Point,
    force: float,
    scaling_factor: int,
    angle_point_xz: Point,
    angle_point_sloped_xz: Point,
    floor_points: list = None,
    ax: plt.Axes = None,
) -> tuple[float, float]:
    """Constructs a parallelogram with the anchor point as its base, the force on the anchor as its height and the angle between the anchor tangent and the cr tangent as its angle.
    Based on Stampfer Forstmaschinen und Holzbringung Heft P. 17

    Args:
        tower_xz_point (_type_): the central sideways-viewed top of the anchor
        s_max_point (_type_): the sloped point of the cable road with the force applied in xz view
        s_a_point_real (_type_): the real anchor point (not the point with the force applied)
        force (float): the force applied to the cable road
        scaling_factor (int): the scaling factor to convert the force to a distance
        ax (plt.Axes, optional): the axis to plot the parallelogram on. Defaults to None.

    Returns:
        float: the force applied to the anchor
        float: the force applied the tower
    """
    # x distance from s_max to anchor
    s_max_to_anchor = abs(s_max_point.coords[0][0] - tower_xz_point.coords[0][0])

    # shifting anchor by this distance to the right to get s_a
    tower_anchor_line = LineString([tower_xz_point, s_a_point_real])
    s_a_point_force = tower_anchor_line.interpolate(force / scaling_factor)

    # shifting s_max z down by s_a distance to get a_3
    s_max_length = s_a_point_force.distance(tower_xz_point)
    a_3_point = Point(
        [s_max_point.coords[0][0], s_max_point.coords[0][1] - s_max_length]
    )

    # shifting s_a down by s_a_distance
    s_a_length = s_a_point_force.distance(tower_xz_point)
    a_4_point = Point(
        [s_a_point_force.coords[0][0], s_a_point_force.coords[0][1] - s_a_length]
    )

    # z distance of anchor to a_4
    z_distance_anchor_to_a_3 = tower_xz_point.coords[0][1] - a_3_point.coords[0][1]
    z_distance_anchor_to_a_4 = tower_xz_point.coords[0][1] - a_4_point.coords[0][1]
    z_distance_anchor_a5 = z_distance_anchor_to_a_3 + z_distance_anchor_to_a_4
    # and now shifting the tower point down by this distance
    a_5_point = Point(
        [
            tower_xz_point.coords[0][0],
            tower_xz_point.coords[0][1] - z_distance_anchor_a5,
        ]
    )

    force_on_anchor = s_max_to_anchor * scaling_factor
    # now resulting force = distance from anchor to a_5*scaling factor
    force_on_tower = z_distance_anchor_a5 * scaling_factor

    if ax:
        ax.clear()
        ax.set_xlim(-35, -17)
        ax.set_ylim(-5, 12)

        # plot the points
        ax.plot(*s_max_point.xy, "o", color="black")
        ax.plot(*s_a_point_force.xy, "o", color="blue")
        ax.plot(*a_3_point.xy, "o", color="red")
        ax.plot(*a_4_point.xy, "o", color="red")
        ax.plot(*a_5_point.xy, "o", color="blue")

        ax.plot(*angle_point_xz.xy, "o", color="blue")
        ax.plot(*angle_point_sloped_xz.xy)

        for lines in [
            [s_max_point, tower_xz_point],
            [s_a_point_force, tower_xz_point],
            [a_3_point, s_max_point],
            [a_4_point, s_a_point_force],
            [a_5_point, tower_xz_point],
            [a_5_point, a_3_point],
            [a_5_point, a_4_point],
            [tower_xz_point, angle_point_xz],
            [tower_xz_point, angle_point_sloped_xz],
            [
                Point([tower_xz_point.coords[0][0], s_max_point.coords[0][1]]),
                Point([s_max_point.coords[0][0], s_max_point.coords[0][1]]),
            ],  # smax to anchor line
            [
                Point([s_a_point_force.coords[0][0], s_a_point_force.coords[0][1]]),
                Point([tower_xz_point.coords[0][0], s_max_point.coords[0][1]]),
            ],  # sa to anchor line
            [
                Point([a_3_point.coords[0][0], a_3_point.coords[0][1]]),
                Point(
                    [
                        tower_xz_point.coords[0][0],
                        tower_xz_point.coords[0][1] - s_max_length,
                    ]
                ),
            ],  # s3 to anchor with length of smax
            [
                Point([a_4_point.coords[0][0], a_4_point.coords[0][1]]),
                Point([tower_xz_point.coords[0][0], a_4_point.coords[0][1]]),
            ],
        ]:
            ax.plot(*LineString(lines).xy, color="black")

        ax.annotate(
            "Force on Cable",
            s_max_point.coords[0],
            xytext=(3, -15),
            textcoords="offset points",
        )
        ax.annotate(
            "Force on Anchor",
            s_a_point_force.coords[0],
            xytext=(3, -15),
            textcoords="offset points",
        )
        ax.annotate(
            "Force on Tower",
            a_5_point.coords[0],
            xytext=(3, -15),
            textcoords="offset points",
        )
        ax.annotate(
            "Buckling Force left",
            a_3_point.coords[0],
            xytext=(5, -5),
            textcoords="offset points",
        )
        ax.annotate(
            "Buckling Force right",
            a_4_point.coords[0],
            xytext=(3, -15),
            textcoords="offset points",
        )
        ax.annotate(
            "Unloaded Cable",
            angle_point_xz.coords[0],
            xytext=(-90, -5),
            textcoords="offset points",
        )

    return force_on_anchor, force_on_tower


def pestal_load_path(cable_road: classes.Cable_Road, point: Point):
    """Calculates the load path of the cable road based on the pestal method

    Args:
        cable_road (classes.Cable_Road): the cable road
        point (Point): the point to calculate the load path for
    Returns:
        float: the deflection of the cable road along the load path
    """
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


def euler_knicklast(tree_diameter: float, height_of_attachment: float) -> float:
    """Calculates the euler case 2 knicklast of a tree
    Args:
        tree_diameter (float): the diameter of the tree in cm
        height_of_attachment (float): the height of the attachment
    Returns:
        float: the force the tree can withstand
    """
    if height_of_attachment == 0:
        height_of_attachment = 1

    E_module_wood = 11000
    security_factor = 5
    withstood_force = (math.pi**2 * E_module_wood * math.pi * tree_diameter**4) / (
        height_of_attachment**2 * 64 * security_factor
    )

    return withstood_force
