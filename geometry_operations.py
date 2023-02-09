from copy import deepcopy
from shapely.geometry import LineString, Point
import numpy as np
import itertools
import pandas as pd
import vispy.scene
from vispy.scene import visuals
import math

import geometry_utilities
import plotting
import classes
import mechanical_computations


def generate_possible_lines(
    road_points,
    target_trees,
    anchor_trees,
    overall_trees,
    slope_line,
    height_gdf,
    plot_possible_lines,
):
    """Compute which lines can be made from road_points to anchor_trees without having an angle greater than max_main_line_slope_deviation
    First, we generate all possible lines between  each point along the road and all head anchors.
    For those which do not deviate more than max_main_line_slope_deviation degrees from the slope line, we compute head anchor support trees along the lines.
    If those are present, we compute triples of tail anchor support trees.
    If those are present, valid configurations are appended to the respective lists.

    Args:
        road_points (_type_): _description_
        target_trees (_type_): _description_
        anchor_trees (_type_): _description_
        slope_line (_type_): _description_
        max_main_line_slope_deviation (_type_): How much the central of three lines can deviate from the slope
        max_anchor_distance (_type_): How far away should the anchors be at most

    Returns:
        _type_: _description_
    """
    max_main_line_slope_deviation = 45

    if plot_possible_lines:
        # Make a canvas and add simple view
        canvas = vispy.scene.SceneCanvas(keys="interactive", show=True)
        view = canvas.central_widget.add_view()
    else:
        view = None

    # generate the list of line candidates within max_slope_angle
    line_candidate_list = list(itertools.product(road_points, target_trees.geometry))
    line_candidate_list_combinations = [
        LineString(combination) for combination in line_candidate_list
    ]
    line_df = pd.DataFrame(data={"line_candidates": line_candidate_list_combinations})
    print(len(line_df), " candidates initially")

    # filter by max_main_line_slope_deviation
    line_df["slope_deviation"] = [
        geometry_utilities.angle_between(line, slope_line)
        for line in line_candidate_list_combinations
    ]
    line_df = line_df[line_df["slope_deviation"] < max_main_line_slope_deviation]
    print(len(line_df), " after slope deviations")

    line_df = line_df.iloc[1:250]

    # filter the candidates for support trees
    # overall_trees, target, point, possible_line
    line_df["possible_support_trees"] = [
        generate_support_trees(
            overall_trees, Point(line.coords[1]), Point(line.coords[0]), line
        )
        for line in line_df["line_candidates"]
    ]
    # add to df and filter empty entries
    line_df = line_df[line_df["possible_support_trees"].apply(len) > 0]
    print(len(line_df), " after supports trees")

    # filter the triple angles for good supports
    line_df["possible_anchor_triples","center_tree_bhd"] = [
        generate_triple_angle(Point(line.coords[0]), line, anchor_trees)
        for line in line_df["line_candidates"]
    ]
    line_df = line_df[line_df["possible_anchor_triples"].notnull()]
    print(len(line_df), " after possible anchor triples")

    # compute the max supported force based on the BHD of the center tree
    security_factor = 5
    line_df["max_supported_force"] = (line_df["center_tree_bhd"]**2)*10/security_factor

    # check if we have no height obstructions - compute the supports we need according to line tension and anchor configs
    pos = []
    line_df["number_of_supports"], line_df["location_of_int_supports"] = zip(
        *[
            compute_required_supports(
                line["line_candidates"], line["possible_anchor_triples"],line["max_supported_force"],height_gdf, 0, plot_possible_lines, view, [], overall_trees, pos
            )
            for line in itertools.iterrows(line_df)
        ]
    )
    # and filter lines out without successful lines
    line_df = line_df[line_df["number_of_supports"].apply(lambda x: x is not False)]
    print(len(line_df), " after checking for height obstructions")

    # compute the angle between the line and the supports
    line_df["angle_between_supports"] = [
        compute_angle_between_supports(line, height_gdf)
        for line in line_df["line_candidates"]
    ]

    # create a dict of the coords of the starting points
    start_point_dict = dict(
        [(key, value.coords[0]) for key, value in enumerate(line_df["line_candidates"])]
    )

    if plot_possible_lines:
        height_gdf_small = height_gdf.iloc[::10, :]
        # pos of lines
        pos_lines = np.hstack((pos)).T
        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(pos_lines, edge_width=0, face_color=(1, 1, 0.5, 1), size=5)
        view.add(scatter)
        # possibility to connect lines, but doesnt really look good
        # N,S = pos_lines.shape
        # connect = np.empty((N*S-1,2), np.int32)
        # connect[:, 0] = np.arange(N*S-1)
        # connect[:, 1] = connect[:, 0] + 1
        # for i in range(S, N*S, S):
        #     connect[i-1, 1] = i-1
        # view.add(vispy.scene.Line(pos=pos_lines, connect=connect, width=5))

        # pos of heightgdf
        pos_height_gdf = np.vstack(
            (height_gdf_small["x"], height_gdf_small["y"], height_gdf_small["elev"])
        ).T
        # create scatter object and fill in the data
        scatter = visuals.Markers()
        scatter.set_data(
            pos_height_gdf, edge_width=0, face_color=(1, 1, 1, 0.5), size=5
        )
        view.add(scatter)
        view.camera = "turntable"  # or try 'arcball'
        # add a colored 3D axis for orientation
        axis = visuals.XYZAxis(parent=view.scene)

    return line_df, start_point_dict


def compute_angle_between_supports(possible_line, height_gdf):
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
    start_point_xy_height = fetch_point_elevation(
        start_point_xy, height_gdf, max_deviation
    )
    end_point_xy_height = fetch_point_elevation(end_point_xy, height_gdf, max_deviation)

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


def compute_line_height(this_cable_road, height_gdf):
    x_points, y_points = zip(
        *[(point.x, point.y) for point in this_cable_road.points_along_line]
    )

    # get the first elevation point of the list which satisfies the max_deviation condition
    this_cable_road.floor_height_below_line_points = [
        height_gdf.loc[
            (
                height_gdf.x.between(
                    x_points[i] - this_cable_road.max_deviation,
                    x_points[i] + this_cable_road.max_deviation,
                )
            )
            & (
                height_gdf.y.between(
                    y_points[i] - this_cable_road.max_deviation,
                    y_points[i] + this_cable_road.max_deviation,
                )
            ),
            "elev",
        ].values[0]
        for i in range(len(x_points))
    ]

    # create arrays for start and end point
    this_cable_road.line_start_point_array = np.array(
        [
            this_cable_road.start_point.x,
            this_cable_road.start_point.y,
            this_cable_road.start_point_height,
        ]
    )
    this_cable_road.line_end_point_array = np.array(
        [
            this_cable_road.end_point.x,
            this_cable_road.end_point.y,
            this_cable_road.end_point_height,
        ]
    )


def create_cable_road_object(possible_line):
    """Create a cr object that holds all the properties we need

    Args:
        possible_line (_type_): _description_

    Returns:
        _type_: _description_
    """
    start_point, end_point = Point(possible_line.coords[0]), Point(
        possible_line.coords[1]
    )
    this_cable_road = classes.Cable_Road(start_point, end_point)

    return this_cable_road

def initialize_line_tension(this_cable_road, current_supports):
        #set tension of the cable_road
    s_br_mindestbruchlast = 170000#in newton
    this_cable_road.s_max_maximalspannkraft = s_br_mindestbruchlast/3
    this_cable_road = this_cable_road.s_max_maximalspannkraft*(current_supports/current_supports+1)


def check_if_no_collisions_overall_line(this_cable_road, plot_possible_lines, view, pos, current_supports, anchor_triplets, center_tree_bhd):

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
    #mechanical_computations.calculate_length_unloaded_skyline(this_cable_road)

    # Zweifel Schritt 2 - length of skyline with load
    #mechanical_computations.calculate_length_loaded_skyline(this_cable_road)

    # increase tension by predefined amount
    #this_cable_road.t_v_j_bar_tensile_force_at_center_span = (
    #    this_cable_road.t_v_j_bar_tensile_force_at_center_span + 10
    #)

    # Zweifel Schritt 3 - calculate properties of skyline under load (ie deflection)
    #y_x_deflections = mechanical_computations.calculate_deflections(this_cable_road)

    # Process of updating the tension and checking if we touch ground and anchors hold
   
    this_cable_road.anchors_hold = True
    this_cable_road.no_collisions = False

    while this_cable_road.s_current_tension < this_cable_road.s_max_maximalspannkraft:

        #1. do the anchors hold? break the loop - this configuration doesnt work
        if not check_if_anchor_trees_hold(this_cable_road, anchor_triplets, center_tree_bhd):
            this_cable_road.anchors_hold = False
            break

        #1. calculate current deflections with a given tension
        y_x_deflections = [
            mechanical_computations.lastdurchhang_at_point(this_cable_road, point, this_cable_road.s_current_tension)
            for point in this_cable_road.points_along_line
        ]

        #  check the distances between each floor point and the ldh point
        this_cable_road.sloped_line_to_floor_distances = (
            this_cable_road.line_to_floor_distances - y_x_deflections
        )

        lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

        # check if the line is above the ground and set it to false if we have a collision
        if lowest_point_height > min_height:
            this_cable_road.no_collisions = False
        else:
            # we found no collisions and exit the loop
            this_cable_road.no_collisions = True
            break

        this_cable_road.s_current_tension+=1000

    # plot the lines if true
    if plot_possible_lines:
        plotting.plot_lines(
            this_cable_road.floor_points,
            this_cable_road.floor_height_below_line_points,
            this_cable_road.sloped_line_to_floor_distances,
            view,
            pos,
        )





def check_if_no_collisions_segments(this_cable_road):
    # Process of updating the tension and checking if we touch ground and anchors hold
    this_cable_road.no_collisions = False

    #1. calculate current deflections with a given tension
    y_x_deflections = [
        mechanical_computations.lastdurchhang_at_point(this_cable_road, point, this_cable_road.s_current_tension)
        for point in this_cable_road.points_along_line
    ]

    #  check the distances between each floor point and the ldh point
    this_cable_road.sloped_line_to_floor_distances = (
        this_cable_road.line_to_floor_distances - y_x_deflections
    )

    lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)

    # check if the line is above the ground and set it to false if we have a collision
    if lowest_point_height > this_cable_road.min_height:
            this_cable_road.no_collisions = False
    else:
            # we found no collisions and exit the loop
        this_cable_road.no_collisions = True


def check_if_support_withstands_tension(diameter_at_height,attached_at_height, loaded_cable_road, empt_cable_road, current_tension):
    """
    This function calculates the exerted force on a support tree, based on the tension in a loaded cable road and the angle between it and an empty cable road. The calculation uses trigonometry and the sine function to determine the force on the support. The maximum force that the support can bear is then determined using a Euler buckling calculation. The function returns True if the support can handle more force than is being exerted on it, and False otherwise.
    """

    # 1. construct tangents - from the left middle of the loaded cr to its endpoint
    full_tangent = LineString(loaded_cable_road.line.centroid, loaded_cable_road.end_point)
    empt_tangent = LineString(empt_cable_road.start_point, empt_cable_road.line.centroid)

    # get their angles
    angle_tangents = geometry_utilities.angle_between(full_tangent,empt_tangent)

    # compute the exerted force with trigonometr"""  """
    # gegenkathete = hpotenuse*sin(angle/2)
    # doppeltes Dreieck - gegenkathete*2
    force_on_support = (current_tension*math.sin(angle_tangents/2))*2

    # get the supported force of the support tree
    # TBD this can also be done in advance
    max_force_of_support = mechanical_computations.euler_knicklast(diameter_at_height, attached_at_height)

    # return true if the support can bear more than the exerted force
    return max_force_of_support > force_on_support


def compute_initial_cable_road(possible_line, height_gdf):
    """Create a CR object and compute its initial properties like height, points along line etc

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    this_cable_road = create_cable_road_object(possible_line)

    this_cable_road.start_point_height = (
        fetch_point_elevation(
            this_cable_road.start_point, height_gdf, this_cable_road.max_deviation
        )
        + this_cable_road.support_height
    )
    this_cable_road.end_point_height = (
        fetch_point_elevation(
            this_cable_road.end_point, height_gdf, this_cable_road.max_deviation
        )
        + this_cable_road.support_height
    )

    # fetch the floor points along the line
    this_cable_road.points_along_line = generate_road_points(possible_line, interval=2)

    # get the height of those points and set them as attributes to the CR object
    compute_line_height(this_cable_road, height_gdf)

        # generate floor points and their distances
    this_cable_road.floor_points = list(
        zip(
            [point.x for point in this_cable_road.points_along_line],
            [point.y for point in this_cable_road.points_along_line],
            this_cable_road.floor_height_below_line_points,
        )
    )

    this_cable_road.line_to_floor_distances = [
        geometry_utilities.lineseg_dist(
            point,
            this_cable_road.line_start_point_array,
            this_cable_road.line_end_point_array,
        )
        for point in this_cable_road.floor_points
    ]

    # get the rope lenght
    this_cable_road.b_length_whole_section = this_cable_road.start_point.distance(
        this_cable_road.end_point
    )
    this_cable_road.c_rope_length = geometry_utilities.distance_between_3d_points(
        this_cable_road.line_start_point_array, this_cable_road.line_end_point_array
    )

    return this_cable_road


def compute_required_supports(
    possible_line,
    anchor_triplets,
    center_tree_bhd,
    height_gdf,
    current_supports,
    plot_possible_lines,
    view,
    location_supports,
    overall_trees,
    pos,
):
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points
     elevation plus the support height) which are less than min_height away from the line.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    this_cable_road = compute_initial_cable_road(possible_line, height_gdf)

    initialize_line_tension(this_cable_road, current_supports)

    # tension the line and check if anchors hold and we have collisions
    check_if_no_collisions_overall_line(this_cable_road, plot_possible_lines, view, pos, current_supports, anchor_triplets, center_tree_bhd)

    if this_cable_road.no_collisions and this_cable_road.anchors_hold:
        return current_supports, location_supports

    # enter the next recursive loop if not b creating supports
    else:

        # 1. get the point of contact
        lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)
        sloped_line_to_floor_distances_index = int(
            np.where(this_cable_road.sloped_line_to_floor_distances == lowest_point_height)[0]
        )

        # 2. Get all trees which are within 0.5-2 meter distance to the line in general
        intermediate_support_candidates = overall_trees[
            (overall_trees.distance(possible_line) < 2)
            & (overall_trees.distance(possible_line) > 0.5)
        ]

        # 3. stop if there are no candidates, also stop if we have more than four supports - not viable
        if len(intermediate_support_candidates) < 1 or current_supports > 3:
            return None, None

        # 4. enumerate through list of candidates - sort by distance to the point of contact
        point_of_contact = this_cable_road.points_along_line[
            sloped_line_to_floor_distances_index
        ]
        distance_candidates = intermediate_support_candidates.distance(point_of_contact)
        distance_candidates = distance_candidates.sort_values(ascending=True)

        # loop through the candidates to check if one has no obstructions
        for candidate in distance_candidates.index:

            candidate_tree = overall_trees.iloc[candidate]
            
            new_support_point, road_to_support_line, support_to_anchor_line = create_candidate_points_and_lines(
                candidate,
                this_cable_road.start_point,
                this_cable_road.end_point,
                candidate_tree.geometry,
                overall_trees,
            )

            # create left and right sub_cableroad
            left_cable_road = compute_initial_cable_road(possible_line, height_gdf)
            right_cable_road = compute_initial_cable_road(possible_line, height_gdf)

            # iterate through the possible attachments of the support and see if we touch ground
            for diameters_index in len(candidate_tree.hoehe):
                
                support_withstands_tension = check_if_support_withstands_tension(candidate_tree.diameter[diameters_index], candidate_tree.hoehe[diameters_index], left_cable_road, right_cable_road, this_cable_road.current_tension)
                if not support_withstands_tension:
                    continue

                # 6. no collisions left and right?
                check_if_no_collisions_segments(
                    left_cable_road, height_gdf, plot_possible_lines, view, pos
                )
                if not left_cable_road.no_collisions:
                    continue

                check_if_no_collisions_segments(
                    right_cable_road, height_gdf, plot_possible_lines, view, pos
                )

                if not right_cable_road.no_collisions:
                    continue

            # no collisions were found and support holds, return our current supports
            if (left_cable_road.no_collisions and right_cable_road.no_collisions and support_withstands_tension):
                current_supports += 1
                location_supports.append(candidate_tree.geometry)
                return current_supports, location_supports

        # if we passed through the loop without finding suitable candidates, set the first candidate as support and find sub-supports recursively
        current_supports += 1
        first_candidate_point = overall_trees.iloc[
            distance_candidates.index[0]
        ].geometry
        location_supports.append(first_candidate_point)

        # compute necessary supports to the left
        current_supports, location_supports = compute_required_supports(
            support_to_anchor_line,
            height_gdf,
            current_supports,
            plot_possible_lines,
            view,
            location_supports,
            overall_trees,
            pos,
        )

        # fewer than max supports? then check line to the right for suports
        if current_supports and current_supports < 4:
            current_supports, location_supports = compute_required_supports(
                support_to_anchor_line,
                height_gdf,
                current_supports,
                plot_possible_lines,
                view,
                location_supports,
                overall_trees,
                pos,
            )
            # still acceptable amounts of supports?
            if current_supports and current_supports < 4:
                return current_supports, location_supports
            else:
                return False, False
        else:
            return False, False


def create_candidate_points_and_lines(
    candidate, start_point, end_point, candidate_point, overall_trees
):
    # 5. create the new candidate point and lines to/from it
    road_to_support_line = LineString([start_point, candidate_point])
    support_to_anchor_line = LineString([candidate_point, end_point])
    # get the location of the support point - why am Ii not able to do this with intermediate_support_candidates?
    new_support_point = overall_trees.iloc[candidate].geometry
    road_to_support_line = LineString([start_point, new_support_point])
    support_to_anchor_line = LineString([new_support_point, end_point])

    return new_support_point, road_to_support_line, support_to_anchor_line


def fetch_point_elevation(point, height_gdf, max_deviation):
    """
    Fetches the elevation of a given point.

    Args:
    point (Point): The point for which the elevation is to be fetched.
    height_gdf (GeoDataFrame): A GeoDataFrame containing the elevations.
    max_deviation (float): The maximum deviation allowed while fetching the elevation.

    Returns:
    float: The elevation of the given point.
    """
    return height_gdf.loc[
        (height_gdf.x > point.x - max_deviation)
        & (height_gdf.x < point.x + max_deviation)
        & (height_gdf.y < point.y + max_deviation)
        & (height_gdf.y > point.y - max_deviation),
        "elev",
    ].values[0]


def generate_support_trees(overall_trees, target, point, possible_line):
    """find trees in overall_trees along the last bit of the possible_line that are close to the line and can serve as support tree

    Args:
        overall_trees (_type_): GDF of all trees
        target (_type_): The last tree
        point (_type_): The road point we are starting from
        possible_line (_type_): The limne between target and point

    Returns:
        _type_: _description_
    """
    # Parameters
    min_support_sideways_distance = 0.1
    max_support_sideways_distance = 1.5
    min_support_anchor_distance = 10
    max_support_anchor_distance = 20

    # find those trees that are within the sideways distance to the proposed line
    support_tree_candidates = overall_trees[
        overall_trees.geometry.distance(possible_line).between(
            min_support_sideways_distance, max_support_sideways_distance
        )
    ]

    # find those trees that are within the right distance to the target tree
    support_tree_candidates = support_tree_candidates[
        support_tree_candidates.geometry.distance(target).between(
            min_support_anchor_distance, max_support_anchor_distance
        )
    ]

    # select only those support tree candidates which are close to the roadside point than the target tree
    support_tree_candidates = support_tree_candidates[
        support_tree_candidates.geometry.distance(point) < target.distance(point)
    ]

    return support_tree_candidates


def generate_road_points(road_geometry, interval):
    """Generate a list of points with a given interval along the road geometry

    Args:
        road_geometry (_type_): A list of lines that define the road
        interval: The interval in which a new point is calculated

    Returns:
        _type_: _description_
    """
    # thanks to https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
    distance_delta = interval
    distances = np.arange(0, road_geometry.length, distance_delta)
    road_points = [road_geometry.interpolate(distance) for distance in distances] + [
        Point(road_geometry.coords[1])
    ]

    return road_points


def compute_points_covered_by_geometry(points_gdf, geometry, min_trees_covered):
    """Return the points covered by geometry in the points_gdf

    Returns:
        set(geometry), int : set of covered points as well as their amount
    """

    contained_points = filter_gdf_by_contained_elements(points_gdf, geometry)

    if len(contained_points) < min_trees_covered:
        return
    # filter only those points
    # and return and set of the covered points as well as the amount of trees covered
    return set(contained_points["id"].values), len(contained_points)


def compute_points_covered_per_row(points_gdf, row_gdf, buffer_size, min_trees_covered):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """

    # already create buffer to avoid having to recreate this object every time
    row_gdf["buffer"] = row_gdf.apply(
        lambda row: row.geometry.buffer(buffer_size), axis=1
    )

    # appply and return the points covered by each buffer
    return row_gdf["buffer"].apply(
        lambda row: compute_points_covered_by_geometry(
            points_gdf, row, min_trees_covered
        )
    )


def filter_gdf_by_contained_elements(gdf, polygon):
    """Return only the points in the gdf which are covered by the polygon

    Args:
        gdf (_type_): The gdf to filter
        polygon (_type_): A polygon geometry

    Returns:
        _type_: the filtered gdf
    """
    # get the points which are contained in the geometry
    coverage_series = gdf.geometry.intersects(polygon)
    # and select only those points from the point_gdf
    contained_points = gdf[coverage_series]

    return contained_points


def compute_distances_facilities_clients(tree_gdf, line_gdf):
    """Create a numpy matrix with the distance between every tree and line

    Args:
        tree_gdf (_type_): A gdf containing the trees
        line_gdf (_type_): A gdf containing the facilities/lines

    Returns:
        _type_: A numpy matrix of the costs/distances
    """
    # compute the distance to each tree for every row
    tree_line_distances = []
    carriage_support_distances = []

    for line in line_gdf.iterrows():
        line_tree_distance = tree_gdf.geometry.distance(line[1].geometry)
        # get the nearest point between the tree and the cable road for all trees
        # project(tree,line)) gets the distance of the closest point on the line
        carriage_support_distance = [
            line[1].geometry.project(Point(tree_geometry.coords[0]))
            for tree_geometry in tree_gdf.geometry
        ]

        tree_line_distances.append(line_tree_distance)
        carriage_support_distances.append(carriage_support_distance)

    # pivot the table and convert to numpy matrix (solver expects it this way)
    return (
        np.asarray(tree_line_distances).transpose(),
        np.asarray(carriage_support_distances).transpose(),
    )


def generate_triple_angle(point, line_candidate, anchor_trees):
    """Generate a list of line-triples that are within correct angles to the road point and slope line.
    Checks whether:
    - anchor trees are within (less than) correct distance
    - all of those lines have a deviation < max outer anchor angle to the slope line
    Then we create pairs of lines within max_outer_anchor_angle to each other AND one of them with <max_center_tree_slope_angle to the slope line
    This results in pairs of lines were one is the inner and one the right-most (or vice versa)
    Then, we check for a third possible line if it is within 2*min_outer_anch_angle and 2* max_anchor_angle for one point and within min_outer, max_outer for the other -> so two lines are not on the same side!
    This results in triples where we have the central and rightmost line from the tuples section plus one line that is at least 2*min and at most 2*max outer angle away from the rightmost line (or vv.)

    Args:
        point (_type_): The road point we want to check for possible anchors
        line_candidate (_type_): _description_
        anchor_trees (_type_): _description_
        max_anchor_distance (_type_): _description_
        max_outer_anchor_angle (_type_): Max angle between right and left line
        min_outer_anchor_angle (_type_): Minimum angle between right and left line
        max_center_tree_slope_angle (_type_): Max deviation of center line from slope line

    Returns:
        _type_: _description_
    """
    min_outer_anchor_angle = 20
    max_outer_anchor_angle = 50
    max_center_tree_slope_angle = 3
    max_anchor_distance = 30

    # 1. get list of possible anchors -> anchor trees
    anchor_trees_working_copy = deepcopy(anchor_trees)

    # 2. check which points are within distance
    anchor_trees_working_copy = anchor_trees_working_copy[
        anchor_trees_working_copy.geometry.distance(point) < max_anchor_distance
    ]

    # 3. create lines to all these possible connections
    if anchor_trees_working_copy.empty or len(anchor_trees_working_copy) < 3:
        return
    possible_anchor_lines = anchor_trees_working_copy.geometry.apply(
        lambda x: LineString([x, point])
    )

    # check if all of those possible lines are within the max deviation to the slope
    possible_anchor_lines = possible_anchor_lines[
        possible_anchor_lines.apply(
            lambda x: geometry_utilities.angle_between(x, line_candidate)
            < max_outer_anchor_angle
        )
    ].to_list()

    if len(possible_anchor_lines) < 3:
        return

    # 4. check first pairs: one within 10-30 angle to the other and one should be <5 degrees to the slope line
    pairwise_angle = [
        (x, y)
        for x, y in itertools.combinations(possible_anchor_lines, 2)
        if min_outer_anchor_angle
        < geometry_utilities.angle_between(x, y)
        < max_outer_anchor_angle
        and (
            geometry_utilities.angle_between(x, line_candidate)
            < max_center_tree_slope_angle
            or geometry_utilities.angle_between(y, line_candidate)
            < max_center_tree_slope_angle
        )
    ]

    # skip if we dont have enough candidates
    if len(pairwise_angle) < 3:
        return

    # 5. check if the third support line is also within correct angle - within 2*min_outer_anch_angle and 2* max_anchor_angle for one point and within min_outer, max_outer for the other -> so two lines are not on the same side!
    triple_angle = []
    center_tree_bhd = []
    for x, y in pairwise_angle:
        for z in possible_anchor_lines:
            # make sure that we are not comparing a possible line with itself
            if x is not z and y is not z:
                a = (
                    max_outer_anchor_angle * 2 - 10
                    < geometry_utilities.angle_between(x, z)
                    < max_outer_anchor_angle * 2
                    and min_outer_anchor_angle
                    < geometry_utilities.angle_between(y, z)
                    < max_outer_anchor_angle
                )
                b = (
                    max_outer_anchor_angle * 2 - 10
                    < geometry_utilities.angle_between(y, z)
                    < max_outer_anchor_angle * 2
                    and min_outer_anchor_angle
                    < geometry_utilities.angle_between(x, z)
                )

                if (a, b):
                    triple_angle.append([x, y, z])
                    # find the line with the smallest angle
                    center_line = min([geometry_utilities.angle_between(line, line_candidate) for line in [x, y, z]])
                    # get its end tree and retrive its BHD from the DF
                    this_center_tree_bhd = anchor_trees_working_copy[anchor_trees_working_copy.geometry==center_line.coords[0]]["BHD"]
                    center_tree_bhd.append(this_center_tree_bhd)

    return triple_angle, center_tree_bhd


def check_if_anchor_trees_hold(this_cable_road, anchor_triplets, max_supported_tension):
    # get force at last support
    exerted_force = this_cable_road.s_current_tension
    #this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support
    # todo Parallelverschiebung to get actual force

    # check if the supported tension is greater than the exerted force
    sufficient_anchors = [anchor_triplets[i] for i in len(max_supported_tension) if max_supported_tension[i]>exerted_force]

    if sufficient_anchors:
        this_cable_road.anchor_triples = sufficient_anchors
        return True
    else:
        return False

