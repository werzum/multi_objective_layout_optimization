from shapely.geometry import LineString, Point
import numpy as np
import itertools
import pandas as pd
import vispy.scene
import geopandas as gpd
from pandas import DataFrame
from multiprocesspandas import applyparallel

from src import (
    geometry_utilities,
    geometry_operations,
    mechanical_computations,
    classes,
    plotting,
)

# Main functions to compute the cable road which calls the other functions


def generate_possible_lines(
    road_points: list[Point],
    target_trees: gpd.GeoDataFrame,
    anchor_trees: gpd.GeoDataFrame,
    overall_trees: gpd.GeoDataFrame,
    slope_line: LineString,
    height_gdf: gpd.GeoDataFrame,
    plot_possible_lines: bool,
) -> tuple[DataFrame, dict]:
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
    line_df = DataFrame(data={"line_candidates": line_candidate_list_combinations})
    print(len(line_df), " candidates initially")

    # filter by max_main_line_slope_deviation
    line_df["slope_deviation"] = [
        geometry_utilities.angle_between(line, slope_line)
        for line in line_candidate_list_combinations
    ]
    line_df = line_df[line_df["slope_deviation"] < max_main_line_slope_deviation]
    print(len(line_df), " after slope deviations")

    line_df = line_df.iloc[::10]

    # filter the candidates for support trees
    # overall_trees, target, point, possible_line
    line_df["tree_anchor_support_trees"] = [
        generate_tree_anchor_support_trees(
            overall_trees, Point(line.coords[1]), Point(line.coords[0]), line
        )
        for line in line_df["line_candidates"]
    ]
    # add to df and filter empty entries
    line_df = line_df[line_df["tree_anchor_support_trees"].apply(len) > 0]
    print(len(line_df), " after supports trees")

    # filter the triple angles for good supports
    line_df["possible_anchor_triples"], line_df["max_holding_force"] = zip(
        *[
            generate_triple_angle(Point(line.coords[0]), line, anchor_trees)
            for line in line_df["line_candidates"]
        ]
    )
    line_df = line_df[line_df["possible_anchor_triples"].notnull()]
    print(len(line_df), " after possible anchor triples")

    # check if we have no height obstructions - compute the supports we need according to line tension and anchor configs
    pos = []

    (
        line_df["number_of_supports"],
        line_df["location_of_int_supports"],
        line_df["current_tension"],
    ) = zip(
        *[
            compute_required_supports(
                line["line_candidates"],
                line["possible_anchor_triples"],
                line["max_holding_force"],
                line["tree_anchor_support_trees"],
                height_gdf,
                0,
                overall_trees,
                [],
                plot_possible_lines,
                view,
                pos,
            )
            for index, line in line_df.iterrows()
        ]
    )

    # and filter lines out without successful lines
    line_df = line_df[line_df["number_of_supports"].apply(lambda x: x is not False)]
    print(len(line_df), " after checking for height obstructions")

    if len(line_df) < 1:
        raise ValueError("No candidates left")

    # compute the angle between the line and the supports
    line_df["angle_between_supports"] = [
        mechanical_computations.compute_angle_between_supports(line, height_gdf)
        for line in line_df["line_candidates"]
    ]

    # create a dict of the coords of the starting points
    start_point_dict = dict(
        [(key, value.coords[0]) for key, value in enumerate(line_df["line_candidates"])]
    )

    if plot_possible_lines:
        plotting.plot_vispy_scene(height_gdf, view, pos)

    return line_df, start_point_dict


def compute_required_supports(
    possible_line: LineString,
    anchor_triplets: list,
    max_supported_forces: list[float],
    tree_anchor_support_trees: list,
    height_gdf: gpd.GeoDataFrame,
    current_supports: int,
    overall_trees: gpd.GeoDataFrame,
    location_supports: list,
    plot_possible_lines: bool = False,
    view: vispy.scene.ViewBox | None = None,
    pos: list | None = None,
    pre_tension: int = None,
) -> tuple[int, list[Point], int]:
    """A function to check whether there are any points along the line candidate (spanned up by the starting/end points
     elevation plus the support height) which are less than min_height away from the line.

    Args:
        possible_line (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    this_cable_road = classes.Cable_Road(
        possible_line,
        height_gdf,
        pre_tension=pre_tension,
        current_supports=current_supports,
    )

    print("Tension to begin with is", this_cable_road.s_current_tension)

    tower_and_anchors_hold = False
    # set supports_Hold to false if we have supports and need to check them, else set to true
    supports_hold = False if current_supports > 0 else True
    # and set the support height to tower height and update the line heights accordingly
    this_cable_road.start_support_height = this_cable_road.tower_height
    this_cable_road.compute_loaded_unloaded_line_height()

    # decrement by 10kn increments while checking if the towers and intermediate supports hold
    while not (tower_and_anchors_hold and supports_hold):
        tower_and_anchors_hold = (
            mechanical_computations.check_if_tower_and_anchor_trees_hold(
                this_cable_road, max_supported_forces, anchor_triplets, height_gdf
            )
        )
        print(tower_and_anchors_hold)
        if tower_and_anchors_hold and current_supports > 0:
            supports_hold = mechanical_computations.check_if_supports_hold(
                this_cable_road, tree_anchor_support_trees, height_gdf
            )

        # decrement by 10kn increments
        if not (tower_and_anchors_hold and supports_hold):
            this_cable_road.s_current_tension -= 10000
            if this_cable_road.s_current_tension < 20000:
                return return_failed()

        print(this_cable_road.s_current_tension)

    # if we found a tension that is high enough and anchors support it, we continue
    min_cr_tension = 30000
    if tower_and_anchors_hold and this_cable_road.s_current_tension > min_cr_tension:
        this_cable_road.anchors_hold = True
    else:
        print("CR tension is too low with ", this_cable_road.s_current_tension)
        return return_failed()

    print("After the iterative process it is now", this_cable_road.s_current_tension)

    # tension the line and check if anchors hold and we have collisions
    mechanical_computations.check_if_no_collisions_cable_road(this_cable_road)

    if this_cable_road.no_collisions and this_cable_road.anchors_hold:
        print("successful on first try")
        return return_sucessful(current_supports, location_supports, this_cable_road)

    # exit this line since anchors dont hold and supports wont help with that
    if not this_cable_road.anchors_hold:
        print("anchors dont hold")
        return return_failed()

    if current_supports and current_supports < 4:
        print("more than 4 supports not possible")
        return return_failed()
    # enter the next recursive loop if not b creating supports
    else:
        print("Need to find supports")
        # get the distance candidates
        distance_candidates = setup_support_candidates(
            this_cable_road, overall_trees, current_supports, possible_line
        )

        if distance_candidates is False:
            return return_failed()

        # loop through the candidates to check if one has no obstructions
        for candidate in distance_candidates.index:
            # print("looping through distance candidates")
            # fetch the candidate

            (
                road_to_support_cable_road,
                support_to_anchor_cable_road,
                candidate_tree,
            ) = create_sideways_cableroads(
                overall_trees, this_cable_road, candidate, height_gdf, current_supports
            )

            # check if the candidate is too close to the anchor
            if (
                min(
                    len(road_to_support_cable_road.points_along_line),
                    len(support_to_anchor_cable_road.points_along_line),
                )
                < 4
            ):
                print(
                    "candidate too close to anchor, skipping in check if no coll overall line sideways CR"
                )
                continue

            # iterate through the possible attachments of the support and see if we touch ground
            # start with at least three meters height
            if len(candidate_tree.height_series) < 3:
                continue
            for height_index in range(6, len(candidate_tree.height_series)):
                print("raising height to ", height_index)
                # increase the support height
                road_to_support_cable_road.end_support_height = height_index
                support_to_anchor_cable_road.start_support_height = height_index

                support_withstands_tension = (
                    mechanical_computations.check_if_support_withstands_tension(
                        candidate_tree.max_supported_force_series[height_index],
                        road_to_support_cable_road,
                        support_to_anchor_cable_road,
                    )
                )
                if (not support_withstands_tension) or (height_index) > 20:
                    # next candidate - tension just gets worse with more height
                    break

                # 6. no collisions left and right? go to next candidate if this one is already not working out
                # first set the height of the support
                mechanical_computations.check_if_no_collisions_cable_road(
                    road_to_support_cable_road
                )
                mechanical_computations.check_if_no_collisions_cable_road(
                    support_to_anchor_cable_road
                )

                if (
                    support_to_anchor_cable_road.no_collisions
                    and road_to_support_cable_road.no_collisions
                ):
                    # we found a viable configuration - break out of this loop and print if desired
                    break
                print("iteratirt through height series since we have collisions")

            # no collisions were found and support holds, return our current supports
            if (
                road_to_support_cable_road.no_collisions
                and support_to_anchor_cable_road.no_collisions
                and support_withstands_tension
            ):
                print("found viable sub-config")
                current_supports += 1
                location_supports.append(candidate_tree.geometry)
                if plot_possible_lines:
                    plotting.plot_lines(road_to_support_cable_road, pos)
                    plotting.plot_lines(support_to_anchor_cable_road, pos)
                return return_sucessful(
                    current_supports, location_supports, this_cable_road
                )

        # if we passed through the loop without finding suitable candidates, set the first candidate as support and find sub-supports recursively
        print("didnt find suitable candidate")

        # proceed with the working cr and find sub-supports - fetch the candidate we last looked at
        # increment the supports correspondingly
        current_supports += 1
        first_candidate_point = overall_trees.iloc[
            distance_candidates.index[0]
        ].geometry
        location_supports.append(first_candidate_point)

        # select first support as starting point
        candidate = distance_candidates.index[0]
        (
            road_to_support_cable_road,
            support_to_anchor_cable_road,
            candidate_tree,
        ) = create_sideways_cableroads(
            overall_trees, this_cable_road, candidate, height_gdf, current_supports
        )

        # test for collisions left and right - enter the recursive loop to compute subsupports
        (
            current_supports,
            location_supports,
            current_tension,
        ) = test_collisions_left_right(
            [road_to_support_cable_road, support_to_anchor_cable_road],
            current_supports,
            location_supports,
            tree_anchor_support_trees,
            anchor_triplets,
            max_supported_forces,
            overall_trees,
            height_gdf,
            plot_possible_lines,
            view,
            pos,
        )

        # computed sub-supports and see if we had enough
        if current_supports and current_supports > 4 or current_supports is False:
            return return_failed()
        else:
            return return_sucessful(
                current_supports, location_supports, this_cable_road
            )


# helper function to have return functions in one place
def return_failed() -> tuple[bool, bool, bool]:
    return False, False, False


def return_sucessful(
    current_supports: int,
    location_supports: list[Point],
    this_cable_road: classes.Cable_Road,
) -> tuple[int, list[Point], int]:
    return current_supports, location_supports, current_tension(this_cable_road)


def current_tension(this_cable_road: classes.Cable_Road) -> int:
    return int(this_cable_road.s_current_tension)


# Generating different structures


def generate_triple_angle(
    point: Point, line_candidate: LineString, anchor_trees: gpd.GeoDataFrame
) -> tuple[list, list] | tuple[None, None]:
    """Generate a list of line-triples that are within correct angles to the road point and slope line and the corresponding max supported force by the center tree.
    Checks whether:
    - anchor trees are within (less than) correct distance
    - all of those lines have a deviation < max outer anchor angle to the slope line
    - outward anchor trees are within 20 to 60m to each other



    Args:
        point (_type_): The road point we want to check for possible anchors
        line_candidate (_type_): _description_
        anchor_trees (_type_): _description_
        max_anchor_distance (_type_): _description_
        max_outer_anchor_angle (_type_): Max angle between right and left line
        min_outer_anchor_angle (_type_): Minimum angle between right and left line
        max_center_tree_slope_angle (_type_): Max deviation of center line from slope line

    Returns:
        list: A list of possible triple angles for this cable road in the form of [(center line, left line, right line), ...]
        list: A list of max supported force of the corresponding center tree
    """
    min_outer_anchor_angle = 20
    max_outer_anchor_angle = 50
    max_center_tree_slope_angle = 5
    max_anchor_distance = 40
    min_anchor_distane = 15

    # 1. get list of possible anchors -> anchor trees
    anchor_trees_working_copy = anchor_trees.copy()

    # 2. check which points are within distance
    anchor_trees_working_copy = anchor_trees_working_copy[
        (anchor_trees_working_copy.geometry.distance(point) <= max_anchor_distance)
        & (anchor_trees_working_copy.geometry.distance(point) >= min_anchor_distane)
    ]

    if anchor_trees_working_copy.empty or len(anchor_trees_working_copy) < 3:
        return None, None

    # 3. create lines to all these possible connections
    anchor_trees_working_copy["anchor_line"] = anchor_trees_working_copy.geometry.apply(
        lambda x: LineString([x, point])
    )

    # compute the angle between the slope line and the anchor line and get two dfs with possible center and side trees
    anchor_trees_working_copy["slope_angle"] = anchor_trees_working_copy[
        "anchor_line"
    ].apply(lambda x: geometry_utilities.angle_between(x, line_candidate))

    central_trees = anchor_trees_working_copy[
        anchor_trees_working_copy["slope_angle"].between(0, max_center_tree_slope_angle)
    ].copy()
    side_trees = anchor_trees_working_copy[
        anchor_trees_working_copy["slope_angle"].between(
            min_outer_anchor_angle, max_outer_anchor_angle
        )
    ]

    if len(central_trees) < 3 or len(side_trees) < 2:
        return None, None

    central_trees.loc[:, "possible_anchor_triples"] = central_trees[
        "anchor_line"
    ].apply(
        lambda x: [
            (x, LineString([y, point]), LineString([z, point]))
            for y, z in itertools.combinations(side_trees.geometry, 2)
            if y.distance(z) > 20 and y.distance(z) < 60
        ]
    )

    # if this did not yield viable anchors, proceed
    if len(central_trees["possible_anchor_triples"].sum()) < 1:
        return None, None
    else:
        return (
            # return the first combination per main anchor line
            [sublist[0] for sublist in central_trees["possible_anchor_triples"]],
            central_trees["max_holding_force"].to_list(),
        )


def generate_tree_anchor_support_trees(
    overall_trees: gpd.GeoDataFrame,
    target: Point,
    point: Point,
    possible_line: LineString,
):
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


def create_candidate_points_and_lines(
    candidate: gpd.GeoDataFrame,
    start_point: Point,
    end_point: Point,
    candidate_point: Point,
    overall_trees,
) -> tuple[Point, LineString, LineString]:
    # 5. create the new candidate point and lines to/from it
    road_to_support_line = LineString([start_point, candidate_point])
    support_to_anchor_line = LineString([candidate_point, end_point])
    # get the location of the support point - why am Ii not able to do this with intermediate_support_candidates?
    new_support_point = overall_trees.iloc[candidate].geometry
    road_to_support_line = LineString([start_point, new_support_point])
    support_to_anchor_line = LineString([new_support_point, end_point])

    return new_support_point, road_to_support_line, support_to_anchor_line


def setup_support_candidates(
    this_cable_road: classes.Cable_Road,
    overall_trees: gpd.GeoDataFrame,
    current_supports: int,
    possible_line: LineString,
) -> bool | gpd.GeoDataFrame:
    print("Setting up support candidates")

    # 1. get the point of contact
    lowest_point_height = min(this_cable_road.sloped_line_to_floor_distances)
    sloped_line_to_floor_distances_index = int(
        np.where(this_cable_road.sloped_line_to_floor_distances == lowest_point_height)[
            0
        ]
    )

    # 2. Get all trees which are within 0.5-2 meter distance to the line in general
    intermediate_support_candidates = overall_trees[
        (overall_trees.distance(possible_line) < 2)
        & (overall_trees.distance(possible_line) > 0.5)
    ]

    # 3. stop if there are no candidates, also stop if we have more than four supports - not viable
    if len(intermediate_support_candidates) < 1 or current_supports > 3:
        return False

    # 4. enumerate through list of candidates - sort by distance to the point of contact
    point_of_contact = this_cable_road.points_along_line[
        sloped_line_to_floor_distances_index
    ]

    distance_candidates = intermediate_support_candidates.distance(point_of_contact)
    distance_candidates = distance_candidates.sort_values(ascending=True)

    return distance_candidates


def create_sideways_cableroads(
    overall_trees, this_cable_road, candidate, height_gdf, current_supports
):
    """Create the sideways cable roads and return them

    Args:
        overall_trees (_type_): _description_
        this_cable_road (_type_): _description_
        candidate (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        _type_: _description_
    """
    # need to add the height and force per tree here
    candidate_tree = overall_trees.iloc[candidate]

    # create lines and points left and right
    (
        new_support_point,
        road_to_support_line,
        support_to_anchor_line,
    ) = create_candidate_points_and_lines(
        candidate,
        this_cable_road.start_point,
        this_cable_road.end_point,
        candidate_tree.geometry,
        overall_trees,
    )

    # create left and right sub_cableroad
    road_to_support_cable_road = classes.Cable_Road(
        road_to_support_line,
        height_gdf,
        pre_tension=this_cable_road.s_current_tension,
        current_supports=current_supports,
    )
    support_to_anchor_cable_road = classes.Cable_Road(
        support_to_anchor_line,
        height_gdf,
        pre_tension=this_cable_road.s_current_tension,
        current_supports=current_supports,
    )

    return road_to_support_cable_road, support_to_anchor_cable_road, candidate_tree


def test_collisions_left_right(
    cable_roads: list[classes.Cable_Road],
    current_supports: int,
    location_supports: list[Point],
    tree_anchor_support_trees: list,
    anchor_triplets: list[list[Point]],
    max_supported_forces: list[float],
    overall_trees: gpd.GeoDataFrame,
    height_gdf: gpd.GeoDataFrame,
    plot_possible_lines: bool = False,
    view: vispy.scene.ViewBox | None = None,
    pos: list | None = None,
) -> tuple[int, list[Point], int]:
    """test if the left and right cr have collisions and return the new support locations if not

    Args:
        road_to_support_cable_road (_type_): _description_
        support_to_anchor_cable_road (_type_): _description_
        current_supports (_type_): _description_
        location_supports (_type_): _description_
        overall_trees (_type_): _description_
        pos (_type_): _description_
        plot_possible_lines (_type_): _description_
        view (_type_): _description_
        anchor_triplets (_type_): _description_
        max_supported_forces (_type_): _description_
        height_gdf (_type_): _description_

    Returns:
        False, False or current_supports, location_supports
    """

    for cable_road in cable_roads:
        mechanical_computations.check_if_no_collisions_cable_road(cable_road)
        if not cable_road.no_collisions:
            (
                current_supports,
                location_supports,
                current_tension,
            ) = compute_required_supports(
                cable_road.line,
                anchor_triplets,
                max_supported_forces,
                tree_anchor_support_trees,
                height_gdf,
                current_supports,
                overall_trees,
                location_supports,
                plot_possible_lines,
                view,
                pos,
                cable_road.s_current_tension,
            )
            print("current supports", current_supports)
            # if we have more than 4 supports or didnt find any, we stop
            if current_supports and current_supports > 4 or not current_supports:
                return return_failed()

    return return_sucessful(current_supports, location_supports, cable_road)
