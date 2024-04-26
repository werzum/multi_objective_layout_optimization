import itertools
from typing import Union

import geopandas as gpd
import numpy as np

from pandas import DataFrame
from shapely.geometry import LineString, Point

from src.main import (
    global_vars,
    mechanical_computations,
    geometry_utilities,
    cable_road_computation,
)


# Main functions to compute the cable road which calls the other functions
def generate_possible_lines(
    road_points: list[Point],
    target_trees: gpd.GeoDataFrame,
    anchor_trees: gpd.GeoDataFrame,
    overall_trees: gpd.GeoDataFrame,
    slope_line: LineString,
    height_gdf: gpd.GeoDataFrame,
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
    global_vars.init(height_gdf)
    max_main_line_slope_deviation = 45

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

    # line_df = line_df.iloc[::10]

    # filter the candidates for support trees
    # overall_trees, target, point, possible_line
    line_df["tree_anchor_support_trees"] = [
        cable_road_computation.generate_tree_anchor_support_trees(
            overall_trees, Point(line.coords[1]), Point(line.coords[0]), line
        )
        for line in line_df["line_candidates"]
    ]

    # add to df and filter empty entries
    line_df = line_df[line_df["tree_anchor_support_trees"].apply(len) > 0]
    print(len(line_df), " after supports trees")

    # filter the triple angles for good supports
    (
        line_df["possible_anchor_triples"],
        line_df["max_holding_force"],
        line_df["road_anchor_tree_series"],
    ) = zip(
        *[
            cable_road_computation.generate_triple_angle(
                Point(line.coords[0]), line, anchor_trees
            )
            for line in line_df["line_candidates"]
        ]
    )
    line_df = line_df[line_df["possible_anchor_triples"].notnull()]
    print(len(line_df), " after possible anchor triples")

    # check if we have no height obstructions - compute the supports we need according to line tension and anchor configs
    line_df["Cable Road Object"] = [
        cable_road_computation.compute_required_supports(
            line["possible_anchor_triples"],
            line["max_holding_force"],
            line["tree_anchor_support_trees"],
            height_gdf,
            overall_trees,
            from_line=line["line_candidates"],
            recursion_counter=0,
        )
        for index, line in line_df.iterrows()
    ]

    # and filter lines out without successful lines
    line_df = line_df[line_df["Cable Road Object"].apply(lambda x: x is not False)]
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

    return line_df, start_point_dict
