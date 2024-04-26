import numpy as np
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from src.main import geometry_utilities

# functions for computing underlying factors


def compute_cable_road_deviations_from_slope(
    line_gdf: gpd.GeoDataFrame, slope_line: LineString
) -> pd.Series:
    """Compute the deviation of each line from the slope line.
    Return a panda series with an array of the lengths of cable road segments
    for segments with more than 22° horizontal deviation and 25° vertical slope.

    Args:
        line_gdf (gpd.GeoDataFrame): GeoDataFrame of lines
        slope_line (LineString): LineString of the slope

        Returns:
            pd.Series: Series of arrays of deviations
    """
    line_deviations_array = np.empty(len(line_gdf))
    for index, line in line_gdf.iterrows():
        cable_road_object = line["Cable Road Object"]
        temp_arr = []
        sub_segments = list(cable_road_object.get_all_subsegments())
        # count either the deviations of the subsegments or the line itself
        if sub_segments:
            temp_arr = [
                subsegment.cable_road.line.length
                for subsegment in sub_segments
                if 35
                > geometry_utilities.angle_between(
                    subsegment.cable_road.line, slope_line
                )
                <= 25
            ]
        elif (
            35
            > geometry_utilities.angle_between(cable_road_object.line, slope_line)
            <= 25
        ):
            temp_arr = [line.geometry.length]
        line_deviations_array[index] = sum(temp_arr)

    return pd.Series(line_deviations_array)


def compute_line_costs(
    line_gdf: gpd.GeoDataFrame, uphill_yarding: bool, large_yarder: bool
) -> pd.Series:
    """Compute the cost of each line in the GeoDataFrame and reutrn the series
    Args:
        line_gdf (gpd.GeoDataFrame): GeoDataFrame of lines
    Returns:
        gpd.GeoSeries: Series of costs
    """
    line_cost_array = np.empty(len(line_gdf))
    for index, line in line_gdf.iterrows():
        line_length = line.geometry.length

        sub_segments = list(line["Cable Road Object"].get_all_subsegments())
        if sub_segments:
            intermediate_support_height = [
                sub_segment.end_support.attachment_height
                for sub_segment in sub_segments
            ]
            intermediate_support_height = intermediate_support_height[
                :-1
            ]  # skip the last one, since this is the tree anchor
            number_intermediate_supports = len(intermediate_support_height)
            avg_intermediate_support_height = float(
                np.mean(intermediate_support_height)
            )
        else:
            number_intermediate_supports = 0
            avg_intermediate_support_height = 0

        line_cost = line_cost_function(
            line_length,
            uphill_yarding,
            large_yarder,
            avg_intermediate_support_height,
            number_intermediate_supports,
        )

        line_cost_array[index] = line_cost

    return pd.Series(line_cost_array)


def line_cost_function(
    line_length: float,
    uphill_yarding: bool,
    large_yarder: bool,
    intermediate_support_height: float,
    number_intermediate_supports: int,
) -> float:
    """Compute the cost of each line based Kanzian

    Args:
        line_length (float): Length of the line
        uphill_yarding (bool): Wether the line is uphill or downhill
        large_yarder (bool): Wether the yarder is large or small
        intermediate_support_height (float): Height of the intermediate support
        number_intermediate_supports (int): Number of intermediate supports

    Returns:
        float: Cost of the line in Euros
    """
    cost_man_hour = 44

    # rename the variables according to Kanzian publication
    extraction_direction = uphill_yarding
    yarder_size = large_yarder
    corridor_type = True  # treat all corridors as first setup, else its hard to compute

    setup_time = math.e ** (
        1.42
        + 0.00229 * line_length
        + 0.03 * intermediate_support_height  # not available now?
        + 0.256 * corridor_type
        - 0.65 * extraction_direction  # 1 for uphill, 0 for downhill
        + 0.11 * yarder_size  # 1 for larger yarder, 0 for smaller 35kn
        + 0.491 * extraction_direction * yarder_size
    )

    takedown_time = math.e ** (
        0.96
        + 0.00233 * line_length
        - 0.31 * extraction_direction
        + 0.31 * number_intermediate_supports
        + 0.33 * yarder_size
    )

    install_time = setup_time + takedown_time
    line_cost = install_time * cost_man_hour
    return line_cost


def compute_tree_volume(BHD: pd.Series, height: pd.Series) -> pd.Series:
    # per extenden Denzin rule of thumb - https://www.mathago.at/wp-content/uploads/PDF/B_310.pdf
    return ((BHD.astype(int) ** 2) / 1000) * (((3 * height) + 25) / 100)


def calculate_felling_cost(
    client_range: range,
    facility_range: range,
    aij: np.ndarray,
    distance_carriage_support: np.ndarray,
    tree_volume: np.ndarray,
    average_steepness: float,
) -> np.ndarray:
    """Calculate the cost of each client-facility combination based on the productivity
    model by Gaffariyan, Stampfer, Sessions 2013 (Production Equations for Tower Yarders in Austria)
    It yields min/cycle, ie how long it takes in minutes to process a tree.
    We divide the results by 60 to yield hrs/cycle and multiply by 44 to get the cost per cycle

    Args:
        client_range (Range): range of clients
        facility_range (Range): range of facilities
        aij (np.array): Matrix of distances between clients and facilities
        distance_carriage_support (np.array): Distance between carriage and support
        average_steepness (float): Average steepness of the area

    Returns:
        np.array: matrix of costs for each client-facility combination
    """

    productivity_cost_matrix = np.zeros([len(client_range), len(facility_range)])
    # iterate ove the matrix and calculate the cost for each entry
    it = np.nditer(
        productivity_cost_matrix, flags=["multi_index"], op_flags=["readwrite"]
    )
    for x in it:
        cli, fac = it.multi_index
        # the cost per m3 based on the productivity model by Gaffariyan, Stampfer, Sessions 2013
        min_per_cycle = (
            0.007
            * distance_carriage_support[cli][
                fac
            ]  # the yarding distance between carriage and support
            + 0.043
            * (
                aij[cli][fac]
            )  # the distance from tree to cable road, aka lateral yarding distance - squared
            + 1.307 * tree_volume[fac] ** (-0.3)
            + 0.029 * 100  # the harvest intensity set to 100%
            + 0.038 * average_steepness
        )
        # add the remainder of the distance to the produced output
        if aij[cli][fac] > 15:
            min_per_cycle = min_per_cycle + (aij[cli][fac] - 15)

        # total cost with synchrofalke and two workers is 273.67 - we divide by 60 to get the cost per minute (4.56$/min)
        # and now get the cost to harvest this tree
        cost_per_cycle = 4.56 * min_per_cycle

        # hrs_per_cycle = min_per_cycle / 60
        # cost_per_cycle = (
        #     hrs_per_cycle * 44
        # )  # divide by 60 to get hrs/cycle and multiply by 44 to get cost

        x[...] = cost_per_cycle
    return productivity_cost_matrix


def logistic_growth_productivity_cost(productivity_cost: float):
    """Return the logistic growth function for the productivity cost. We grow this up to a value of 100, with a midpoint of 40 and a growth rate of 0.1"""
    return 100 / (1 + math.e ** (-0.1 * (productivity_cost - 40)))
