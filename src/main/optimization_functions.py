import numpy as np
import pulp
import operator
from random import randint, choices
import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from src.main import geometry_utilities, geometry_operations, classes

# functions for computing underlying factors


def compute_length_of_steep_downhill_segments(line_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Compute the length of steep downhill segments of each line in the GeoDataFrame as per Bont 2019 to quantify environmental impact
    Args:
        line_gdf (gpd.GeoDataFrame): GeoDataFrame of lines
    Returns:
        pd.Series: Series of arrays of lengths of steep downhill segments
    """

    line_cost_array = np.empty(len(line_gdf))
    for index, line in line_gdf.iterrows():
        line_length = line.geometry.length
        sub_segments = list(line["Cable Road Object"].get_all_subsegments())
        if sub_segments:
            line_cost_array[index] = sum(
                subsegement.cable_road.line.length
                for subsegement in sub_segments
                if geometry_operations.get_slope(subsegement.cable_road) >= 10
            )
        else:
            line_cost_array[index] = (
                geometry_operations.get_slope(line["Cable Road Object"])
                if geometry_operations.get_slope(line["Cable Road Object"]) >= 10
                else 0
            )

    return pd.Series(line_cost_array)


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
    # add penalty according to deviation
    # def deviation_condition (individual_cost, individual_deviation):
    #     if 10 <= individual_deviation <= 35:
    #         return individual_cost
    #     elif individual_deviation >= 45:
    #         return individual_cost+line_length*2
    #     else:
    #         return individual_cost+(line_length*0.5)

    # apply the penalty computation to each element and return this list
    # return deviation_condition(line_length, slope_deviation)


def compute_tree_volume(BHD: pd.Series, height: pd.Series) -> pd.Series:
    # per extenden Denzin rule of thumb - https://www.mathago.at/wp-content/uploads/PDF/B_310.pdf
    return ((BHD.astype(int) ** 2) / 1000) * (((3 * height) + 25) / 100)


def calculate_felling_cost(
    client_range: range,
    facility_range: range,
    aij: np.ndarray,
    distance_carriage_support: np.ndarray,
    tree_volume: pd.Series,
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
            + 1.307 * tree_volume.values[fac] ** (-0.3)
            + 0.029 * 100  # the harvest intensity set to 100%
            + 0.038 * average_steepness
        )
        # add the remainder of the distance to the produced output
        if aij[cli][fac] > 15:
            min_per_cycle = min_per_cycle + (aij[cli][fac] - 15)

        hrs_per_cycle = min_per_cycle / 60
        cost_per_cycle = (
            hrs_per_cycle * 44
        )  # divide by 60 to get hrs/cycle and multiply by 44 to get cost

        x[...] = cost_per_cycle
    return productivity_cost_matrix


def logistic_growth_productivity_cost(productivity_cost: float):
    """Return the logistic growth function for the productivity cost. We grow this up to a value of 100, with a midpoint of 40 and a growth rate of 0.1"""
    return 100 / (1 + math.e ** (-0.1 * (productivity_cost - 40)))


# Optimization functions itself


def add_facility_variables(
    lscp_optimization: classes.optimization_model,
):
    """Create a list of x_i variables representing wether a facility is active

    Args:
        facility_range (_type_): _description_
        model (_type_): _description_
    """
    var_name = "x[{i}]"
    fac_vars = [
        pulp.LpVariable(
            var_name.format(i=fac), lowBound=0, upBound=1, cat=pulp.LpInteger
        )
        for fac in lscp_optimization.facility_range
    ]

    setattr(lscp_optimization.model, "fac_vars", fac_vars)

    return lscp_optimization.model


def add_facility_client_variables(lscp_optimization):
    """Create a list of variables that represent wether a given facility is assigned to a client

    Args:
        model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    var_name = "z[{i}_{j}]"
    cli_assgn_vars = [
        [
            pulp.LpVariable(
                var_name.format(i=i, j=j), lowBound=0, upBound=1, cat=pulp.LpInteger
            )
            for j in lscp_optimization.facility_range
        ]
        for i in lscp_optimization.client_range
    ]
    setattr(lscp_optimization.model, "cli_assgn_vars", cli_assgn_vars)

    return lscp_optimization.model


def add_weighted_objective_function(
    lscp_optimization,
    start_point_dict,
    obj_a_factor,
    steps,
):
    """Add the objective function to the lscp_optimization.model, compromised of two terms to minimize:
    First term: minimize cost*cli assigned to facility
    Second term: minimize the cost of factories

    Args:
        lscp_optimization.model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
        facility_cost (_type_): _description_
        obj_a_factor (float): The weight of each objective (as int), is converted here to float to represent the 0-1 range
    """
    object_a_factor = obj_a_factor / steps
    print("object a factor", 0.5 + object_a_factor)
    print("object b factor", 1.5 - object_a_factor)

    lscp_optimization.model.problem += (
        pulp.lpSum(
            (0.5 + object_a_factor)
            * (
                np.array(lscp_optimization.model.cli_assgn_vars)
                * np.array(lscp_optimization.productivity_cost)
            )
        )
        + pulp.lpSum(
            (1.5 - object_a_factor)
            * (
                np.array(lscp_optimization.model.fac_vars)
                * np.array(lscp_optimization.facility_cost)
            )
        ),
        "objective function",
    )

    return lscp_optimization.model


def add_single_objective_function(optimization_model: classes.optimization_model):
    """Add the objective function, based on the model.objective_to_select. 0 is the default, 1 is sideways slope deviations, 2 is downhill segments.
    The other objectives are ignored.
    Args:
        optimization_model (classes.optimization_model): The optimization model
    Returns:
        optimization_model (classes.optimization_model): The optimization model with the objective function added
    """
    if optimization_model.objective_to_select == 0:
        optimization_model.model.problem += (
            pulp.lpSum(
                np.array(optimization_model.model.cli_assgn_vars)
                * np.array(optimization_model.productivity_cost)
            )
            + pulp.lpSum(
                (
                    np.array(optimization_model.model.fac_vars)
                    * np.array(optimization_model.facility_cost)
                )
            ),
            "objective function",
        )
        # else only select the sideways slope deviations
    elif optimization_model.objective_to_select == 1:
        optimization_model.model.problem += pulp.lpSum(
            np.array(optimization_model.model.fac_vars)
            * np.array(optimization_model.sideways_slope_deviations_per_cable_road)
        )
        # else only select the downhill segments penalized segments
    elif optimization_model.objective_to_select == 2:
        optimization_model.model.problem += pulp.lpSum(
            np.array(optimization_model.model.fac_vars)
            * np.array(optimization_model.steep_downhill_segments)
        )

    return optimization_model.model


def add_epsilon_objective(optimization_model: classes.optimization_model):
    """
    Args:
        optimization_model (classes.optimization_model): The optimization model
    Returns:
        optimization_model (classes.optimization_model): The optimization model with the objective function added
    """
    optimization_model.model.problem += (
        pulp.lpSum(
            np.array(optimization_model.model.cli_assgn_vars)
            * np.array(optimization_model.productivity_cost)
        )
        + pulp.lpSum(
            (
                np.array(optimization_model.model.fac_vars)
                * np.array(optimization_model.facility_cost)
            )
        )
        + optimization_model.epsilon
        * pulp.lpSum(
            (optimization_model.slack_1 / optimization_model.range_1)
            + (optimization_model.slack_2 / optimization_model.range_2)
        ),
        "objective function",
    )

    return optimization_model.model


def add_singular_assignment_constraint(optimization_object: classes.optimization_model):
    """Add the constraint that the sum of facilities assigned for each client == 1 -> only one facility should be assigned to each line

    Args:
        optimization_object.model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in optimization_object.client_range:
        optimization_object.model.problem += (
            pulp.lpSum(
                [
                    optimization_object.model.cli_assgn_vars[cli][fac]
                    for fac in optimization_object.facility_range
                ]
            )
            == 1
        )

    return optimization_object.model


def add_facility_is_opened_constraint(optimization_object: classes.optimization_model):
    """Add the constraint that for each positive entry in cli_assign_vars (ie., a client is assigned to a facility),
    there should be a corresponding facility (that is, fac_vars = 1)

    Args:
        optimization_object.model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in optimization_object.client_range:
        for fac in optimization_object.facility_range:
            optimization_object.model.problem += (
                optimization_object.model.fac_vars[fac]
                - optimization_object.model.cli_assgn_vars[cli][fac]
                >= 0
            )

    return optimization_object.model


def add_epsilon_constraint(
    optimization_object: classes.optimization_model,
    target_value: float,
    objective_to_constraint: int,
):
    """Add the constraint that the objective function should be less than or equal to the target value
    Args:
        optimization_object (_type_): The optimization model
        target_value (float): The target value for the objective function to constrain to
        objective_to_constraint (int): The objective to constrain. 0 is the default, 1 is sideways slope deviations, 2 is downhill segments.

    Returns:
        optimization_object.model (_type_): The optimization model with the constraint added
    """
    if objective_to_constraint == 1:
        optimization_object.model.problem += (
            pulp.lpSum(
                np.array(optimization_object.model.fac_vars)
                * np.array(optimization_object.sideways_slope_deviations_per_cable_road)
            )
            >= target_value
        )
    elif objective_to_constraint == 2:
        optimization_object.model.problem += (
            pulp.lpSum(
                np.array(optimization_object.model.fac_vars)
                * np.array(optimization_object.steep_downhill_segments)
            )
            >= target_value
        )

    return optimization_object.model


def get_objective_values(
    optimization_object: classes.optimization_model,
    sideways_slope_deviations_max: float,
    steep_downhill_segments_max: float,
):
    """Get the objective values for the optimization model.
    The objective values are the cost, sideways slope deviations, and steep downhill segments.
    Give the true max of the objective value and return the RNI value
    Args:
        optimization_object (classes.optimization_model): The optimization model
    Returns:
        cost_objective (float): The cost objective value
        sideways_slope_deviations (float): The sideways slope deviations objective value in RNI
        steep_downhill_segments (float): The steep downhill segments objective value in RNI
    """

    fac_vars = [bool(var.value()) for var in optimization_object.model.fac_vars]

    f = lambda x: bool(x.value())
    c2f_vars = np.vectorize(f)(np.array(optimization_object.model.cli_assgn_vars))

    cost_objective = np.sum(
        c2f_vars * np.array(optimization_object.productivity_cost)
        + fac_vars * np.array(optimization_object.facility_cost)
    )

    sideways_slope_deviations = np.sum(
        fac_vars
        * np.array(optimization_object.sideways_slope_deviations_per_cable_road)
    )

    steep_downhill_segments = np.sum(
        fac_vars * np.array(optimization_object.steep_downhill_segments)
    )

    sideways_slope_deviation_RNI = (
        sideways_slope_deviations / sideways_slope_deviations_max
    ) * 100
    steep_downhill_segments_RNI = (
        steep_downhill_segments / steep_downhill_segments_max
    ) * 100

    return cost_objective, sideways_slope_deviation_RNI, steep_downhill_segments_RNI


def test_and_reassign_clis(
    fac_range, cli_range, fac_vars, cli_assgn_vars, fac_indices, aij
):
    """Ensure that the opening and assignment constraint are satisfied

    Returns:
        _type_: _description_
    """

    for i in range(fac_range):
        #  find where a cli is assigned to a fac that is not opened (sum of row is negative)
        opening_assignment_test = fac_vars[i] - cli_assgn_vars[:, i]
        if operator.contains(opening_assignment_test, -1):
            # assign all clis to the nearest opened fac:
            for j in range(cli_range):
                # skip if no facs are opened
                if len(fac_indices > 0):
                    # get the 2nd smallest distance to avoid finding distance to self
                    smallest_distance = min(aij[j, fac_indices])

                    # find its position
                    min_index = np.where(aij[j] == smallest_distance)[0]
                    # and assign this client to this one
                    random_fac = choices(min_index)
                    cli_assgn_vars[j, random_fac] = 1

    for j in range(cli_range):
        if np.sum(cli_assgn_vars[j, :]) > 1:
            # get indices of the facs this cli is assigned
            cli_assgn_indices = np.where(fac_vars == 1)[0]

            # skip if this one isnt assigned (which should not happen) or we have no more facs opened
            if len(cli_assgn_indices) < 1 or len(fac_indices) < 1:
                continue

            # get the 2nd smallest to avoid finding distance to self
            smallest_distance = min(aij[j, fac_indices])
            # find its position
            min_index = np.where(aij[j] == smallest_distance)[0]
            # and assign to one of them
            random_fac = choices(min_index)
            # random_fac = choices(cli_assgn_indices)

            # set whole row to zero
            cli_assgn_vars[j] = np.zeros(fac_range)
            # and randomly set one of those facs to 1
            cli_assgn_vars[j, random_fac] = 1

    return cli_assgn_vars, fac_vars
