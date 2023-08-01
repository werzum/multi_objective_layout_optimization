import numpy as np
import pulp
import operator
from random import randint, choices
import math

import geopandas as gpd
import pandas as pd

import pulp

from src.main import optimization_functions, geometry_operations, classes


def optimize_cable_roads(
    lscp_optimization: classes.optimization_objects,
    step: int,
    steps: int,
    start_point_dict: dict,
):
    # init the model with name and the problem - this only gives it a name and tells it to minimize the obj function

    # Add the facilities as fac_vars and facility_clients as cli_assgn_vars
    lscp_optimization = optimization_functions.add_facility_variables(lscp_optimization)
    lscp_optimization = optimization_functions.add_facility_client_variables(
        lscp_optimization
    )

    # Add the objective functions
    lscp_optimization = optimization_functions.add_moo_objective_function(
        lscp_optimization,
        start_point_dict,
        step,
        steps,
    )

    # Assignment/demand constraint - each client should
    # only be assigned to one factory
    lscp_optimization = optimization_functions.add_singular_assignment_constraint(
        lscp_optimization
    )

    # Add opening/shipping constraint - each factory that has a client assigned to it should also be opened
    lscp_optimization = optimization_functions.add_facility_is_opened_constraint(
        lscp_optimization
    )

    lscp_optimization.model = lscp_optimization.model.solve(lscp_optimization.solver)
    return lscp_optimization


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

    setup_time = math.e ** (
        1.42
        + 0.00229 * line_length
        + 0.03 * intermediate_support_height  # not available now?
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


def add_facility_variables(lscp_optimization: classes.optimization_objects):
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

    return lscp_optimization


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

    return lscp_optimization


def calculate_productivity_cost(
    client_range: range,
    facility_range: range,
    aij: np.array,
    distance_carriage_support: np.array,
    average_steepness: float,
) -> np.array:
    """Calculate the cost of each client-facility combination based on the productivity model by Gaffariyan, Stampfer, Sessions 2013

    Args:
        client_range (Range): range of clients
        facility_range (Range): range of facilities
        aij (np.array): Matrix of distances between clients and facilities
        distance_carriage_support (np.array): Distance between carriage and support
        average_steepness (float): Average steepness of the area

    Returns:
        np.array: matrix of costs for each client-facility combination
    """

    productivity_cost_matrix = np.empty([len(client_range), len(facility_range)])
    # iterate ove the matrix and calculate the cost for each entry
    it = np.nditer(
        productivity_cost_matrix, flags=["multi_index"], op_flags=["readwrite"]
    )
    for x in it:
        cli, fac = it.multi_index
        # the cost per m3 based on the productivity model by Gaffariyan, Stampfer, Sessions 2013
        x[...] = (
            0.043 * aij[cli][fac]
        )  # the distance from tree to cable road, aka lateral yarding distance
        # the yarding distance between carriage and support
        +0.007 * distance_carriage_support[cli][fac]
        +0.029 * 100  # the harvest intensity set to 100%
        +0.038 * average_steepness

    return productivity_cost_matrix


def add_moo_objective_function(
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

    lscp_optimization.model.problem += (0.5 + object_a_factor) * pulp.lpSum(
        [
            lscp_optimization.tree_cost_list[cli][fac]
            * lscp_optimization.model.cli_assgn_vars[cli][fac]
            for cli in lscp_optimization.client_range
            for fac in lscp_optimization.facility_range
        ]
    ) + (1.5 - object_a_factor) * pulp.lpSum(
        [
            lscp_optimization.model.fac_vars[fac] * lscp_optimization.facility_cost[fac]
            for fac in lscp_optimization.facility_range
        ]
        # ) + pulp.lpSum(
        #     # add a cost factor of 40 for each starting point that is selected
        #     40
        #     * np.unique(
        #         [
        #             start_point_dict[fac]
        #             for cli in client_range
        #             for fac in facility_range
        #             if bool(lscp_optimization.model.cli_assgn_vars[cli][fac].value())
        #         ]
        #     )
    ), "objective function"

    return lscp_optimization


def add_singular_assignment_constraint(lscp_optimization):
    """Add the constraint that the sum of facilities assigned for each client == 1 -> only one facility should be assigned to each line

    Args:
        lscp_optimization.model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in lscp_optimization.client_range:
        lscp_optimization.model.problem += (
            pulp.lpSum(
                [
                    lscp_optimization.model.cli_assgn_vars[cli][fac]
                    for fac in lscp_optimization.facility_range
                ]
            )
            == 1
        )

    return lscp_optimization


def add_facility_is_opened_constraint(lscp_optimization):
    """Add the constraint that for each positive entry in cli_assign_vars (ie., a client is assigned to a facility),
    there should be a corresponding facility (that is, fac_vars = 1)

    Args:
        lscp_optimization.model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in lscp_optimization.client_range:
        for fac in lscp_optimization.facility_range:
            lscp_optimization.model.problem += (
                lscp_optimization.model.fac_vars[fac]
                - lscp_optimization.model.cli_assgn_vars[cli][fac]
                >= 0
            )

    return lscp_optimization


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
