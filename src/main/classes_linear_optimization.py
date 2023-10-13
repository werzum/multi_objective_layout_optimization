from itertools import pairwise
import pulp
from spopt.locate import PMedian
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.mutation import Mutation

import numpy as np
import geopandas as gpd

from abc import ABC, abstractmethod

# abc is a builtin module, we have to import ABC and abstractmethod


class optimization_object(ABC):  # Inherit from ABC(Abstract base class)
    @abstractmethod
    def __init__():
        pass

    @abstractmethod  # Decorator to define an abstract method
    def get_objective_values(self):
        pass

    @abstractmethod
    def get_fac_vars(self):
        pass

    @abstractmethod
    def get_fac2cli(self):
        pass

    @abstractmethod
    def get_objective_values(self):
        return optimization_functions.get_objective_values(self)

    @abstractmethod
    def solve(self):
        self.model = self.model.solve(self.solver)

    @property
    def facility_range(self):
        return range(self.distance_tree_line.shape[1])

    @property
    def client_range(self):
        return range(self.distance_tree_line.shape[0])

    @property
    def aij(self):
        return self.distance_tree_line

    @property
    def facility_cost(self):
        return self.facility_points_gdf.line_cost.values

    @property
    def tree_volumes_list(self):
        return self.harvesteable_trees_gdf["cubic_volume"]

    @property
    def average_steepness(self):
        return geometry_operations.compute_average_terrain_steepness(
            self.line_gdf, self.height_gdf
        )


class optimization_object_soo(optimization_object):
    def __init__(
        self,
        name: str,
        line_gdf: gpd.GeoDataFrame,
        harvesteable_trees_gdf: gpd.GeoDataFrame,
        height_gdf: gpd.GeoDataFrame,
        slope_line: LineString,
        objective_to_select: int = -1,
    ):
        # Create a matrix with the distance between every tree and line and the distance between the support (beginning of the CR) and the carriage
        # (cloests point on the CR to the tree)
        (
            self.distance_tree_line,
            self.distance_carriage_support,
        ) = geometry_operations.compute_distances_facilities_clients(
            harvesteable_trees_gdf, line_gdf
        )

        self.objective_to_select = objective_to_select

        # sort the facility (=lines) and demand points (=trees)
        self.facility_points_gdf = line_gdf.reset_index()
        self.demand_points_gdf = harvesteable_trees_gdf.reset_index()

        # set up the solver
        self.solver = pulp.PULP_CBC_CMD(msg=False, warmStart=False)
        self.name = "model"

        # create the nr of possible facilities and clients
        self.client_range = range(self.distance_tree_line.shape[0])
        self.facility_range = range(self.distance_tree_line.shape[1])

        self.facility_cost = line_gdf.line_cost.values
        self.aij = self.distance_tree_line

        # collect the matrices needed for the optimization
        self.tree_volumes_list = harvesteable_trees_gdf["cubic_volume"]
        self.angle_between_supports_list = line_gdf["angle_between_supports"]

        self.average_steepness = geometry_operations.compute_average_terrain_steepness(
            line_gdf, height_gdf
        )

        # # calculate the deviations of the subsegments of the cable road from the slope line
        # self.sideways_slope_deviations_per_cable_road = (
        #     optimization_functions.compute_cable_road_deviations_from_slope(
        #         line_gdf, slope_line
        #     )
        # )

        # calulate the environmental impact of each line beyond 10m lateral distance
        ecological_penalty_threshold = 10
        self.ecological_penalty_lateral_distances = np.where(
            self.distance_tree_line > ecological_penalty_threshold,
            self.distance_tree_line - ecological_penalty_threshold,
            0,
        )

        # double all distances greater than penalty_treshold
        ergonomics_penalty_treshold = 15
        self.ergonomic_penalty_lateral_distances = np.where(
            self.distance_tree_line > ergonomics_penalty_treshold,
            (self.distance_tree_line - ergonomics_penalty_treshold) * 2,
            0,
        )

        # and the productivity cost combination of each line combination
        self.productivity_cost = optimization_functions.calculate_felling_cost(
            self.client_range,
            self.facility_range,
            self.aij,
            self.distance_carriage_support,
            self.tree_volumes_list.values,
            self.average_steepness,
        )

        self.problem = pulp.LpProblem()
        self.model = PMedian(name, self.problem, self.aij)

    def add_generic_vars_and_constraints(self):
        # Add the facilities as fac_vars and facility_clients as cli_assgn_vars
        self.model = optimization_functions.add_facility_variables(self)
        self.model = optimization_functions.add_facility_client_variables(self)

        # Assignment/demand constraint - each client should
        # only be assigned to one factory
        self.model = optimization_functions.add_singular_assignment_constraint(self)

        # Add opening/shipping constraint - each factory that has a client assigned to it should also be opened
        self.model = optimization_functions.add_facility_is_opened_constraint(self)

    def add_epsilon_objective(
        self,
        i_slack: float,
        j_slack: float,
        i_range: range,
        j_range: range,
    ):
        """Adds an epsilon objective to the model to minimize the first objective and further minimize the epsilon-scaled other objectives"""
        self.epsilon = 1
        self.slack_1 = i_slack
        self.slack_2 = j_slack
        # get the range (as in from .. to ..) of each objective
        self.range_1 = i_range.max() - i_range.min()
        self.range_2 = j_range.max() - j_range.min()

        self.model = optimization_functions.add_epsilon_objective(self)

    def add_epsilon_constraint(
        self, target_value: float, objective_to_select: int = -1
    ):
        """Adds an epsilon constraint to the model - constrain the objective to be within a certain range of
        Args:
            target_value (float): the minimum value of the objective -
            objective_to_select (int, optional): the objective to select. Defaults to -1.
        """
        self.model = optimization_functions.add_epsilon_constraint(
            self, target_value, objective_to_select
        )

    def get_objective_values(self):
        return optimization_functions.get_objective_values(self)

    def get_total_epsilon_objective_value(
        self, i_range_min_max: float, j_range_min_max: float
    ):
        """Returns the total objective value of the model, ie. the first objective plus the epsilon-scaled other objectives"""
        cost, ecological, ergonomics = optimization_functions.get_objective_values(self)
        self.epsilon = 1
        return cost + self.epsilon * (
            (ecological / i_range_min_max) + (ergonomics / j_range_min_max)
        )

    def add_single_objective(self):
        self.model = optimization_functions.add_single_objective_function(self)

    def solve(self):
        self.model = self.model.solve(self.solver)




def add_facility_variables(
    lscp_optimization: classes.optimization_object,
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


def add_single_objective_function(optimization_object: classes.optimization_object):
    """Add the objective function, based on the model.objective_to_select. 0 is the default, 1 is ecological_distance, 2 is ergonomic segments.
    The other objectives are ignored.
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        optimization_object (classes.optimization_object): The optimization model with the objective function added
    """
    if optimization_object.objective_to_select == 0:
        optimization_object.model.problem += add_cost_objective(optimization_object)
        # else only select the ecological_distance
    elif optimization_object.objective_to_select == 1:
        optimization_object.model.problem += add_ecological_objective(
            optimization_object
        )

        # else penalize the ergonomic segments
    elif optimization_object.objective_to_select == 2:
        optimization_object.model.problem += add_ergonomic_objective(
            optimization_object
        )

    return optimization_object.model


def add_cost_objective(optimization_object: classes.optimization_object):
    return pulp.lpSum(
        np.array(optimization_object.model.cli_assgn_vars)
        * np.array(optimization_object.productivity_cost)
    ) + pulp.lpSum(
        (
            np.array(optimization_object.model.fac_vars)
            * np.array(optimization_object.facility_cost)
        )
    )


def add_ecological_objective(optimization_object: classes.optimization_object):
    return pulp.lpSum(
        np.array(optimization_object.model.fac_vars)
        * np.array(optimization_object.ecological_penalty_lateral_distances)
    )


def add_ergonomic_objective(optimization_object: classes.optimization_object):
    """Add the objective function for the ergonomic segments
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
    """
    return pulp.lpSum(
        np.array(optimization_object.model.fac_vars)
        * np.array(optimization_object.ergonomic_penalty_lateral_distances)
    )


def add_epsilon_objective(optimization_object: classes.optimization_object):
    """
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        optimization_object (classes.optimization_object): The optimization model with the objective function added
    """
    optimization_object.model.problem += (
        add_cost_objective(optimization_object)
        + optimization_object.epsilon
        * pulp.lpSum(
            (optimization_object.slack_1 / optimization_object.range_1)
            + (optimization_object.slack_2 / optimization_object.range_2)
        ),
        "objective function",
    )

    return optimization_object.model


def add_singular_assignment_constraint(
    optimization_object: classes.optimization_object,
):
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


def add_facility_is_opened_constraint(optimization_object: classes.optimization_object):
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
    optimization_object: classes.optimization_object,
    target_value: float,
    objective_to_constraint: int,
):
    """Add the constraint that the objective function should be less than or equal to the target value
    Args:
        optimization_object (_type_): The optimization model
        target_value (float): The target value for the objective function to constrain to
        objective_to_constraint (int): The objective to constrain. 0 is the default, 1 is ecological_distance, 2 is bad ergonomic segments.

    Returns:
        optimization_object.model (_type_): The optimization model with the constraint added
    """
    if objective_to_constraint == 1:
        # add a named constraint to facilitate overwriting it later
        sum_deviations_variables = pulp.lpSum(
            np.array(optimization_object.model.fac_vars)
            * np.array(optimization_object.ecological_penalty_lateral_distances)
        )

        # if constraint does not exist, add it to the problem
        if "sw_constraint" not in optimization_object.model.problem.constraints:
            optimization_object.model.problem += LpConstraint(
                sum_deviations_variables,
                sense=LpConstraintLE,
                rhs=target_value,
                name="sw_constraint",
            )
        else:  # update it
            optimization_object.model.problem.constraints[
                "sw_constraint"
            ] = LpConstraint(
                sum_deviations_variables,
                sense=LpConstraintLE,
                rhs=target_value,
                name="sw_constraint",
            )

    elif objective_to_constraint == 2:
        sum_bad_ergonomic_distances_variables = pulp.lpSum(
            np.array(optimization_object.model.fac_vars)
            * np.array(optimization_object.ergonomic_penalty_lateral_distances)
        )

        # if constraint does not exist, add it to the problem
        if "ergo_constraint" not in optimization_object.model.problem.constraints:
            optimization_object.model.problem += LpConstraint(
                sum_bad_ergonomic_distances_variables,
                sense=LpConstraintLE,
                rhs=target_value,
                name="ergo_constraint",
            )
        else:  # update it
            optimization_object.model.problem.constraints[
                "ergo_constraint"
            ] = LpConstraint(
                sum_bad_ergonomic_distances_variables,
                sense=LpConstraintLE,
                rhs=target_value,
                name="ergo_constraint",
            )

    return optimization_object.model


def get_secondary_objective_values_with_fac2cli(
    fac2cli: list, ecological_distance, bad_ergonomic_distance
):
    fac_vars = [True if entry else False for entry in fac2cli]

    ecological_distance_here = np.sum(fac_vars * np.array(ecological_distance))
    bad_ergonomic_distance_here = np.sum(
        np.min(
            bad_ergonomic_distance[:, np.array(fac_vars)],
            axis=1,
        )
    )

    bad_ergonomic_distance_here = np.sum(
        np.min(
            bad_ergonomic_distance[:, np.array(fac_vars)],
            axis=1,
        )
    )

    return ecological_distance_here, bad_ergonomic_distance_here


def get_objective_values(
    optimization_object: classes.optimization_object,
):
    """Get the objective values for the optimization model.
    The objective values are the cost, ecological, and ergonomically bad segments.
    Give the true max of the objective value and return the RNI value
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        cost_objective (float): The cost objective value
        ecological_distance (float): The ecological_distance bjective value in RNI
        bad_ergonomic_distance (float): The ergonomically bad segments objective value in RNI
    """

    fac_vars = get_fac_vars(optimization_object)
    c2f_vars = get_c2f_vars(optimization_object)

    cost_objective = compute_cost_objective(c2f_vars, fac_vars, optimization_object)
    ecological_distance = compute_ecological_objective(fac_vars, optimization_object)
    bad_ergonomic_distance = compute_ergonomics_objective(fac_vars, optimization_object)

    return cost_objective, ecological_distance, bad_ergonomic_distance


def get_c2f_vars(optimization_object: classes.optimization_object) -> np.ndarray:
    """Get the client to facility variables from the optimization model
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        c2f_vars (np.ndarray): The client to facility variables"""
    f = lambda x: bool(x.value())
    return np.vectorize(f)(np.array(optimization_object.model.cli_assgn_vars))


def get_fac_vars(optimization_object: classes.optimization_object):
    """Get the facility variables from the optimization model
    Args:
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        fac_vars (list[bool]): The list of facility variables
    """
    return [bool(var.value()) for var in optimization_object.model.fac_vars]


def compute_cost_objective(
    c2f_vars: np.ndarray,
    fac_vars: list[bool],
    optimization_object: classes.optimization_object,
) -> float:
    """Compute the cost objective value for the optimization model
    Args:
        c2f_vars (np.ndarray): The client to facility variables
        fac_vars (list[bool]): The facility variables
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        float: The cost objective value
    """
    return np.sum(c2f_vars * np.array(optimization_object.productivity_cost)) + np.sum(
        fac_vars * np.array(optimization_object.facility_cost)
    )


def compute_ecological_objective(
    fac_vars: list[bool], optimization_object: classes.optimization_object
) -> float:
    """Compute the ecological_distance value for the optimization model
    Args:
        fac_vars (list[bool]): The facility variables
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        float: The ecological_distance value"""

    try:
        ecological__obj = np.sum(
            np.min(optimization_object.ecological_penalty_lateral_distances)[
                :, fac_vars
            ],
            axis=1,
        )
    except:
        ecological__obj = 0

    return ecological__obj


def compute_ergonomics_objective(
    fac_vars: list[bool], optimization_object: classes.optimization_object
) -> float:
    """Compute the ergonomics objective value for the optimization model
    Args:
        fac_vars (list[bool]): The facility variables
        optimization_object (classes.optimization_object): The optimization model
    Returns:
        float: The ergonomics objective value"""

    try:
        ergonomics_obj = np.sum(
            np.min(optimization_object.ergonomic_penalty_lateral_distances)[
                :, fac_vars
            ],
            axis=1,
        )
    except:
        ergonomics_obj = 0

    return ergonomics_obj


class result_object(ABC):

    @abstractmethod
    def __init__(self):
        pass
    
    @property
    def fac2cli(self):
        return self.fac2cli

    @property
    def optimized_model(self):
        return self.optimized_model

    @property
    def selected_lines(self):
        # compute the number of supports
        self.selected_lines["number_int_supports"] = [
            len(list(cable_road.get_all_subsegments())) - 1
            if list(cable_road.get_all_subsegments())
            else 0
            for cable_road in self.cable_road_objects
        ]
        return self.selected_lines.copy()
    
    @property
    def name(self):
        return self.name


    def model_results_comparison(
        result_list: list[classes_cable_road_computation.optimization_result],
        line_gdf: gpd.GeoDataFrame,
        aij: np.ndarray,
        distance_carriage_support: np.ndarray,
        productivity_cost_matrix: np.ndarray,
        tree_volumes_list: pd.Series,
        ecological_penalty_lateral_distances: pd.Series,
        ergonomic_penalty_lateral_distances: np.ndarray,
    ):
        """Compare the results of the different models in one table
        Args:
            result_list (list): a list of the models with different tradeoffs
            productivity_cost_matrix (np.ndarray): the productivity cost matrix
            aij (np.ndarray): the distance matrix
            distance_carriage_support (np.ndarray): the distance matrix
        """
        productivity_array = []
        aij_array = []
        distance_carriage_support_array = []
        overall_profit = []
        cable_road_costs = []
        facility_cost = line_gdf["line_cost"].values

        ecological_distances = []
        overall_ergonomic_penalty_lateral_distances = []

        total_profit_per_layout_baseline = 0
        for index, row in enumerate(result_list[0].fac2cli):
            if row:
                profit_per_row = tree_volumes_list.iloc[row] * 80
                profit_this_cr = profit_per_row.sum()
                total_profit_per_layout_baseline += profit_this_cr

        for result in result_list:
            # and the corresponding rows from the distance matrix
            row_sums = []
            for index, row in enumerate(result.fac2cli):
                if row:
                    distance_per_this_row = aij[row, index]
                    row_sum_distance = distance_per_this_row.sum()
                    row_sums.append(row_sum_distance)
            aij_array.append(sum(row_sums))

            row_sums = []
            for index, row in enumerate(result.fac2cli):
                if row:
                    productivity_per_row = productivity_cost_matrix[row, index]
                    row_sum_distance = productivity_per_row.sum()
                    row_sums.append(row_sum_distance)
            productivity_array.append(sum(row_sums))

            row_sums = []
            for index, row in enumerate(result.fac2cli):
                if row:
                    distance_per_this_row = distance_carriage_support[row, index]
                    row_sum_distance = distance_per_this_row.sum()
                    row_sums.append(row_sum_distance)
            distance_carriage_support_array.append(sum(row_sums))

            # subtract the productivity cost from the total profit
            overall_profit.append(total_profit_per_layout_baseline - productivity_array[-1])

            # get the total cable road costs
            total_cable_road_costs = 0
            for index, row in enumerate(result.fac2cli):
                if row:
                    cable_road_cost = facility_cost[index]
                    total_cable_road_costs += cable_road_cost
            cable_road_costs.append(total_cable_road_costs)

            (
                ecological_distances_here,
                bad_ergonomic_distance_here,
            ) = optimization_compute_quantification.get_secondary_objective_values_with_fac2cli(
                result.fac2cli,
                ecological_penalty_lateral_distances,
                ergonomic_penalty_lateral_distances,
            )

            ecological_distances.append(ecological_distances_here)
            overall_ergonomic_penalty_lateral_distances.append(bad_ergonomic_distance_here)

        overall_profit_unscaled = np.array(overall_profit)  # * profit_scaling
        profit_baseline = min(overall_profit_unscaled)
        print(f"Profit baseline is {profit_baseline}")
        profit_comparison = overall_profit_unscaled - profit_baseline

        name_list = [result.name for result in result_list]

        return pd.DataFrame(
            data={
                "Total distance of trees to cable roads": aij_array,
                "Productivity cost per m3 as per Stampfer": productivity_array,
                "Total distance from carriage to support": distance_carriage_support_array,
                "overall_profit": overall_profit,
                "cable_road_costs": cable_road_costs,
                "profit_comparison": profit_comparison,
                "name": name_list,
                "ecological_distances": ecological_distances,
                "bad_ergonomics_distance": overall_ergonomic_penalty_lateral_distances,
            }
        )



class optimization_result:


class SOO_result(result_object):
        def __init__(
        self,
        optimization_object: optimization_object,
        line_gdf: gpd.GeoDataFrame,
        selection_index: int,
        print_results: bool = False,
        name: str = "model",
        ):
        # extract the model object itself as well as the fac2cli assignment
        if hasattr(optimization_object, "model"):
            self.optimized_model = optimization_object.model
            # extract lines and CR objects
            (
                self.selected_lines,
                self.cable_road_objects,
            ) = helper_functions.model_to_line_gdf(optimization_object, line_gdf)
            self.fac2cli = optimization_object.model.fac2cli

class MOO_result(result_object):

    def __init__(
        self,
        optimization_object: optimization_object):
            
        self.optimized_model = optimization_object
        # the cli2fac
        X = optimization_object.X
        # the objectives
        F = optimization_object.F

        # reshape and transpose the var matrices to get the fac2cli format
        variable_matrix = X[selection_index].reshape(
            (
                optimization_object.problem.client_range + 1,
                optimization_object.problem.facility_range,
            )
        )
        # transpose the variable matrix to the fac2cli format and then get the indices of the selected lines
        fac2cli = variable_matrix[:-1].T
        self.fac2cli = [np.where(row)[0].tolist() for row in fac2cli]

        # add the fac vars to the optimized model
        self.optimized_model.fac_vars = [
            True if entry else False for entry in self.fac2cli
        ]

        # also extract lines and CR objects
        (
            self.selected_lines,
            self.cable_road_objects,
        ) = helper_functions.model_to_line_gdf(self.optimized_model, line_gdf)

