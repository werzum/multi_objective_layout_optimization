import pulp
from spopt.locate import PMedian
from pulp import LpConstraint, LpConstraintLE

import numpy as np
import pandas as pd
import geopandas as gpd

from abc import ABC, abstractmethod
from src.main import (
    geometry_operations,
    optimization_compute_quantification,
    classes_cable_road_computation,
)

# abc is a builtin module, we have to import ABC and abstractmethod


class optimization_object(ABC):  # Inherit from ABC(Abstract base class)
    model: PMedian | pulp.LpProblem
    fac_vars: list[bool]
    fac2cli: list[list[int]]
    cost_objective: float
    ecological_objective: float
    ergonomics_objective: float

    @abstractmethod
    def get_objective_values(self):
        pass

    # @abstractmethod
    # def solve(self):
    #     pass


class optimization_object_spopt(optimization_object):
    def __init__(
        self,
        name: str,
        line_gdf: gpd.GeoDataFrame,
        harvesteable_trees_gdf: gpd.GeoDataFrame,
        height_gdf: gpd.GeoDataFrame,
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

        self.facility_cost = line_gdf["line_cost"].values

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
        self.productivity_cost = (
            optimization_compute_quantification.calculate_felling_cost(
                self.client_range,
                self.facility_range,
                self.distance_tree_line,
                self.distance_carriage_support,
                self.tree_volumes_list.values,
                self.average_steepness,
            )
        )

        self.problem = pulp.LpProblem()
        self.model = PMedian(name, self.problem, self.distance_tree_line)

    def add_generic_vars_and_constraints(self):
        # Add the facilities as fac_vars and facility_clients as cli_assgn_vars
        self.model = optimization_object_spopt.add_facility_variables(self)
        self.model = optimization_object_spopt.add_facility_client_variables(self)

        # Assignment/demand constraint - each client should
        # only be assigned to one factory
        self.model = optimization_object_spopt.add_singular_assignment_constraint(self)

        # Add opening/shipping constraint - each factory that has a client assigned to it should also be opened
        self.model = optimization_object_spopt.add_facility_is_opened_constraint(self)

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

        self.model.problem += (
            self.add_cost_objective() + self.epsilon
            # THIS DOESNT MAKE SENSE; NEED TO ADD OUR VARS HERE
            * pulp.lpSum((self.slack_1 / self.range_1) + (self.slack_2 / self.range_2)),
            "objective function",
        )

        return self.model

    def add_epsilon_constraint(
        self, target_value: float, objective_to_select: int = -1
    ):
        """Adds an epsilon constraint to the model - constrain the objective to be within a certain range of
        Args:
            target_value (float): the minimum value of the objective -
            objective_to_select (int, optional): the objective to select. Defaults to -1.
        """
        if objective_to_select == 1:
            # add a named constraint to facilitate overwriting it later
            sum_deviations_variables = self.numpy_minimal_lateral_distances(
                self.ecological_penalty_lateral_distances,
                operate_on_model_vars=True,
            )

            # if constraint does not exist, add it to the problem
            if "sw_constraint" not in self.model.problem.constraints:
                self.model.problem += (
                    sum_deviations_variables <= target_value,
                    "sw_constraint",
                )

            else:  # update it
                self.model.problem.constraints["sw_constraint"] = (
                    sum_deviations_variables <= target_value
                )

        elif objective_to_select == 2:
            sum_bad_ergonomic_distances_variables = (
                self.numpy_minimal_lateral_distances(
                    self.ergonomic_penalty_lateral_distances, operate_on_model_vars=True
                )
            )

            # if constraint does not exist, add it to the problem
            if "ergo_constraint" not in self.model.problem.constraints:
                self.model.problem += LpConstraint(
                    sum_bad_ergonomic_distances_variables,
                    sense=LpConstraintLE,
                    rhs=target_value,
                    name="ergo_constraint",
                )
            else:  # update it
                self.model.problem.constraints["ergo_constraint"] = LpConstraint(
                    sum_bad_ergonomic_distances_variables,
                    sense=LpConstraintLE,
                    rhs=target_value,
                    name="ergo_constraint",
                )

        return self.model

    def remove_epsilon_constraint(self, constraint_to_delete: str):
        """
        Removes the epsilon constraint from the model
        """
        self.model.problem.constraints.pop(constraint_to_delete)

    @property
    def c2f_vars(self) -> np.ndarray:
        """Get the client to facility variables from the optimization model
        Args:
            optimization_object (classes.optimization_object): The optimization model
        Returns:
            c2f_vars (np.ndarray): The client to facility variables"""
        f = lambda x: bool(x.value())
        return np.vectorize(f)(np.array(self.model.cli_assgn_vars))

    @property
    def fac_vars(self):
        """Get the facility variables from the optimization model
        Args:
            self (classes.self): The optimization model
        Returns:
            fac_vars (list[bool]): The list of facility variables
        """
        return [bool(var.value()) for var in self.model.fac_vars]

    @property
    def cost_objective(
        self,
    ) -> float:
        """Compute the cost objective value for the optimization model
        Args:
            c2f_vars (np.ndarray): The client to facility variables
            fac_vars (list[bool]): The facility variables
            optimization_object (classes.optimization_object): The optimization model
        Returns:
            float: The cost objective value
        """
        return np.sum(self.c2f_vars * np.array(self.productivity_cost)) + np.sum(
            self.fac_vars * np.array(self.facility_cost)
        )

    @property
    def ecological_objective(self) -> float:
        """Compute the ecological_distance value for the optimization model
        Args:
            fac_vars (list[bool]): The facility variables
            optimization_object (classes.optimization_object): The optimization model
        Returns:
            float: The ecological_distance value"""

        try:
            ecological__obj = self.numpy_minimal_lateral_distances(
                self.ecological_penalty_lateral_distances
            )
        except:
            ecological__obj = 0

        return ecological__obj

    @property
    def ergonomics_objective(self) -> float:
        """Compute the ergonomics objective value for the optimization model
        Args:
            fac_vars (list[bool]): The facility variables
            self (classes.self): The optimization model
        Returns:
            float: The ergonomics objective value"""

        try:
            ergonomics_obj = self.numpy_minimal_lateral_distances(
                self.ergonomic_penalty_lateral_distances
            )
        except:
            ergonomics_obj = 0

        return ergonomics_obj

    def get_objective_values(
        self,
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

        return self.cost_objective, self.ecological_objective, self.ergonomics_objective

    def get_total_epsilon_objective_value(
        self, i_range_min_max: float, j_range_min_max: float
    ):
        """Returns the total objective value of the model, ie. the first objective plus the epsilon-scaled other objectives"""
        cost, ecological, ergonomics = self.get_objective_values()
        self.epsilon = 1
        return cost + self.epsilon * (
            (ecological / i_range_min_max) + (ergonomics / j_range_min_max)
        )

    def solve(self):
        self.model = self.model.solve(self.solver)

    def add_facility_variables(self):
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
            for fac in self.facility_range
        ]

        setattr(self.model, "fac_vars", fac_vars)

        return self.model

    def add_facility_client_variables(self):
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
                for j in self.facility_range
            ]
            for i in self.client_range
        ]
        setattr(self.model, "cli_assgn_vars", cli_assgn_vars)

        return self.model

    def add_single_objective_function(self):
        """Add the objective function, based on the model.objective_to_select. 0 is the default, 1 is ecological_distance, 2 is ergonomic segments.
        The other objectives are ignored.
        Args:
            optimization_object (classes.optimization_object): The optimization model
        Returns:
            optimization_object (classes.optimization_object): The optimization model with the objective function added
        """
        if self.objective_to_select == 0:
            self.model.problem += self.add_cost_objective()
            # else only select the ecological_distance
        elif self.objective_to_select == 1:
            self.model.problem += self.add_ecological_objective()

            # else penalize the ergonomic segments
        elif self.objective_to_select == 2:
            self.model.problem += self.add_ergonomic_objective()

        return self.model

    def add_cost_objective(self):
        return pulp.lpSum(
            np.array(self.model.cli_assgn_vars) * np.array(self.productivity_cost)
        ) + pulp.lpSum((np.array(self.model.fac_vars) * np.array(self.facility_cost)))

    def add_ecological_objective(self):
        return pulp.lpSum(
            self.numpy_minimal_lateral_distances(
                self.ecological_penalty_lateral_distances
            )
        )

    def add_ergonomic_objective(self):
        """Add the objective function for the ergonomic segments
        Args:
            self (classes.self): The optimization model
        Returns:
        """
        return pulp.lpSum(
            self.numpy_minimal_lateral_distances(
                self.ergonomic_penalty_lateral_distances
            )
        )

    def numpy_minimal_lateral_distances(
        self, set_of_distances: np.ndarray, operate_on_model_vars=False
    ):
        """Compute the minimal lateral distance for each fac var for the given set of distances
        Args:
            set_of_distances (np.ndarray): The set of distances to compute the minimal lateral distance for
        Returns:
            float: The minimal lateral distance
        """
        try:
            # if operate_on_model_vars is True, use the actual variables from the model, else use the boolean list

            if operate_on_model_vars:
                return_value = np.sum(
                    np.multiply(set_of_distances, self.model.cli_assgn_vars)
                )

            else:
                return_value = np.sum(
                    np.min(
                        set_of_distances[:, self.fac_vars],
                        axis=1,
                    )
                )

        except:
            return_value = 0

        return return_value

    def add_singular_assignment_constraint(self):
        """Add the constraint that the sum of facilities assigned for each client == 1 -> only one facility should be assigned to each line

        Args:
            optimization_object.model (_type_): _description_
            facility_range (_type_): _description_
            client_range (_type_): _description_
        """
        for cli in self.client_range:
            self.model.problem += (
                pulp.lpSum(
                    [self.model.cli_assgn_vars[cli][fac] for fac in self.facility_range]
                )
                == 1
            )

        return self.model

    def add_facility_is_opened_constraint(self):
        """Add the constraint that for each positive entry in cli_assign_vars (ie., a client is assigned to a facility),
        there should be a corresponding facility (that is, fac_vars = 1)

        Args:
            optimization_object.model (_type_): _description_
            facility_range (_type_): _description_
            client_range (_type_): _description_
        """
        for cli in self.client_range:
            for fac in self.facility_range:
                self.model.problem += (
                    self.model.fac_vars[fac] - self.model.cli_assgn_vars[cli][fac] >= 0
                )

        return self.model


class result_object(ABC):
    fac2cli: list[list[int]]
    name: str
    cable_road_objects: list[classes_cable_road_computation.Cable_Road]
    fac_vars: list[bool]
    cli_assgn_vars: list[list[bool]]


class spopt_result(result_object):
    def __init__(
        self,
        optimization_object: optimization_object,
        line_gdf: gpd.GeoDataFrame,
        name: str = "model",
    ):
        # extract the model object itself as well as the fac2cli assignment
        self.optimized_model = optimization_object.model
        self.name = name

        self.fac2cli = optimization_object.model.fac2cli
        self.fac_vars = optimization_object.fac_vars
        self.cli_assgn_vars = optimization_object.model.cli_assgn_vars

        self.selected_lines = line_gdf[self.fac_vars]
        self.cable_road_objects = self.selected_lines["Cable Road Object"]

        (
            self.cost_objective,
            self.ecological_objective,
            self.ergonomics_objective,
        ) = optimization_object.get_objective_values()


def model_results_comparison(
    result_list: list[result_object],
    line_gdf: gpd.GeoDataFrame,
    distance_tree_line: np.ndarray,
    distance_carriage_support: np.ndarray,
    productivity_cost_matrix: np.ndarray,
    tree_volumes_list: pd.Series,
):
    """Compare the results of the different models in one table
    Args:
        result_list (list): a list of the models with different tradeoffs
        productivity_cost_matrix (np.ndarray): the productivity cost matrix
        distance_tree_line (np.ndarray): the distance matrix
        distance_carriage_support (np.ndarray): the distance matrix
    """
    productivity_array = []
    distance_tree_line_array = []
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
                distance_per_this_row = distance_tree_line[row, index]
                row_sum_distance = distance_per_this_row.sum()
                row_sums.append(row_sum_distance)
        distance_tree_line_array.append(sum(row_sums))

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

        ecological_distances.append(result.ecological_objective)
        overall_ergonomic_penalty_lateral_distances.append(result.ergonomics_objective)

    overall_profit_unscaled = np.array(overall_profit)  # * profit_scaling
    profit_baseline = min(overall_profit_unscaled)
    print(f"Profit baseline is {profit_baseline}")
    profit_comparison = overall_profit_unscaled - profit_baseline

    name_list = [result.name for result in result_list]

    return pd.DataFrame(
        data={
            "Total distance of trees to cable roads": distance_tree_line_array,
            "Productivity cost per m3 as per Stampfer": productivity_array,
            "Total distance from carriage to support": distance_carriage_support_array,
            "overall_profit": overall_profit,
            "cable_road_costs": cable_road_costs,
            "profit_comparison": profit_comparison,
            "name": name_list,
            "ecological_distances": ecological_distances,
            "ergonomics_distances": overall_ergonomic_penalty_lateral_distances,
        }
    )
