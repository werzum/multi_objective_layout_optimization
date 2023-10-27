from pymoo.core.sampling import Sampling
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.mutation import Mutation
from pymoo.core.crossover import Crossover

import geopandas as gpd
import numpy as np
from random import randint

from src.main import classes_linear_optimization


class optimization_object_pymoo(
    classes_linear_optimization.optimization_object, ElementwiseProblem
):
    def __init__(
        self,
        distance_tree_line,
        productivity_cost,
        facility_cost,
        ecological_penalty_lateral_distances: np.ndarray,
        ergonomic_penalty_lateral_distances: np.ndarray,
        **kwargs
    ):
        self.distance_tree_line = distance_tree_line

        # create the nr of possible facilities and clients
        self.client_range = distance_tree_line.shape[0]
        self.facility_range = distance_tree_line.shape[1]

        self.productivity_cost = productivity_cost
        self.facility_cost = facility_cost

        self.ecological_penalty_lateral_distances = ecological_penalty_lateral_distances
        self.ergonomic_penalty_lateral_distances = ergonomic_penalty_lateral_distances

        # pymoo properties
        self.epsilon = 0.1
        self.n_var = self.client_range * self.facility_range + self.facility_range

        ElementwiseProblem.__init__(
            self,
            n_var=self.n_var,
            n_obj=3,
            n_eq_constr=self.client_range,
            n_ieq_constr=self.client_range,
            xl=0,
            xu=1,
            vtype=int,
            **kwargs,
        )

    def compute_sum_lateral_distances(self, distances):
        """Compute the sum of the lateral distances for the given facility variables
        Args:
            distances (np.ndarray): Array containing the lateral distances.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
        Returns:
            float: Sum of the lateral distances.
        """
        try:
            obj_here = np.sum(
                np.min(
                    distances[:, np.array(self.fac_vars).astype(bool)],
                    axis=1,
                )
            )
        except:
            obj_here = 0

        return obj_here

    def get_objective_values(self) -> tuple[float, float, float]:
        # overall_distance_obj = np.sum(cli_assgn_vars * problem.distance_tree_line)
        overall_cost_obj = np.sum(self.fac_vars * self.facility_cost) + np.sum(
            self.cli_assgn_vars * self.productivity_cost
        )

        ecological_obj = self.compute_sum_lateral_distances(
            self.ecological_penalty_lateral_distances
        )
        ergonomics_obj = self.compute_sum_lateral_distances(
            self.ergonomic_penalty_lateral_distances
        )

        return overall_cost_obj, ecological_obj, ergonomics_obj

    def get_total_objective_value(self):
        """Get the combined objective value as per AUGMECON"""

        overall_cost_obj, ecological_obj, ergonomics_obj = self.get_objective_values()

        return (
            overall_cost_obj + ecological_obj + ergonomics_obj
        )  # self.epsilon * (ecological_obj + ergonomics_obj)

    @property
    def aij(self):
        return self.distance_tree_line

    @property
    def fac_vars(self):
        variable_matrix = self.x.reshape((self.client_range + 1, self.facility_range))
        return variable_matrix[-1]

    @property
    def cli_assgn_vars(self):
        variable_matrix = self.x.reshape((self.client_range + 1, self.facility_range))
        return variable_matrix[:-1]

    def set_fac_vars(self, fac_vars):
        variable_matrix = self.x.reshape((self.client_range + 1, self.facility_range))
        variable_matrix[-1] = fac_vars
        self.x = variable_matrix.flatten()

    def set_cli_assgn_vars(self, cli_assgn_vars):
        variable_matrix = self.x.reshape((self.client_range + 1, self.facility_range))
        variable_matrix[:-1] = cli_assgn_vars
        self.x = variable_matrix.flatten()

    def reassign_clients(
        self,
        fac_indices: np.ndarray,
    ):
        """
        Reassign clients to the closest open facilities.

        Args:
            problem ('NSGA2Problem'): The problem instance containing cost and facility data.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
            cli_assgn_vars (np.ndarray): Binary array representing client assignments to facilities.
            fac_indices (np.ndarray): Array containing indices of open facilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing updated cli_assgn_vars and fac_vars.
        """
        # Find the positions of the closest facilities for each client
        min_indices = np.argmin(self.distance_tree_line[:, fac_indices], axis=1)

        # Create an array for the updated client assignments
        updated_cli_assgn_vars = np.zeros_like(self.cli_assgn_vars)

        # Use numpy fancy indexing to update the client assignments efficiently
        rows = np.arange(self.client_range)
        cols = fac_indices[min_indices]
        updated_cli_assgn_vars[rows, cols] = 1

        fac_vars_updated = np.zeros(len(self.fac_vars))
        fac_vars_updated[fac_indices] = 1

        self.set_cli_assgn_vars(updated_cli_assgn_vars)
        self.set_fac_vars(fac_vars_updated)

    def _evaluate(self, x, out, *args, **kwargs):
        # reshape n_var to (n_trees*n_facs+n_facs)
        variable_matrix = x.reshape((self.client_range + 1, self.facility_range))
        self.set_cli_assgn_vars(variable_matrix[:-1])
        self.set_fac_vars(variable_matrix[-1])

        (
            cost_obj,
            ecological_obj,
            ergonomic_obj,
        ) = self.get_objective_values()

        # for each row sum should be 1 -> only one client allowed
        singular_assignment_constr = np.sum(self.cli_assgn_vars, axis=1) - 1

        # want to enforce that for each row where a 1 exists, fac_vars also has a 1
        facility_is_opened_constr = -np.sum(self.fac_vars - self.cli_assgn_vars, axis=1)

        out["F"] = np.column_stack([cost_obj, ecological_obj, ergonomic_obj])
        # ieq constr
        out["G"] = np.column_stack([facility_is_opened_constr])
        # eq constr
        out["H"] = np.column_stack([singular_assignment_constr])


class MyRepair(Repair):
    def _do(self, optimization_object, x, **kwargs):
        buffer = []  # Create a buffer to store mutated solutions
        x_shape = x.shape[0]  # Get the number of solutions in 'x'

        for j in range(x_shape):
            optimization_object.x = x[j]

            # # Reshape the solution 'x[j]' into a matrix with 'optimization_object.client_range + 1' rows and 'optimization_object.facility_range' columns
            # variable_matrix = x[j].reshape(
            #     (
            #         optimization_object.client_range + 1,
            #         optimization_object.facility_range,
            #     )
            # )
            # cli_assgn_vars = variable_matrix[
            #     :-1
            # ]  # Extract the rows except the last one (client assignment variables)
            # fac_vars = variable_matrix[-1]  # Extract the last row (facility variables)

            # get indices of open facs
            fac_indices = np.where(optimization_object.fac_vars == 1)[0]

            if fac_indices.any():
                # reassign all clis to open facs
                optimization_object.reassign_clients(fac_indices)

            # append this solution
            buffer.append(optimization_object.x)

        x = np.array(buffer).reshape(
            (
                x.shape[0],
                (optimization_object.client_range + 1)
                * optimization_object.facility_range,
            )
        )
        return x


class CustomSampling(Sampling):
    """Custom sampling with one open fac for the start configuration"""

    def _do(self, optimization_object, n_samples, **kwargs):
        # initially zero array
        vars = np.zeros(
            (optimization_object.client_range + 1, optimization_object.facility_range)
        )

        # Randomly open a facility (set its variable to 1)
        factory_to_open = randint(0, optimization_object.facility_range - 1)
        # set all clients to this fac (ie all rows)
        vars[:, factory_to_open] = 1
        # and open this fac
        vars[-1, factory_to_open] = 1

        vars = vars.flatten()

        # repeat this for all samples
        return np.vstack([vars] * n_samples)


class CustomCrossOver(Crossover):
    """Custom crossover to iteratively select cable roads from our fit configurations"""

    def __init__(self):
        super().__init__(2, 2)

    def _do(
        self, problem, X, **kwargs
    ):  # , n_parents, n_offsprings, optimization_object, n_samples, **kwargs):
        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):
            # get the first and the second parent
            a, b = X[0, k], X[1, k]

            a_reshape = a.reshape((problem.client_range + 1, problem.facility_range))
            fac_vars_a = a_reshape[-1]

            b_reshape = b.reshape((problem.client_range + 1, problem.facility_range))
            fac_vars_b = b_reshape[-1]

            # create the offspring in the same shape as the parents
            offspring_a = np.zeros(fac_vars_a.shape)
            offspring_b = np.zeros(fac_vars_b.shape)

            # randomly select their own or the other parent's fac vars
            for i in range(len(fac_vars_a)):
                if np.random.random() < 0.5:
                    offspring_a[i] = fac_vars_a[i]
                    offspring_b[i] = fac_vars_b[i]
                else:
                    offspring_a[i] = fac_vars_b[i]
                    offspring_b[i] = fac_vars_a[i]

            # set the new fac vars for the offspring
            a_reshape[-1] = offspring_a
            b_reshape[-1] = offspring_b

            # and flatten them
            a = a_reshape.flatten()
            b = b_reshape.flatten()

            Y[0, k], Y[1, k] = a, b

        return Y


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def remove_facility(
        self,
        optimization_object: optimization_object_pymoo,
        t: float,
    ):
        """
        Randomly removes a facility and reassigns clients to other open facilities.

        Args:
            optimization_object ('SupportLinesoptimization_object'): The optimization_object instance containing cost and facility data.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
            cli_assgn_vars (np.ndarray): Binary array representing client assignments to facilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing updated cli_assgn_vars and fac_vars.
        """
        # duplicate code, but dont see how to remove this if we fork in different ways during the loop
        for _ in range(10):
            # Get the indices of open facilities
            fac_indices = np.where(optimization_object.fac_vars == 1)[0]
            if len(fac_indices) == 0:
                break  # No open facilities, no mutation possible

            # Randomly choose a facility to delete
            fac_to_delete = np.random.choice(fac_indices)
            modified_fac_vars = optimization_object.fac_vars.copy()
            modified_fac_vars[fac_to_delete] = 0
            modified_fac_indices = np.where(modified_fac_vars == 1)[0]

            # Create a boolean mask for the condition cli_assgn_vars[j, fac_to_delete] == 1
            mask = optimization_object.cli_assgn_vars[:, fac_to_delete] == 1
            # Use the mask for boolean indexing and set the corresponding elements to 0
            modified_cli_assgn_vars = optimization_object.cli_assgn_vars.copy()
            modified_cli_assgn_vars[mask, fac_to_delete] = 0

            objective_value_before = optimization_object.get_total_objective_value()

            # Reassign clients to the closest open facilities
            optimization_object.reassign_clients(modified_fac_indices)

            objective_value_after = optimization_object.get_total_objective_value()

            # Check if this mutation decreased the objective function
            if self.metropolis_decision(
                objective_value_after, objective_value_before, t
            ):
                break  # Mutation improved the objective, stop trying
            else:
                # Undo this mutation and keep trying other facilities
                # Reassign clients again to their original facility (using precomputed values)
                optimization_object.reassign_clients(fac_indices)

    def add_facility(
        self,
        optimization_object: optimization_object_pymoo,
        # fac_vars: np.ndarray,
        # cli_assgn_vars: np.ndarray,
        t: float,
    ):
        """
        Randomly opens a facility and reassigns clients to the closest open facilities.

        Args:
            optimization_object (SupportLinesoptimization_object): The optimization_object instance containing cost and facility data.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
            cli_assgn_vars (np.ndarray): Binary array representing client assignments to facilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing updated cli_assgn_vars and fac_vars.
        """
        for _ in range(1):
            # Get the objective value before the mutation
            objective_value_before = optimization_object.get_total_objective_value()

            # Randomly open a facility (set its variable to 1)
            factory_to_open = randint(0, optimization_object.facility_range - 1)
            modified_fac_vars = optimization_object.fac_vars.copy()
            modified_fac_vars[factory_to_open] = 1

            fac_indices = np.where(modified_fac_vars == 1)[
                0
            ]  # Get the indices of open facilities

            # Reassign clients to the closest open facilities
            if len(fac_indices) > 0:
                # cli_assgn_vars, fac_vars =
                optimization_object.reassign_clients(fac_indices)

                # Get the objective value after the mutation
                objective_value_after = optimization_object.get_total_objective_value()

                # Check if this mutation decreased the objective function or if the metropolis criterion is fulfilled
                # we can accept a worse solution with decreasing chance
                if self.metropolis_decision(
                    objective_value_after, objective_value_before, t
                ):
                    break  # Mutation improved the objective, stop trying
                else:
                    # Undo this mutation and keep trying other facilities
                    modified_fac_vars[factory_to_open] = 0
                    optimization_object.set_fac_vars(modified_fac_vars)
                    # # Reassign clients again to their original facility
                    # cli_assgn_vars, fac_vars = optimization_object.reassign_clients(
                    #     optimization_object, fac_vars, cli_assgn_vars, fac_indices
        #             )

        # return cli_assgn_vars, fac_vars

    # def get_fac_cli_assgn_vars(self, optimization_object):
    #     # Reshape the solution 'x[j]' into a matrix with 'optimization_object.client_range + 1' rows and 'optimization_object.facility_range' columns

    #     variable_matrix = x.reshape(
    #         (optimization_object.client_range + 1, optimization_object.facility_range)
    #     )
    #     cli_assgn_vars = variable_matrix[
    #         :-1
    #     ]  # Extract the rows except the last one (client assignment variables)

    #     fac_vars = variable_matrix[-1]  # Extract the last row (facility variables)

    #     return fac_vars, cli_assgn_vars

    def metropolis_decision(
        self, objective_value_after: float, objective_value_before: float, t: float
    ) -> bool:
        """Return True if we accept the mutation"""
        metroplis_criterion = np.exp(
            -(objective_value_after - objective_value_before) / t
        )

        # Check if this mutation decreased the objective function or if the metropolis criterion is fulfilled
        # we can accept a worse solution with decreasing chance
        if (
            objective_value_after < objective_value_before
            or metroplis_criterion > np.random.uniform()
        ):
            return True
        else:
            return False

    def _do(
        self, optimization_object: optimization_object_pymoo, x: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Applies the mutation operator to the solutions in 'x'.

        Args:
            optimization_object (SupportLinesoptimization_object): The optimization_object instance containing cost and facility data.
            x (np.ndarray): Array of solutions to be mutated.

        Returns:
            np.ndarray: Mutated solutions.
        """
        buffer = []  # Create a buffer to store mutated solutions
        x_shape = x.shape[0]  # Get the number of solutions in 'x'

        temperature = 1
        iteration = kwargs["algorithm"].n_gen
        t = temperature / iteration

        for j in range(x_shape):
            optimization_object.x = x[
                j
            ]  # update the x in the optimization object to be this specific x out of all x's
            # fac_vars, cli_assgn_vars = self.get_fac_cli_assgn_vars(
            #     optimization_object, x, j
            # )

            for _ in range(10):
                fac_indices = np.where(optimization_object.fac_vars == 1)[0]

                # add a facility if there are none - we always need at least one
                if len(fac_indices) == 1:
                    self.add_facility(optimization_object, t)
                else:
                    # Try to remove a facility that decreases the objective value or add one if not possible
                    if randint(0, 1):
                        self.remove_facility(optimization_object, t)
                    else:
                        self.add_facility(optimization_object, t)

            buffer.append(optimization_object.x)  # Store the mutated solution

        # Reshape the buffer into the same shape as 'x'
        x = np.array(buffer).reshape(
            (
                x_shape,
                (optimization_object.client_range + 1)
                * optimization_object.facility_range,
            )
        )
        return x  # Return the mutated solutions


class pymoo_result(classes_linear_optimization.result_object):
    def __init__(
        self,
        optimization_object: optimization_object_pymoo,
        res_array: np.ndarray,
        line_gdf: gpd.GeoDataFrame,
        client_range,
        facility_range,
        name: str,
    ):
        # get the pymoo model object
        self.optimized_model = optimization_object
        # and update the result object with the correct X result array to the individual fac and cli vars
        X = res_array
        self.optimized_model.x = X

        # # reshape and transpose the var matrices to get the fac2cli format
        variable_matrix = X.reshape(
            (
                client_range + 1,
                facility_range,
            )
        )

        # transpose the variable matrix to the fac2cli format and then get the indices of the selected lines
        fac2cli = variable_matrix[:-1].T
        self.fac2cli = [np.where(row)[0].tolist() for row in fac2cli]
        self.c2f_vars = np.array(variable_matrix[:-1], dtype=bool)

        # # add the fac vars
        self.fac_vars = [True if entry else False for entry in self.fac2cli]

        # also extract lines and CR objects
        self.selected_lines = line_gdf[self.fac_vars]
        self.cable_road_objects = self.selected_lines["Cable Road Object"]

        (
            self.cost_objective,
            self.ecological_objective,
            self.ergonomics_objective,
        ) = self.optimized_model.get_objective_values()

        self.name = name
