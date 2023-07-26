from random import randint, choices
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
import numpy as np
from pymoo.core.mutation import Mutation

from src.main import optimization_functions


class SupportLinesProblem(ElementwiseProblem):
    def __init__(self, cost_matrix, line_cost, **kwargs):
        self.cost_matrix = cost_matrix

        # create the nr of possible facilities and clients
        self.client_range = cost_matrix.shape[0]
        self.facility_range = cost_matrix.shape[1]

        # add facility cost
        self.facility_cost = np.array(line_cost)

        # = (n_trees*n_facs+n_facs)+n_facs
        self.n_var = self.client_range * self.facility_range + self.facility_range

        super().__init__(
            n_var=self.n_var,
            n_obj=2,
            n_eq_constr=self.client_range,
            n_ieq_constr=self.client_range,
            xl=0,
            xu=1,
            vtype=int,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # reshape n_var to (n_trees*n_facs+n_facs)
        variable_matrix = x.reshape((self.client_range + 1, self.facility_range))
        cli_assgn_vars = variable_matrix[:-1]
        fac_vars = variable_matrix[-1]

        overall_distance_obj = np.sum(cli_assgn_vars * self.cost_matrix)
        overall_cost_obj = np.sum(fac_vars * self.facility_cost)

        # for each row sum should be 1 -> only one client allowed
        singular_assignment_constr = np.sum(cli_assgn_vars, axis=1) - 1

        # want to enforce that for each row where a 1 exists, fac_vars also has a 1
        facility_is_opened_constr = -np.sum(fac_vars - cli_assgn_vars, axis=1)

        out["F"] = np.column_stack([overall_distance_obj, overall_cost_obj])
        # ieq constr
        out["G"] = np.column_stack([facility_is_opened_constr])
        # eq constr
        out["H"] = np.column_stack([singular_assignment_constr])


class MyRepair(Repair):
    def _do(self, problem, x, **kwargs):
        buffer = []  # Create a buffer to store mutated solutions
        x_shape = x.shape[0]  # Get the number of solutions in 'x'

        for j in range(x_shape):
            # Reshape the solution 'x[j]' into a matrix with 'problem.client_range + 1' rows and 'problem.facility_range' columns
            variable_matrix = x[j].reshape(
                (problem.client_range + 1, problem.facility_range)
            )
            cli_assgn_vars = variable_matrix[
                :-1
            ]  # Extract the rows except the last one (client assignment variables)
            fac_vars = variable_matrix[-1]  # Extract the last row (facility variables)

            # get indices of open facs
            fac_indices = np.where(fac_vars == 1)[0]

            if fac_indices.any():
                # reassign all clis to open facs
                cli_assgn_vars, fac_vars = reassign_clients(
                    problem, fac_vars, cli_assgn_vars, fac_indices
                )

            # append this solution
            buffer.append(np.vstack([cli_assgn_vars, fac_vars]))

        x = np.array(buffer).reshape(
            (x.shape[0], (problem.client_range + 1) * problem.facility_range)
        )
        return x


from pymoo.core.sampling import Sampling


class CustomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        return np.zeros((n_samples, problem.n_var))


class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def remove_facility(
        self,
        problem: SupportLinesProblem,
        fac_vars: np.ndarray,
        cli_assgn_vars: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly removes a facility and reassigns clients to other open facilities.

        Args:
            problem ('SupportLinesProblem'): The problem instance containing cost and facility data.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
            cli_assgn_vars (np.ndarray): Binary array representing client assignments to facilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing updated cli_assgn_vars and fac_vars.
        """
        # duplicate code, but dont see how to remove this if we fork in different ways during the loop
        for _ in range(10):
            # Get the indices of open facilities
            fac_indices = np.where(fac_vars == 1)[0]
            if len(fac_indices) == 0:
                break  # No open facilities, no mutation possible

            # Randomly choose a facility to delete
            fac_to_delete = np.random.choice(fac_indices)
            fac_vars[fac_to_delete] = 0

            # Create a boolean mask for the condition cli_assgn_vars[j, fac_to_delete] == 1
            mask = cli_assgn_vars[:, fac_to_delete] == 1
            # Use the mask for boolean indexing and set the corresponding elements to 0
            cli_assgn_vars[mask, fac_to_delete] = 0

            # Reassign clients to the closest open facilities
            reassign_clients(problem, fac_vars, cli_assgn_vars, fac_indices)

            objective_value_after = self.objective_value_after(
                problem, fac_vars, cli_assgn_vars
            )
            objective_value_before = self.objective_value_before(
                problem, fac_vars, cli_assgn_vars, fac_to_delete
            )

            # Check if this mutation decreased the objective function
            if objective_value_after < objective_value_before:
                break  # Mutation improved the objective, stop trying
            else:
                # Undo this mutation and keep trying other facilities
                fac_vars[fac_to_delete] = 1
                # Reassign clients again to their original facility (using precomputed values)
                cli_assgn_vars, fac_vars = reassign_clients(
                    problem, fac_vars, cli_assgn_vars, fac_indices
                )

        return cli_assgn_vars, fac_vars

    def objective_value_after(
        self,
        problem: SupportLinesProblem,
        fac_vars: np.ndarray,
        cli_assgn_vars: np.ndarray,
    ) -> float:
        # Calculate objective values after the mutation
        overall_distance_obj_after = np.sum(cli_assgn_vars * problem.cost_matrix)
        overall_cost_obj_after = np.sum(fac_vars * problem.facility_cost)

        return overall_cost_obj_after + overall_distance_obj_after

    def objective_value_before(
        self, problem, fac_vars, cli_assgn_vars, fac_to_delete
    ) -> float:
        overall_distance_obj_before = np.sum(cli_assgn_vars * problem.cost_matrix)
        overall_cost_obj_before = np.sum(fac_vars * problem.facility_cost)
        objective_value_before = overall_cost_obj_before + overall_distance_obj_before

        return objective_value_before

    def add_facility(
        self,
        problem: SupportLinesProblem,
        fac_vars: np.ndarray,
        cli_assgn_vars: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Randomly opens a facility and reassigns clients to the closest open facilities.

        Args:
            problem (SupportLinesProblem): The problem instance containing cost and facility data.
            fac_vars (np.ndarray): Binary array representing open/closed status of facilities.
            cli_assgn_vars (np.ndarray): Binary array representing client assignments to facilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing updated cli_assgn_vars and fac_vars.
        """
        for _ in range(10):
            # Get the objective value before the mutation
            overall_distance_obj_before = np.sum(cli_assgn_vars * problem.cost_matrix)
            overall_cost_obj_before = np.sum(fac_vars * problem.facility_cost)
            objective_value_before = (
                overall_cost_obj_before + overall_distance_obj_before
            )

            # Randomly open a facility (set its variable to 1)
            factory_to_open = randint(0, problem.facility_range - 1)
            fac_vars[factory_to_open] = 1
            fac_indices = np.where(fac_vars == 1)[
                0
            ]  # Get the indices of open facilities

            # Reassign clients to the closest open facilities
            if len(fac_indices) > 0:
                cli_assgn_vars, fac_vars = reassign_clients(
                    problem, fac_vars, cli_assgn_vars, fac_indices
                )

                # Get the objective value after the mutation
                overall_distance_obj_after = np.sum(
                    cli_assgn_vars * problem.cost_matrix
                )
                overall_cost_obj_after = np.sum(fac_vars * problem.facility_cost)
                objective_value_after = (
                    overall_cost_obj_after + overall_distance_obj_after
                )

                # Check if this mutation decreased the objective function
                if objective_value_after < objective_value_before:
                    break  # Mutation improved the objective, stop trying
                else:
                    # Undo this mutation and keep trying other facilities
                    fac_vars[factory_to_open] = 0
                    # Reassign clients again to their original facility
                    cli_assgn_vars, fac_vars = reassign_clients(
                        problem, fac_vars, cli_assgn_vars, fac_indices
                    )

        return cli_assgn_vars, fac_vars

    def _do(self, problem: SupportLinesProblem, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the mutation operator to the solutions in 'x'.

        Args:
            problem (SupportLinesProblem): The problem instance containing cost and facility data.
            x (np.ndarray): Array of solutions to be mutated.

        Returns:
            np.ndarray: Mutated solutions.
        """
        buffer = []  # Create a buffer to store mutated solutions
        x_shape = x.shape[0]  # Get the number of solutions in 'x'

        for j in range(x_shape):
            # Reshape the solution 'x[j]' into a matrix with 'problem.client_range + 1' rows and 'problem.facility_range' columns
            variable_matrix = x[j].reshape(
                (problem.client_range + 1, problem.facility_range)
            )
            cli_assgn_vars = variable_matrix[
                :-1
            ]  # Extract the rows except the last one (client assignment variables)
            fac_vars = variable_matrix[-1]  # Extract the last row (facility variables)

            for _ in range(10):
                fac_indices = np.where(fac_vars == 1)[0]

                if len(fac_indices) == 0:
                    cli_assgn_vars, fac_vars = self.add_facility(
                        problem, fac_vars, cli_assgn_vars
                    )
                else:
                    # Try to remove a facility that decreases the objective value or add one if not possible
                    if randint(0, 1):
                        cli_assgn_vars, fac_vars = self.remove_facility(
                            problem, fac_vars, cli_assgn_vars
                        )
                    else:
                        cli_assgn_vars, fac_vars = self.add_facility(
                            problem, fac_vars, cli_assgn_vars
                        )

            buffer.append(
                np.vstack([cli_assgn_vars, fac_vars])
            )  # Store the mutated solution

        # Reshape the buffer into the same shape as 'x'
        x = np.array(buffer).reshape(
            (x_shape, (problem.client_range + 1) * problem.facility_range)
        )
        return x  # Return the mutated solutions


def reassign_clients(
    problem: SupportLinesProblem,
    fac_vars: np.ndarray,
    cli_assgn_vars: np.ndarray,
    fac_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    min_indices = np.argmin(problem.cost_matrix[:, fac_indices], axis=1)

    # Create an array for the updated client assignments
    updated_cli_assgn_vars = np.zeros_like(cli_assgn_vars)

    # Use numpy fancy indexing to update the client assignments efficiently
    rows = np.arange(problem.client_range)
    cols = fac_indices[min_indices]
    updated_cli_assgn_vars[rows, cols] = 1

    a = np.zeros(len(fac_vars))
    a[fac_indices] = 1

    return updated_cli_assgn_vars, fac_vars
