import optimization_functions

from random import randint, choices
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
import numpy as np

from pymoo.core.mutation import Mutation


class SupportLinesProblem(ElementwiseProblem):
    def __init__(self, cost_matrix, line_cost, **kwargs):
        
        self.cost_matrix = cost_matrix

        # create the nr of possible facilities and clients 
        self.client_range = cost_matrix.shape[0]
        self.facility_range = cost_matrix.shape[1]

        # add facility cost
        self.facility_cost = np.array(line_cost)
        
        # = (n_trees*n_facs+n_facs)+n_facs
        self.n_var = self.client_range*self.facility_range+self.facility_range
        
        super().__init__(n_var=self.n_var, n_obj=2, n_eq_constr=self.client_range, n_ieq_constr=self.client_range, xl=0, xu=1, vtype=int, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        # reshape n_var to (n_trees*n_facs+n_facs)
        variable_matrix = x.reshape((self.client_range+1,self.facility_range))
        cli_assgn_vars = variable_matrix[:-1]
        fac_vars = variable_matrix[-1]

        overall_distance_obj = np.sum(cli_assgn_vars*self.cost_matrix)
        overall_cost_obj = np.sum(fac_vars*self.facility_cost)

        # for each row sum should be 1 -> only one client allowed
        singular_assignment_constr = np.sum(cli_assgn_vars,axis=1) - 1

        # want to enforce that for each row where a 1 exists, fac_vars also has a 1
        facility_is_opened_constr = - np.sum(fac_vars-cli_assgn_vars, axis=1)

        out["F"] = np.column_stack([overall_distance_obj, overall_cost_obj])
        # ieq constr
        out["G"] = np.column_stack([facility_is_opened_constr])
        # eq constr
        out["H"] = np.column_stack([singular_assignment_constr])

class MyRepair(Repair):
    def _do(self, problem, x, **kwargs):

        buffer = []
        x_shape = x.shape[0]

        # enumerate through all solution populations
        for j in range(x_shape):
            variable_matrix = x[j].reshape((problem.client_range+1,problem.facility_range))
            cli_assgn_vars = variable_matrix[:-1]
            fac_vars = variable_matrix[-1]
            
            # get indices of open facs
            fac_indices = np.where(fac_vars==1)[0]

            # reassign all clis to open facs
            cli_assgn_vars, fac_vars = optimization_functions.test_and_reassign_clis(problem.facility_range, problem.client_range, fac_vars, cli_assgn_vars, fac_indices, problem.cost_matrix)

            # append this solution
            buffer.append(np.vstack([cli_assgn_vars,fac_vars]))      
        
        x = np.array(buffer).reshape((x.shape[0],(problem.client_range+1)*problem.facility_range))
        return x

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, x, **kwargs):

        buffer = []
        x_shape = x.shape[0]

        for j in range(x_shape):
            variable_matrix = x[j].reshape((problem.client_range+1,problem.facility_range))
            cli_assgn_vars = variable_matrix[:-1]
            fac_vars = variable_matrix[-1]

            #randomly remove a facility
            fac_indices = np.where(fac_vars==1)[0]
            if randint(0,1) and len(fac_indices)>0:
                fac_vars[choices(fac_indices)] = 0

                #reshuffle those clis which had previously been assigned to this
                cli_assgn_vars, fac_vars = optimization_functions.test_and_reassign_clis(problem.facility_range, problem.client_range, fac_vars, cli_assgn_vars, fac_indices, problem.cost_matrix)

            # else randomly open one facility up
            else:
                fac_vars[randint(0,problem.facility_range-1)] = 1
                fac_indices = np.where(fac_vars==1)[0]

                # just now for speedup:
                if randint(0,1):
                    # and reassign all facs to their closest version
                    if len(fac_indices)>0:
                        for j in range(problem.client_range):

                            # smallest distance of this cli to open facs
                            smallest_distance = min(problem.cost_matrix[j,fac_indices])

                            # find its position
                            min_index = np.where(problem.cost_matrix[j]==smallest_distance)[0]
                            # and assign to one of them
                            random_fac = choices(min_index)

                            cli_assgn_vars[j, random_fac] = 1
        
            buffer.append(np.vstack([cli_assgn_vars,fac_vars]))
        
        x = np.array(buffer).reshape((x_shape,(problem.client_range+1)*problem.facility_range))
        return x