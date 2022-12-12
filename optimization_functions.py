import numpy as np
import pulp
import operator
from random import randint, choices
import math


def line_cost_function(line_length, uphill_yarding, large_yarder, intermediate_support_height):
    """Compute the cost of each line based on Bont (2018) Rigi Penalty, line cost based on Stampfer (2003) heuristic

    Args:
        line_length (_type_): _description_
        slope_deviation (_type_): _description_

    Returns:
        _type_: _description_
    """
    # taken from Bont (2018) figure
    #line_cost = (0.005455*line_length) + 21.73
    # Line install cost as per Bont (2019)
    #line_cost+= 200
    cost_man_hour = 44

    # rename the variables according to Kanzian publication
    extraction_direction = uphill_yarding
    yarder_size = large_yarder

    setup_time = math.e**(1.42
                          + 0.00229*line_length
                          + 0.03*intermediate_support_height  # not available now?
                          #+ 0.256*corridor_type  # not easliy determineable -
                          - 0.65*extraction_direction          # 1 for uphill, 0 for downhill
                          + 0.11*yarder_size  # 1 for larger yarder, 0 for smaller 35kn
                          + 0.491*extraction_direction*yarder_size)

    takedown_time = math.e**(0.96
                             + 0.00233*line_length
                             - 0.31 * extraction_direction
                             - 0.31 * intermediate_support_height
                             + 0.33 * yarder_size)

    install_time = setup_time+takedown_time
    line_cost = install_time*cost_man_hour
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


def tree_cost_function(BHD):
    # per Denzin rule of thumb
    volume = (BHD.astype(int)**2)/1000
    # per Bont 2019
    cost = 65*volume
    return cost


def add_facility_variables(model, facility_range):
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
        for fac in facility_range
    ]

    setattr(model, "fac_vars", fac_vars)


def add_facility_client_variables(model, facility_range, client_range):
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
            for j in facility_range
        ]
        for i in client_range
    ]
    setattr(model, "cli_assgn_vars", cli_assgn_vars)

def calculate_productivity_cost(client_range, facility_range, aij, distance_carriage_support, angle_between_supports):
    productivity_cost_matrix = np.empty([len(client_range),len(facility_range)])
    # iterate ove the matrix and calculate the cost for each entry
    it = np.nditer(productivity_cost_matrix, flags=['multi_index'],op_flags=['readwrite'])
    for x in it:
        cli, fac = it.multi_index
        # the cost per m3 based on the productivity model by Gaffariyan, Stampfer, Sessions 2013
        x[...] = 0.043*aij[cli][fac]  # the distance from tree to cable road, aka lateral yarding distance
         # the yarding distance between carriage and support
        + 0.007*distance_carriage_support[cli][fac]
         # +tree_volumes_list[cli]**-0.3 # the crown volume of the tree
        + 0.029*100  # the harvest intensity set to 100%
        + 0.038*angle_between_supports[fac]  # the angle between the supports of this cable road

    return productivity_cost_matrix

def add_moo_objective_function(model, facility_range, client_range, facility_cost, obj_a_factor, start_point_dict, productivity_cost):
    """Add the objective function to the model, compromised of two terms to minimize:
    First term: minimize cost*cli assigned to facility
    Second term: minimize the cost of factories

    Args:
        model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
        facility_cost (_type_): _description_
        obj_a_factor (float): The weight of each objective (as int), is converted here to float to represent the 0-1 range
    """
    obj_a_factor = obj_a_factor*0.1
    
    model.problem += (obj_a_factor)*pulp.lpSum([
            productivity_cost[cli][fac] * model.cli_assgn_vars[cli][fac]
            for cli in client_range
            for fac in facility_range]
        ) + (1-obj_a_factor)*pulp.lpSum([
            model.fac_vars[fac]*facility_cost[fac] for fac in facility_range]
        ) + pulp.lpSum(
            # add a cost factor of 40 for each starting point that is selected
            40*np.unique([start_point_dict[fac] for cli in client_range for fac in facility_range if bool(model.cli_assgn_vars[cli][fac].value())
        ])), "objective function"


def add_singular_assignment_constraint(model, facility_range, client_range):
    """Add the constraint that the sum of facilities assigned for each client == 1 -> only one facility should be assigned to each line

    Args:
        model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in client_range:
        model.problem += pulp.lpSum([model.cli_assgn_vars[cli][fac]
                                    for fac in facility_range]) == 1


def add_facility_is_opened_constraint(model, facility_range, client_range):
    """Add the constraint that for each positive entry in cli_assign_vars (ie., a client is assigned to a facility), 
    there should be a corresponding facility (that is, fac_vars = 1)

    Args:
        model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """
    for cli in client_range:
        for fac in facility_range:
            model.problem += model.fac_vars[fac] - \
                model.cli_assgn_vars[cli][fac] >= 0


def test_and_reassign_clis(fac_range, cli_range, fac_vars, cli_assgn_vars, fac_indices, aij):
    """ Ensure that the opening and assignment constraint are satisfied

    Returns:
        _type_: _description_
    """

    for i in range(fac_range):
        #  find where a cli is assigned to a fac that is not opened (sum of row is negative)
        opening_assignment_test = fac_vars[i]-cli_assgn_vars[:, i]
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
            #random_fac = choices(cli_assgn_indices)

            # set whole row to zero
            cli_assgn_vars[j] = np.zeros(fac_range)
            # and randomly set one of those facs to 1
            cli_assgn_vars[j, random_fac] = 1

    return cli_assgn_vars, fac_vars
