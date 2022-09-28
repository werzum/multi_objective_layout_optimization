import numpy as np
import pulp


def line_cost_function(line_length, slope_deviation):
    """Compute the cost of each line based on different factors

    Args:
        line_length (_type_): _description_
        slope_deviation (_type_): _description_

    Returns:
        _type_: _description_
    """
    return line_length**1.5+slope_deviation**2


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


def add_moo_objective_function(model, facility_range, client_range, facility_cost, obj_a_factor):
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
        model.aij[cli][fac] * model.cli_assgn_vars[cli][fac]
        for cli in client_range
        for fac in facility_range
    ]) + (1-obj_a_factor)*pulp.lpSum([
        model.fac_vars[fac]*facility_cost[fac] for fac in facility_range]
    ), "objective function"

def add_singular_assignment_constraint(model, facility_range, client_range):
    """Add the constraint that the sum of facilities assigned for each client == 1 -> only one facility should be assigned to each line

    Args:
        model (_type_): _description_
        facility_range (_type_): _description_
        client_range (_type_): _description_
    """    
    for cli in client_range:
        model.problem += pulp.lpSum([model.cli_assgn_vars[cli][fac] for fac in facility_range]) == 1

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
            model.problem += model.fac_vars[fac] - model.cli_assgn_vars[cli][fac] >= 0