import re
import numpy as np
from itertools import islice

from src.main import (
    classes_linear_optimization,
    classes_mo_optimization,
    geometry_operations,
)


def soo_optimization_augmecon(model_list, optimization_result_list, forest_area_gdf):
    for i in range(0, 3):
        print(f"Starting with objective {i}")
        # set up the model with firs the monetary objective (0), then sideways slope (1) and steep segments (2) as single objective
        lscp_optimization = classes_linear_optimization.optimization_object_spopt(
            "Single Objective",
            forest_area_gdf.line_gdf,
            forest_area_gdf.harvesteable_trees_gdf,
            forest_area_gdf.height_gdf,
            objective_to_select=i,
            maximum_nuber_cable_roads=5,
        )
        lscp_optimization.add_generic_vars_and_constraints()
        lscp_optimization.add_single_objective_function()
        # and solve it
        # %lprun -T tmp0.txt -f lscp_optimization.solve lscp_optimization.solve()
        lscp_optimization.solve()
        model_list.append(lscp_optimization)

    for count, optimization_object in enumerate(model_list):
        optimization_result_list.append(
            classes_linear_optimization.spopt_result(
                optimization_object,
                forest_area_gdf.line_gdf,
                optimization_object.name + str(count),
            )
        )

    return model_list, optimization_result_list


def soo_optimization_manual_weights(
    model_list, optimization_result_list, forest_area_gdf
):
    lscp_optimization = classes_linear_optimization.optimization_object_spopt(
        "Single Objective Manual Weights",
        forest_area_gdf.line_gdf,
        forest_area_gdf.harvesteable_trees_gdf,
        forest_area_gdf.height_gdf,
        objective_to_select=0,
        maximum_nuber_cable_roads=4,
    )
    lscp_optimization.add_generic_vars_and_constraints()

    # lscp_optimization.add_single_objective_function()
    lscp_optimization.model.problem += lscp_optimization.add_cost_objective()
    lscp_optimization.model.problem += lscp_optimization.add_ecological_objective() / 2
    lscp_optimization.model.problem += lscp_optimization.add_ergonomic_objective() / 2
    # and solve it
    # %lprun -T tmp0.txt -f lscp_optimization.solve lscp_optimization.solve()
    lscp_optimization.solve()
    model_list.append(lscp_optimization)

    optimization_result_list.append(
        classes_linear_optimization.spopt_result(
            lscp_optimization,
            forest_area_gdf.line_gdf,
            lscp_optimization.name,
        )
    )

    return optimization_result_list


def create_results_df(optimization_result_list, sample_model, forest_area_gdf):
    """
    Create the first hardcoded version of the resutls dataframe as basis for the AUGMECON optimization
    """

    tree_volumes_list = forest_area_gdf.harvesteable_trees_gdf["cubic_volume"]
    (
        distance_tree_line,
        distance_carriage_support,
    ) = geometry_operations.compute_distances_facilities_clients(
        forest_area_gdf.harvesteable_trees_gdf, forest_area_gdf.line_gdf
    )

    results_df = classes_linear_optimization.model_results_comparison(
        optimization_result_list,
        forest_area_gdf.line_gdf,
        sample_model.distance_tree_line,
        distance_carriage_support,
        sample_model.productivity_cost,
        tree_volumes_list,
    )

    return results_df


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


def nsga_optimization(model_list, optimization_result_list, forest_area_gdf):
    cost_matrix = model_list[0].distance_tree_line
    nsga_problem = classes_mo_optimization.optimization_object_pymoo(
        cost_matrix,
        model_list[0].productivity_cost,
        model_list[0].facility_cost,
        model_list[0].ecological_penalty_lateral_distances,
        model_list[0].ergonomic_penalty_lateral_distances,
    )
    termination = get_termination("n_gen", 50)

    client_range = cost_matrix.shape[0]
    facility_range = cost_matrix.shape[1]

    algorithm = NSGA2(
        pop_size=20,
        sampling=classes_mo_optimization.CustomSampling(),  # initally zero matrix, nothing assigned
        mutation=classes_mo_optimization.MyMutation(),
        repair=classes_mo_optimization.MyRepair(),
        crossover=classes_mo_optimization.CustomCrossOver(),  # crossover: take the sum of CRs from both and then randomly pick the same amount from both
    )

    # %prun minimize(problem,algorithm,termination,verbose=True,return_least_infeasible=True,seed=0)
    res = minimize(
        nsga_problem,
        algorithm,
        termination,
        verbose=False,
        return_least_infeasible=True,
        seed=0,
    )

    X = res.X
    F = res.F

    len_x = len(res.X)
    # get the five best solutions
    samples = np.argsort(np.sum(F, axis=1))[:5]
    for i in samples:  # np.linspace(0, len_x - 1, samples).astype(int):
        optimization_result_list.append(
            classes_mo_optimization.pymoo_result(
                res.problem,
                res.X[i],
                forest_area_gdf.line_gdf,
                client_range,
                facility_range,
                "NSGA2 " + str(i),
            )
        )

    return optimization_result_list


def compute_augmecon_ranges(results_df):
    """
    Determine the ranges to constraint the optimization along for the AUGMECON algorithm
    """
    # determine the ranges of the objectives and divide them in 10 equal parts
    ecological_true_max = results_df["ecological_distances"].max()
    ecological_true_min = results_df["ecological_distances"].min() + 1
    # ecological_true_min = 120
    ergonomics_true_max = results_df["ergonomics_distances"].max()
    ergonomics_true_min = results_df["ergonomics_distances"].min() + 1
    # ergonomics_true_min = 63

    # first determine the ranges of the objectives
    max_overall_profit = results_df["overall_profit"].max()
    min_overall_profit = results_df["overall_profit"].min() + 1

    grid_points = 5
    # create a grid of points to evaluate the objective functions
    profit_range = np.linspace(min_overall_profit, max_overall_profit, grid_points)

    ecological_range, ecological_step = np.linspace(
        ecological_true_max,
        ecological_true_min,
        grid_points,
        retstep=True,
    )

    ergonomics_range, ergonomics_step = np.linspace(
        ergonomics_true_max, ergonomics_true_min, grid_points, retstep=True
    )

    return ecological_range, ecological_step, ergonomics_range, ergonomics_step


def augmecon_optimization(
    optimization_result_list,
    ecological_range,
    ecological_step,
    ergonomics_range,
    ergonomics_step,
    forest_area_gdf,
):
    initial_model = classes_linear_optimization.optimization_object_spopt(
        "Single Objective",
        forest_area_gdf.line_gdf,
        forest_area_gdf.harvesteable_trees_gdf,
        forest_area_gdf.height_gdf,
        objective_to_select=0,
    )

    initial_model.add_generic_vars_and_constraints()
    # add the main monetary objective
    initial_model.add_single_objective_function()
    initial_model.solve()

    # set up the ranges at iteration objects so we can skip steps in the loop
    i_range = iter(ecological_range)
    for i in i_range:
        print("i should be :", i)
        initial_model.add_epsilon_constraint(
            target_value=i,
            constraint_to_select="eco_constraint",
            distances_to_use=initial_model.ecological_penalty_lateral_distances,
        )

        try:
            initial_model.remove_epsilon_constraint("ergo_constraint")
        except:
            # if the constraint doesnt exist, we pass
            pass

        try:
            initial_model.solve()
        except:
            print("couldnt solve with i ", i)
            break

        (
            cost_objective,
            ecological_objective,
            ergonomics_objective,
        ) = initial_model.get_objective_values()
        print("i is :", ecological_objective)

        #
        # # determine the slack variable of the ecological constraint - this is the value of the objective function minus the expected value as per the ecological range
        # i_slack = ecological_objective - i
        # print("i is :", ecological_objective)

        j_range = iter(ergonomics_range)
        # loop through the inner objective
        for j in j_range:
            print("          j should be:", j)
            initial_model.add_epsilon_constraint(
                target_value=j,
                constraint_to_select="ergo_constraint",
                distances_to_use=initial_model.ergonomic_penalty_lateral_distances,
            )

            try:
                initial_model.solve()
            except:
                print("couldnt solve with j ", j)
                break

            (
                cost_objective,
                ecological_objective,
                ergonomics_objective,
            ) = initial_model.get_objective_values()
            i_slack = ecological_objective - i
            # determine the slack variable of the ergonomics constraint - this is the value of the objective function minus the expected value as per the ergonomics range
            j_slack = ergonomics_objective - j
            print("          j is : ", ergonomics_objective)
            print("          cost is :", cost_objective)

            ecological_index = np.where(ecological_range == i)[0][0]
            ergonomics_index = np.where(ergonomics_range == j)[0][0]
            result = classes_linear_optimization.spopt_result(
                initial_model,
                forest_area_gdf.line_gdf,
                "Augmecon" + str(ecological_index) + str(ergonomics_index),
            )

            # pareto_optimal_objective_values.append(overall_objective)
            optimization_result_list.append(result)

            (
                cost_objective,
                ecological_objective,
                ergonomics_objective,
            ) = initial_model.get_objective_values()

            # set the new objective
            initial_model.add_epsilon_objective(
                i_slack, j_slack, ecological_range, ergonomics_range
            )

            # surface_plot_data_x.append(cost_objective)
            # surface_plot_data_y.append(ecological_objective)
            # surface_plot_data_z.append(ergonomics_objective)

            if j_slack > 0:
                print("couldnt improve objective?")
                break  # skipping the rest of the ergonomics range since we cant improve the objective anymore
            else:
                # the eFxpected value as per the ergonomics range. If the slack variable is greater than what we would constrain for the next step, we skip those iterations
                j_bypass = int(abs(np.floor(j_slack / ergonomics_step)))
                if j_bypass > 0:
                    # for iterator j, skip j_bypass steps
                    print("         skipping j_bypass:", j_bypass)
                    next(islice(j_range, j_bypass, j_bypass), None)

    # surface_plot_data_x = []
    # surface_plot_data_y = []
    # surface_plot_data_z = []
    return optimization_result_list


def expert_layout_optimization(optimization_result_list, sample_model, forest_area_gdf):
    # selected_lines = [[30, 37], [32, 59, 71]]
    # modified this after reworking cable road computation, need to reevaluate the expert layouts (23.02.2024)
    selected_lines = [[1, 10], [7, 25]]

    for config in selected_lines:
        expert_result = classes_linear_optimization.expert_result(
            indices=config,
            name="expert_layout_" + str(config),
            line_gdf=forest_area_gdf.line_gdf,
            harvesteable_trees_gdf=forest_area_gdf.harvesteable_trees_gdf,
            sample_productivity_cost_matrix=sample_model.productivity_cost,
            ecological_penalty_lateral_distances=sample_model.ecological_penalty_lateral_distances,
            ergonomics_penalty_lateral_distances=sample_model.ergonomic_penalty_lateral_distances,
        )

        optimization_result_list.append(expert_result)

    return optimization_result_list
