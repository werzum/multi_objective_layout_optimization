
# reload(optimization_compute_quantification)
# reload(classes_linear_optimization)

# Lists to store the results
optimization_result_list = []
lscp_model_list = []

for i in range(0, 3):
    print(f"Starting with objective {i}")
    # set up the model with firs the monetary objective (0), then sideways slope (1) and steep segments (2) as single objective
    lscp_optimization = classes_linear_optimization.optimization_object_spopt(
        "Single Objective",
        line_gdf,
        harvesteable_trees_gdf,
        height_gdf,
        objective_to_select=i,
        maximum_nuber_cable_roads=4,
    )
    lscp_optimization.add_generic_vars_and_constraints()
    lscp_optimization.add_single_objective_function()
    # and solve it
    # %lprun -T tmp0.txt -f lscp_optimization.solve lscp_optimization.solve()
    lscp_optimization.solve()
    lscp_model_list.append(lscp_optimization)
