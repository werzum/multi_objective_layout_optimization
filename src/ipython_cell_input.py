# reload(geometry_operations)
# reload(optimization_functions)

# Apply the Line Cost Function:
uphill_yarding = 1
large_yarder = 0
intermediate_support_height = 12 #take the average for now
line_gdf["line_cost"] = [optimization_functions.line_cost_function(line_gdf["line_length"][index], uphill_yarding, large_yarder,intermediate_support_height,line_gdf["number_of_supports"][index]) for index in range(len(line_gdf))]

# Create a matrix with the distance between every tree and line and the distance between the support (beginning of the CR) and the carriage (cloests point on the CR to the tree)
distance_tree_line, distance_carriage_support = geometry_operations.compute_distances_facilities_clients(harvesteable_trees_gdf, line_gdf)

# sort the facility (=lines) and demand points (=trees)
facility_points_gdf = line_gdf.reset_index()
demand_points_gdf = harvesteable_trees_gdf.reset_index()

# set up the solver
solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
name = "model"

# create the nr of possible facilities and clients 
client_range = range(distance_tree_line.shape[0])
facility_range = range(distance_tree_line.shape[1])

# add facility cost with an optional scaling factor
facility_scaling_factor = 1

facility_cost = line_gdf.line_cost.values*facility_scaling_factor

# create the aij cost matrix, which is really just the distance from the tree to the line
aij = distance_tree_line

# collect the matrices needed for the optimization
tree_volumes_list = harvesteable_trees_gdf["crownVolume"]
angle_between_supports_list = line_gdf["angle_between_supports"]
# cost_distance_tree_line
# distance_carriage_support

# and the productivity cost combination of each line combination
productivity_cost = optimization_functions.calculate_productivity_cost(client_range, facility_range, aij, distance_carriage_support, angle_between_supports)
