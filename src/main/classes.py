from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd
from itertools import pairwise
import pulp
from spopt.locate import PMedian
from src.tests import helper_functions


class Point_3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def xyz(self):
        return np.array([self.x, self.y, self.z])

    def distance(self, other):
        """Returns the distance between two 3dpoints"""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


class LineString_3D:
    def __init__(self, start_point: Point_3D, end_point: Point_3D):
        self.start_point = start_point
        self.end_point = end_point

    def interpolate(self, distance: float) -> Point_3D:
        """Returns the interpolated point at a given distance from the start point
        Args:
            distance (float): distance from the start point
        Returns:
            Point_3D: interpolated point"""

        vector = self.end_point.xyz - self.start_point.xyz
        # normalize the vector
        vector = vector / np.linalg.norm(vector)
        # multiply the vector with the distance
        vector = vector * distance
        # add the vector to the start point
        return Point_3D(
            self.start_point.x + vector[0],
            self.start_point.y + vector[1],
            self.start_point.z + vector[2],
        )

    def length(self):
        return self.start_point.distance(self.end_point)


class optimization_model:
    def __init__(
        self,
        name: str,
        line_gdf: gpd.GeoDataFrame,
        harvesteable_trees_gdf: gpd.GeoDataFrame,
        height_gdf: gpd.GeoDataFrame,
        slope_line: LineString,
        uphill_yarding: bool = False,
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

        # add facility cost with an optional scaling factor
        facility_scaling_factor = 1

        self.facility_cost = line_gdf.line_cost.values * facility_scaling_factor

        # # create the aij cost matrix, which is really just the distance from the tree to the line
        # distance_greater_15_mask = self.distance_tree_line > 15
        # self.distance_tree_line[
        #     distance_greater_15_mask
        # ] = (  # square all distances greater than 15
        #     self.distance_tree_line[distance_greater_15_mask]
        #     + (self.distance_tree_line[distance_greater_15_mask] - 15) * 2
        # )
        self.aij = self.distance_tree_line

        # collect the matrices needed for the optimization
        self.tree_volumes_list = harvesteable_trees_gdf["cubic_volume"]
        self.angle_between_supports_list = line_gdf["angle_between_supports"]

        self.average_steepness = geometry_operations.compute_average_terrain_steepness(
            line_gdf, height_gdf
        )

        # calculate the deviations of the subsegments of the cable road from the slope line
        self.sideways_slope_deviations_per_cable_road = (
            optimization_functions.compute_cable_road_deviations_from_slope(
                line_gdf, slope_line
            )
        )

        # calculate the segments which have a vertical slope greater than 10 and are treated in downhill logging
        if not uphill_yarding:
            self.steep_downhill_segments = (
                optimization_functions.compute_length_of_steep_downhill_segments(
                    line_gdf
                )
            )

        # and the productivity cost combination of each line combination
        self.productivity_cost = optimization_functions.calculate_felling_cost(
            self.client_range,
            self.facility_range,
            self.aij,
            self.distance_carriage_support,
            self.tree_volumes_list,
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
        cost, sideways, downhill = optimization_functions.get_objective_values(self)
        self.epsilon = 1
        return cost + self.epsilon * (
            (sideways / i_range_min_max) + (downhill / j_range_min_max)
        )

    def add_single_objective(self):
        self.model = optimization_functions.add_single_objective_function(self)

    def add_weighted_objective(self):
        self.model = optimization_functions.add_weighted_objective_function(self)

    def solve(self):
        self.model = self.model.solve(self.solver)


class optimization_result:
    def __init__(
        self,
        optimization_object,
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
            ) = helper_functions.model_to_line_gdf(self.optimized_model, line_gdf)
            self.fac2cli = optimization_object.model.fac2cli

        elif hasattr(optimization_object, "X"):
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

            # also extract lines and CR objects
            (
                self.selected_lines,
                self.cable_road_objects,
            ) = helper_functions.model_to_line_gdf(variable_matrix, line_gdf)

        self.selected_lines["number_int_supports"] = [
            len(list(cable_road.get_all_subsegments())) - 1
            if list(cable_road.get_all_subsegments())
            else 0
            for cable_road in self.cable_road_objects
        ]

        self.name = name


class Support:
    def __init__(
        self,
        attachment_height: int,
        xy_location: Point,
        height_gdf: gpd.GeoDataFrame,
        max_supported_force: list[float],
        max_deviation: float = 1,
        max_holding_force: float = 0.0,
        is_tower: bool = False,
    ):
        self.attachment_height: int = attachment_height
        self.xy_location: Point = xy_location
        self.xy_location_numpy = np.array(self.xy_location.xy).T
        self.max_deviation: float = max_deviation
        self.is_tower: bool = is_tower
        self.max_supported_force = max_supported_force
        self.max_holding_force: float = max_holding_force

        # get the elevation of the floor below the support
        self.floor_height = geometry_operations.fetch_point_elevation(
            self.xy_location, height_gdf, self.max_deviation
        )

    @property
    def total_height(self):
        return self.floor_height + self.attachment_height

    @property
    def xyz_location(self):
        return Point_3D(self.xy_location.x, self.xy_location.y, self.total_height)

    @property
    def max_supported_force_at_attachment_height(self):
        return self.max_supported_force[self.attachment_height]

    @max_supported_force_at_attachment_height.setter
    def max_supported_force_at_attachment_height(self, value):
        self.max_supported_force[self.attachment_height] = value


class Cable_Road:
    def __init__(
        self,
        line,
        height_gdf,
        start_support: Support,
        end_support: Support,
        pre_tension: float = 0,
        number_sub_segments: int = 0,
    ):
        self.start_support: Support = start_support
        self.end_support: Support = end_support

        """heights"""
        self.floor_height_below_line_points = (
            []
        )  # the elevation of the floor below the line
        self.sloped_line_to_floor_distances = np.array([])
        self.unloaded_line_to_floor_distances = np.array([])
        """ geometry features """
        self.line = line
        # self.start_point = Point(line.coords[0])
        # self.end_point = Point(line.coords[1])
        self.points_along_line = []
        self.floor_points = []
        self.max_deviation = 1
        self.anchor_triplets = []
        """ Fixed cable road parameters """
        self.q_s_self_weight_center_span = 10
        self.q_load = 80000
        self.b_length_whole_section = 0.0
        self.s_max_maximalspannkraft = 0.0
        """ Modifiable collision parameters """
        self.no_collisions = True
        self.anchors_hold = True

        # Parameters to keep track of segments+
        self.number_sub_segments = number_sub_segments
        self.supported_segments: list[
            SupportedSegment
        ] = []  # list of SupportedSegment objects, ie. sub cable roads

        self._s_current_tension = 0.0
        print(
            "Cable road created from line: ",
            self.line.coords[0],
            "to ",
            self.line.coords[1],
        )

        # fetch the floor points along the line - xy view
        self.points_along_line = geometry_operations.generate_road_points(
            self.line, interval=1
        )

        self.points_along_line_x, self.points_along_line_y = [
            point.x for point in self.points_along_line
        ], [point.y for point in self.points_along_line]

        self.points_along_line_xy = np.column_stack(
            (self.points_along_line_x, self.points_along_line_y)
        )

        # get the height of those points and set them as attributes to the CR object
        self.compute_floor_height_below_line_points(height_gdf)
        # generate floor points and their distances
        self.floor_points = list(
            zip(
                self.points_along_line_x,
                self.points_along_line_y,
                self.floor_height_below_line_points,
            )
        )

        self.b_length_whole_section = self.start_support.xy_location.distance(
            self.end_support.xy_location
        )

        self.c_rope_length = self.start_support.xyz_location.distance(
            self.end_support.xyz_location
        )

        self.initialize_line_tension(number_sub_segments, pre_tension)

    @property
    def line_to_floor_distances(self):
        return np.asarray(
            [
                geometry_utilities.lineseg_dist(
                    point,
                    self.start_support.xyz_location.xyz,
                    self.end_support.xyz_location.xyz,
                )
                for point in self.floor_points
            ]
        )

    @property
    def absolute_unloaded_line_height(self):
        return (
            self.floor_height_below_line_points + self.unloaded_line_to_floor_distances
        )

    @property
    def absolute_loaded_line_height(self):
        return self.floor_height_below_line_points + self.sloped_line_to_floor_distances

    @property
    def rope_points_xyz(self):
        return list(
            zip(
                self.points_along_line_x,
                self.points_along_line_y,
                self.absolute_loaded_line_height,
            )
        )

    @property
    def s_current_tension(self) -> float:
        return self._s_current_tension

    # ensure that the CR height is updated when we change the tension
    @s_current_tension.setter
    def s_current_tension(self, value):
        self._s_current_tension = value
        self.compute_loaded_unloaded_line_height()

        # also for the sub-CRs
        if self.supported_segments:
            for segment in self.supported_segments:
                segment.cable_road.s_current_tension = value
                segment.cable_road.compute_loaded_unloaded_line_height()

    def count_segments(self, number_sub_segments: int = 0) -> int:
        """recursively counts the number of segments in the cable road"""
        if self.supported_segments:
            number_sub_segments += 2
            for segment in self.supported_segments:
                number_sub_segments = segment.cable_road.count_segments(
                    number_sub_segments
                )

        return number_sub_segments

    def get_all_subsegments(self):
        """get a generator of all subsegments of the cable road
        Loosely based on https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python

        Returns:
            generator: generator of all subsegments (apply list to get list of it)

        """
        for i in self.supported_segments:
            if i.cable_road.supported_segments:
                yield from i.cable_road.get_all_subsegments()
            else:
                yield i

    def compute_floor_height_below_line_points(self, height_gdf: gpd.GeoDataFrame):
        """compute the height of the line above the floor as well as the start and end point in 3d. Query the global kdtree for that
        Sets the floor_height_below_line_points and the line_start_point_array and line_end_point_array
        Args:
            height_gdf (gpd.GeoDataFrame): the floor height data
        """
        d, i = global_vars.kdtree.query(
            list(zip(self.points_along_line_x, self.points_along_line_y))
        )
        # Use the final condition to filter the height_gdf and get the elev values
        self.floor_height_below_line_points = height_gdf.iloc[i]["elev"].values

    def compute_loaded_unloaded_line_height(self):
        """compute the loaded and unloaded line to floor distances"""
        self.calculate_cr_deflections(loaded=True)
        self.calculate_cr_deflections(loaded=False)

    def calculate_cr_deflections(self, loaded: bool = True):
        """calculate the deflections of the CR line due to the load, either loaded or unlaoded
        Args:
            loaded (bool, optional): whether the line is loaded or not. Defaults to True.
        """

        y_x_deflections = mechanical_computations.pestal_load_path(self, loaded)

        if loaded:
            self.sloped_line_to_floor_distances = (
                self.line_to_floor_distances - y_x_deflections
            )
        else:
            self.unloaded_line_to_floor_distances = (
                self.line_to_floor_distances - y_x_deflections
            )

    def initialize_line_tension(self, current_supports: int, pre_tension: int = 0):
        # set tension of the cable_road
        s_br_mindestbruchlast = 170000  # in newton
        self.s_max_maximalspannkraft = s_br_mindestbruchlast / 2
        if pre_tension:
            self.s_current_tension = pre_tension
        else:
            self.s_current_tension = self.s_max_maximalspannkraft


class SupportedSegment:
    def __init__(
        self,
        cable_road: Cable_Road,
        start_support: Support,
        end_support: Support,
    ):
        self.cable_road = cable_road
        self.start_support = start_support
        self.end_support = end_support


# Helper Functions for setting up the cable road


def initialize_cable_road_with_supports(
    line: LineString,
    height_gdf: gpd.GeoDataFrame,
    start_point_max_supported_force: list[float],
    end_point_max_supported_force: list[float],
    pre_tension=0,
    is_tower=False,
):
    start_support = Support(
        attachment_height=11,
        xy_location=Point(line.coords[0]),
        height_gdf=height_gdf,
        max_supported_force=start_point_max_supported_force,
        is_tower=is_tower,
    )
    end_support = Support(
        attachment_height=8,
        xy_location=Point(line.coords[-1]),
        height_gdf=height_gdf,
        max_supported_force=end_point_max_supported_force,
        is_tower=False,
    )
    return Cable_Road(line, height_gdf, start_support, end_support, pre_tension)


def load_cable_road(line_gdf: gpd.GeoDataFrame, index: int) -> Cable_Road:
    """Helper function to abstract setting up a sample cable road from the line_gdf"""
    return line_gdf.iloc[index]["Cable Road Object"]


from src.main import (
    geometry_operations,
    geometry_utilities,
    mechanical_computations,
    global_vars,
    optimization_functions,
)
