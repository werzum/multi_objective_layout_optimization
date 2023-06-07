from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd

from src import geometry_operations, geometry_utilities, mechanical_computations


class Cable_Road:
    def __init__(
        self,
        line,
        height_gdf,
        pre_tension=0,
        current_supports=0,
    ):
        """heights"""
        self.start_support_height = 11
        self.end_support_height = 11
        self.min_height = 3
        self.start_point_height = 0.0
        self.end_point_height = 0.0
        self.floor_height_below_line_points = (
            []
        )  # the elevation of the floor below the line
        self.sloped_line_to_floor_distances = np.array([])
        self.unloaded_line_to_floor_distances = np.array([])
        """ geometry features """
        self.line = line
        self.start_point = Point(line.coords[0])
        self.end_point = Point(line.coords[1])
        self.points_along_line = []
        self.floor_points = []
        self.max_deviation = 0.1
        self.anchor_triplets = []
        """ Fixed cable road parameters """
        self.q_s_self_weight_center_span = 10
        self.q_load = 80000
        self.c_rope_length = 0.0
        self.b_length_whole_section = 0.0
        self.s_max_maximalspannkraft = 0.0
        """ Modifiable collision parameters """
        self.no_collisions = True
        self.anchors_hold = True
        self.s_current_tension = 0.0

        # and further init:
        self.start_point_height = geometry_operations.fetch_point_elevation(
            self.start_point, height_gdf, self.max_deviation
        )
        self.end_point_height = geometry_operations.fetch_point_elevation(
            self.end_point, height_gdf, self.max_deviation
        )

        # fetch the floor points along the line
        self.points_along_line = geometry_operations.generate_road_points(
            self.line, interval=2
        )

        # get the height of those points and set them as attributes to the CR object
        self.compute_line_height(height_gdf)

        # generate floor points and their distances
        self.floor_points = list(
            zip(
                [point.x for point in self.points_along_line],
                [point.y for point in self.points_along_line],
                self.floor_height_below_line_points,
            )
        )

        # get the rope length
        self.b_length_whole_section = self.start_point.distance(self.end_point)

        self.c_rope_length = geometry_utilities.distance_between_3d_points(
            self.line_start_point_array, self.line_end_point_array
        )

        mechanical_computations.initialize_line_tension(
            self, current_supports, pre_tension
        )

        # and calculate the sloped ltfd
        y_x_deflections_loaded = np.asarray(
            [
                mechanical_computations.pestal_load_path(self, point)
                for point in self.points_along_line
            ],
            dtype=np.float32,
        )

        # as well as the empty deflections
        y_x_deflections_unloaded = np.asarray(
            [
                mechanical_computations.pestal_load_path(self, point, loaded=False)
                for point in self.points_along_line
            ],
            dtype=np.float32,
        )

        #  check the distances between each floor point and the ldh point
        self.sloped_line_to_floor_distances = (
            self.line_to_floor_distances - y_x_deflections_loaded
        )

        self.unloaded_line_to_floor_distances = (
            self.line_to_floor_distances - y_x_deflections_unloaded
        )

    @property
    def line_to_floor_distances(self):
        return np.asarray(
            [
                geometry_utilities.lineseg_dist(
                    point,
                    self.line_start_point_array,
                    self.line_end_point_array,
                )
                for point in self.floor_points
            ]
        )

    @property
    def total_start_point_height(self):
        return self.start_point_height + self.start_support_height

    @property
    def total_end_point_height(self):
        return self.end_point_height + self.end_support_height

    @property
    def line_start_point_array(self):
        return np.array(
            [
                self.start_point.x,
                self.start_point.y,
                self.total_start_point_height,
            ]
        )

    @property
    def line_end_point_array(self):
        return np.array(
            [
                self.end_point.x,
                self.end_point.y,
                self.total_end_point_height,
            ]
        )

    def compute_line_height(self, height_gdf: gpd.GeoDataFrame):
        """compute the height of the line above the floor as well as the start and end point in 3d.
        Sets the floor_height_below_line_points and the line_start_point_array and line_end_point_array
        Args:
            height_gdf (gpd.GeoDataFrame): the floor height data
        """
        # generate four lists of x and y values with min and max values for each point
        x_point_min, x_point_max, y_point_min, y_point_max = zip(
            *[
                (
                    point.x - self.max_deviation,
                    point.x + self.max_deviation,
                    point.y - self.max_deviation,
                    point.y + self.max_deviation,
                )
                for point in self.points_along_line
            ]
        )

        # for each value in the list, find the elevation of the floor below the line in the height_gdf by selecting the
        # first matching value in the height_gdf
        self.floor_height_below_line_points = [
            height_gdf[
                height_gdf.x.between(x_point_min[i], x_point_max[i])
                & (height_gdf.y.between(y_point_min[i], y_point_max[i]))
            ]["elev"].values[0]
            for i in range(len(x_point_min))
        ]
