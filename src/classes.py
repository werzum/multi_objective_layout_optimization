from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd


class Cable_Road:
    def __init__(self, line):
        """heights"""
        self.support_height = 11
        self.min_height = 3
        self.start_point_height = 0.0
        self.end_point_height = 0.0
        self.floor_height_below_line_points = (
            []
        )  # the elevation of the floor below the line
        self.line_to_floor_distances = np.array([])
        self.sloped_line_to_floor_distances = np.array([])
        """ geometry features """
        self.line = line
        self.start_point = Point(line.coords[0])
        self.end_point = Point(line.coords[1])
        self.points_along_line = []
        self.floor_points = []
        self.line_start_point_array = []
        self.line_end_point_array = []
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

    def compute_line_height(self, height_gdf: gpd.GeoDataFrame):
        x_points, y_points = zip(
            *[(point.x, point.y) for point in self.points_along_line]
        )

        # get the first elevation point of the list which satisfies the max_deviation condition
        self.floor_height_below_line_points = [
            height_gdf.loc[
                (
                    height_gdf.x.between(
                        x_points[i] - self.max_deviation,
                        x_points[i] + self.max_deviation,
                    )
                )
                & (
                    height_gdf.y.between(
                        y_points[i] - self.max_deviation,
                        y_points[i] + self.max_deviation,
                    )
                ),
                "elev",
            ].values[0]
            for i in range(len(x_points))
        ]

        # create arrays for start and end point
        self.line_start_point_array = np.array(
            [
                self.start_point.x,
                self.start_point.y,
                self.start_point_height,
            ]
        )
        self.line_end_point_array = np.array(
            [
                self.end_point.x,
                self.end_point.y,
                self.end_point_height,
            ]
        )
