from shapely.geometry import LineString, Point
import numpy as np
import geopandas as gpd

from src.main import (
    geometry_operations,
    geometry_utilities,
    mechanical_computations,
)


# TODO - dont need this since we store the CR object in the line_gdf and therefore dont need to restore it?
# def set_up_CR_from_linegdf(line_gdf, index, height_gdf):
#     """Helper function to abstract setting up a sample cable road from the line_gdf
#     Args:
#         line_gdf (gpd.GeoDataFrame): the line_gdf
#         index (int): the index of the line_gdf to use
#         height_gdf (gpd.GeoDataFrame): the height_gdf

#     Returns:
#         Cable_Road: the cable road
#     """

#     sample_line = line_gdf.iloc[index].geometry
#     left_support = Support(8,
#     return Cable_Road(sample_line, height_gdf, line_gdf.iloc[index].current_tension)


# def create_support_from_CR(cable_road: Cable_Road, use_left_support: bool, height_gdf):
#     if use_left_support:
#         support = Support(8, cable_road.line.)
#     else:
#         support = cable_road.right_support


def initialize_cable_road_with_supports(
    line: LineString,
    height_gdf: gpd.GeoDataFrame,
    pre_tension=0,
    is_tower=False,
    left_max_supported_force=0.0,
    right_max_supported_force=0.0,
):
    left_support = Support(
        attachment_height=11,
        xy_location=Point(line.coords[0]),
        height_gdf=height_gdf,
        max_supported_force=left_max_supported_force,
        is_tower=is_tower,
    )
    right_support = Support(
        attachment_height=8,
        xy_location=Point(line.coords[-1]),
        height_gdf=height_gdf,
        max_supported_force=right_max_supported_force,
        is_tower=False,
    )
    return Cable_Road(line, height_gdf, left_support, right_support, pre_tension)


class Support:
    def __init__(
        self,
        attachment_height: float,
        xy_location: Point,
        height_gdf: gpd.GeoDataFrame,
        max_deviation: float = 1,
        max_supported_force: float = 0.0,
        max_holding_force: float = 0.0,
        is_tower: bool = False,
    ):
        self.attachment_height = attachment_height
        self.xy_location = xy_location
        self.max_deviation = max_deviation
        self.is_tower = is_tower
        self.max_supported_force = max_supported_force
        self.max_holding_force = max_holding_force

        # get the elevation of the floor below the support
        self.floor_height = geometry_operations.fetch_point_elevation(
            self.xy_location, height_gdf, self.max_deviation
        )

    @property
    def total_height(self):
        return self.floor_height + self.attachment_height


class Cable_Road:
    def __init__(
        self,
        line,
        height_gdf,
        left_support: Support,
        right_support: Support,
        pre_tension=0,
        number_sub_segments=0,
    ):
        self.left_support: Support = left_support
        self.right_support: Support = right_support

        """heights"""
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

        # Parameters to keep track of segments+
        self.number_sub_segments = number_sub_segments
        self.supported_segments: list[
            SupportedSegment
        ] = []  # list of SupportedSegment objects, ie. sub cable roads

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

        # get the height of those points and set them as attributes to the CR object
        self.compute_floor_height_below_line_points(height_gdf)
        # generate floor points and their distances
        self.floor_points = list(
            zip(
                [point.x for point in self.points_along_line],
                [point.y for point in self.points_along_line],
                self.floor_height_below_line_points,
            )
        )

        # set up further rope parameters
        self.b_length_whole_section = self.start_point.distance(self.end_point)
        self.c_rope_length = geometry_utilities.distance_between_3d_points(
            self.line_start_point_array, self.line_end_point_array
        )

        self.initialize_line_tension(number_sub_segments, pre_tension)

        # and finally the loaded and unlaoded line to floor distances
        self.compute_loaded_unloaded_line_height()

    # Computing the line to floor distances
    @property
    def line_start_point_array(self):
        return np.array(
            [
                self.start_point.x,
                self.start_point.y,
                self.left_support.total_height,
            ]
        )

    @property
    def line_end_point_array(self):
        return np.array(
            [
                self.end_point.x,
                self.end_point.y,
                self.right_support.total_height,
            ]
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
    def absolute_unloaded_line_height(self):
        return (
            self.floor_height_below_line_points + self.unloaded_line_to_floor_distances
        )

    @property
    def absolute_loaded_line_height(self):
        return self.floor_height_below_line_points + self.sloped_line_to_floor_distances

    def count_segments(self, number_sub_segments) -> int:
        """recursively counts the number of segments in the cable road"""
        if self.supported_segments:
            number_sub_segments += 2
            for segment in self.supported_segments:
                number_sub_segments += segment.cable_road.count_segments(
                    number_sub_segments
                )

            return number_sub_segments
        else:
            return number_sub_segments

    def compute_floor_height_below_line_points(self, height_gdf: gpd.GeoDataFrame):
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

    def compute_loaded_unloaded_line_height(self):
        """compute the loaded and unloaded line to floor distances"""
        self.calculate_cr_deflections(loaded=True)
        self.calculate_cr_deflections(loaded=False)

    def calculate_cr_deflections(self, loaded: bool = True):
        """calculate the deflections of the CR line due to the load, either loaded or unlaoded
        Args:
            loaded (bool, optional): whether the line is loaded or not. Defaults to True.
        """

        y_x_deflections = np.asarray(
            [
                mechanical_computations.pestal_load_path(self, point, loaded)
                for point in self.points_along_line
            ],
            dtype=np.float32,
        )

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
        left_support: Support,
        right_support: Support,
    ):
        self.cable_road = cable_road
        self.left_support = left_support
        self.right_support = right_support
