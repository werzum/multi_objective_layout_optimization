from shapely.geometry import LineString, Point
from shapely.affinity import rotate, scale
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.cluster import KMeans

# to stop the shapely deprecation warning about the coords interface
import warnings
from shapely.errors import ShapelyDeprecationWarning

from src.main import (
    classes_geometry_objects,
    geometry_operations,
    geometry_utilities,
    global_vars,
    cable_road_computation_main,
    mechanical_computations_separate_dependencies,
    optimization_compute_quantification,
)

# from src.main.geometry_operations import (
#     fetch_point_elevation,
#     generate_road_points,
# )


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
        return classes_geometry_objects.Point_3D(
            self.xy_location.x, self.xy_location.y, self.total_height
        )

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
        self.supported_segments: list[SupportedSegment] = (
            []
        )  # list of SupportedSegment objects, ie. sub cable roads

        self._s_current_tension = 0.0

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

        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        y_x_deflections = (
            mechanical_computations_separate_dependencies.pestal_load_path(self, loaded)
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


class forest_area:
    def __init__(self, tree_gdf, height_gdf, extra_geometry_gpd):
        """
        # A class to represent a forest area, storing the tree and height dataframes, as well as the extra geometry dataframes.
        """
        self.tree_gdf = tree_gdf
        self.height_gdf = height_gdf

        interval = 2
        self.road_points = geometry_operations.generate_road_points(
            extra_geometry_gpd.loc["road"].geometry, interval
        )

        # get the eligible anchor and target trees inside the polygon
        anchor_trees_gdf = geometry_operations.filter_gdf_by_contained_elements(
            tree_gdf, extra_geometry_gpd.loc["uphill_anchors"].geometry
        )
        target_trees_gdf = geometry_operations.filter_gdf_by_contained_elements(
            tree_gdf, extra_geometry_gpd.loc["downhill_anchors"].geometry
        )
        inner_forest_gdf = geometry_operations.filter_gdf_by_contained_elements(
            tree_gdf, extra_geometry_gpd.loc["inner_forest"].geometry
        )

        self.harvesteable_trees_gdf = pd.concat([target_trees_gdf, inner_forest_gdf])

        # filter the anchor and target trees for a BHD of 30cm or larger
        self.anchor_trees_gdf = anchor_trees_gdf[anchor_trees_gdf["BHD"] >= 30]
        self.target_trees_gdf = target_trees_gdf[target_trees_gdf["BHD"] >= 30]

        # set the slope
        slope_degree = 29

        # set a orientation line we can plan the line deviation around
        slope_line = LineString([(0, 0), (0, 1)])
        slope_line = rotate(slope_line, slope_degree)

        # scale the line by a factor of 100 and plot it
        self.slope_line = scale(slope_line, 100, 100)

    def compute_cable_road(self):
        print("loaded new")
        (
            line_gdf,
            start_point_dict,
        ) = cable_road_computation_main.generate_possible_lines(
            self.road_points,
            self.target_trees_gdf,
            self.anchor_trees_gdf,
            self.tree_gdf,
            self.slope_line,
            self.height_gdf,
        )
        print("we have n lines: ", len(line_gdf))

        # unpack to geopandas
        buffer = gpd.GeoSeries(line_gdf["line_candidates"].values)
        new_line_gdf = gpd.GeoDataFrame(geometry=buffer)
        new_line_gdf[line_gdf.columns] = line_gdf[line_gdf.columns].values
        new_line_gdf["line_length"] = new_line_gdf.geometry.length
        self.line_gdf = new_line_gdf
        self.start_point_dict = start_point_dict

    def compute_line_costs(self):
        # compute the line costs
        uphill_yarding = True
        self.line_gdf["line_cost"] = (
            optimization_compute_quantification.compute_line_costs(
                self.line_gdf, uphill_yarding, large_yarder=True
            )
        )

        # and the volume of the harvesteable trees
        self.harvesteable_trees_gdf["cubic_volume"] = (
            optimization_compute_quantification.compute_tree_volume(
                self.harvesteable_trees_gdf["BHD"], self.harvesteable_trees_gdf["h"]
            )
        )

    def cluster_trees(self):
        n_clusters = len(self.harvesteable_trees_gdf) // 20
        k_means = KMeans(n_clusters=n_clusters)
        predicted_clusters = k_means.fit_predict(
            X=[*zip(self.harvesteable_trees_gdf["x"], self.harvesteable_trees_gdf["y"])]
        )

        return n_clusters, predicted_clusters

    def a_value_selection(self, A_value=5):
        """Selects the trees to fell based on the A value and sets the harvesteable_trees_gdf_sortiment attribute
        Args:
            A_value (int, optional): The A value. Defaults to 5.
        """
        indexes_to_keep = []

        n_clusters, predicted_clusters = self.cluster_trees()

        # doing this too many times
        for cluster in range(n_clusters):
            # select trees
            index = np.where(predicted_clusters == cluster)
            sub_df = self.harvesteable_trees_gdf.iloc[index]

            # determine index of largest tree
            z_tree_index = sub_df["BHD"].idxmax()
            z_tree = sub_df.loc[z_tree_index]
            z_tree_bhd = z_tree["BHD"]
            z_tree_h = z_tree["h"]

            # compute the distance of all trees to the largest tree
            sub_df["distance_to_z_tree"] = sub_df.distance(
                sub_df.loc[z_tree_index].geometry
            )

            # drop the largest tree from the dataframe so we dont fell it
            sub_df.drop(z_tree_index, inplace=True)

            # # A=H/E * d/D
            # sub_df["A_value"] = (z_tree_h / sub_df["distance_to_z_tree"]) * (
            #     sub_df["BHD"] / z_tree_bhd
            # )

            # select those trees which are too close to the z tree with GD < H/A*d/D
            # negate the condition, since those are the trees we keep. The trees in the list will be felled
            sub_df = sub_df[
                ~(
                    sub_df["distance_to_z_tree"]
                    < (z_tree_h / A_value) * (sub_df["BHD"] / z_tree_bhd)
                )
            ]

            # add the indexes to fell to the list
            indexes_to_keep.extend(sub_df.index)

        # filter the harvesteable trees to only those we want to fell
        self.harvesteable_trees_gdf_sortiment = self.harvesteable_trees_gdf.loc[
            indexes_to_keep
        ]
