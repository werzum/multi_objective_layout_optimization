import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point

from src.main import classes


def set_up_gdfs():
    # load the preprocessed data
    # bestand_gdf, height_gdf = data_loading.load_processed_gdfs()
    line_gdf = load_pickle("line_gdf_full")
    height_gdf = load_pickle("height_gdf")
    tree_gdf = load_pickle("tree_gdf")

    return line_gdf, tree_gdf, height_gdf


def load_cable_road(
    line_gdf: gpd.GeoDataFrame, height_gdf: gpd.GeoDataFrame, index: int
):
    """Helper function to abstract setting up a sample cable road from the line_gdf"""
    return classes.set_up_CR_from_linegdf(line_gdf, index, height_gdf)


# TODO - not necessary anymore since I aim to store the segments in the DF
# def create_cable_road_segments(
#     cable_road: classes.Cable_Road, line_gdf: gpd.GeoDataFrame, height_gdf: gpd.GeoDataFrame, index: int,
# ) -> list[classes.Cable_Road]:
#     """Helper function to abstract setting up the supported segments of a sample cable road from the line_gdf
#     Args:
#         line_gdf (gpd.GeoDataFrame): the line_gdf
#         height_gdf (gpd.GeoDataFrame): the height_gdf
#         index (int): the index of the line_gdf to use

#     Returns:
#         list[Cable_Road]: a list of the cable road segments
#     """
#     # get the waypoints
#     start_point = Point(line_gdf.iloc[index].geometry.coords[0])
#     end_point = line_gdf.iloc[index].geometry.coords[1]
#     supports = line_gdf.iloc[index].location_of_int_supports

#     cable_road_segment_list = []
#     # for all individual road segments in triples
#     waypoints = [start_point, *supports, end_point]
#     for previous, support, current in zip(waypoints, waypoints[1:], waypoints[2:]):

#         # create our two sub cable roads
#         road_to_support_cable_road = classes.Cable_Road(
#         LineString([previous, support]), height_gdf, cable_road.s_current_tension, 0)

#         support_to_anchor_cable_road = classes.Cable_Road(
#         LineString([support, current]), height_gdf, cable_road.s_current_tension, 0)

#         segment = classes.SupportSegment(
#         road_to_support_cable_road,
#         support_to_anchor_cable_road,
#         candidate_tree,
#     )
#         cable_road_segment_list.append(sample_cable_road)

#     return cable_road_segment_list


def load_pickle(var_name: str):
    path = f"~/.ipython/profile_default/db/autorestore/{var_name}"
    return pd.read_pickle(path)
