import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
from spopt.locate import PMedian


from src.main import classes


def set_up_gdfs():
    # load the preprocessed data
    # bestand_gdf, height_gdf = data_loading.load_processed_gdfs()
    line_gdf = load_pickle("line_gdf_reworked")
    height_gdf = load_pickle("height_gdf")
    tree_gdf = load_pickle("tree_gdf")

    return line_gdf, tree_gdf, height_gdf


def load_pickle(var_name: str):
    path = f"~/.ipython/profile_default/db/autorestore/{var_name}"
    return pd.read_pickle(path)


def model_to_line_gdf(
    optimized_model: PMedian, line_gdf: gpd.GeoDataFrame
) -> (gpd.GeoDataFrame, pd.Series):
    """This function takes the optimized model and the line gdf and returns the selected lines and the cable road objects
    Args:
        optimized_model (PMedian): The optimized model
        line_gdf (gpd.GeoDataFrame): The line gdf
    Returns:
        gpd.GeoDataFrame: The selected lines
        pd.Series: The extracted cable road objects
    """

    # get the positive facility variables and select those from the line gdf
    fac_vars = [bool(var.value()) for var in optimized_model.fac_vars]
    selected_lines = line_gdf[fac_vars]

    cable_road_objects = selected_lines.loc[:, "Cable Road Object"]

    return selected_lines, cable_road_objects
