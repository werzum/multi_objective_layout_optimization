import pandas as pd
import geopandas as gpd
from main import optimization_compute_quantification

from main import classes_cable_road_computation


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
    optimized_object: classes_cable_road_computation.optimization_object,
    line_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, pd.Series]:
    """This function takes the optimized model and the line gdf and returns the selected lines and the cable road objects
    Args:
        optimized_object: The optimized model
        line_gdf (gpd.GeoDataFrame): The line gdf
    Returns:
        gpd.GeoDataFrame: The selected lines
        pd.Series: The extracted cable road objects
    """

    # get the positive facility variables and select those from the line gdf
    if hasattr(optimized_object, "model"):
        fac_vars = optimization_compute_quantification.get_fac_vars(optimized_object)
        selected_lines = line_gdf[fac_vars]
    else:
        selected_lines = line_gdf[optimized_object.fac_vars]

    return selected_lines, selected_lines["Cable Road Object"]
