import pandas as pd
import geopandas as gpd

from src.main import classes_cable_road_computation, optimization_compute_quantification


def set_up_gdfs():
    # load the preprocessed data
    # bestand_gdf, height_gdf = data_loading.load_processed_gdfs()
    # line_gdf = load_pickle("line_gdf_reworked")
    height_gdf = load_pickle("height_gdf")
    tree_gdf = load_pickle("tree_gdf")

    return tree_gdf, height_gdf


def load_pickle(var_name: str):
    path = f"~/.ipython/profile_default/db/autorestore/{var_name}"
    return pd.read_pickle(path)
