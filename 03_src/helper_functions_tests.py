import pandas as pd
from shapely.geometry import LineString, Point

import cable_road_computation, data_loading


def create_cable_road():
    bestand_gdf, height_gdf = data_loading.load_processed_gdfs()

    """ the points are saved as strings and therefore have to be loaded differently - no intention to fiddle around with this now """
    possible_line = LineString(
        [bestand_gdf.iloc[1].geometry, bestand_gdf.iloc[100].geometry]
    )
    this_cable_road = cable_road_computation.compute_initial_cable_road(
        possible_line, height_gdf
    )

    return this_cable_road
