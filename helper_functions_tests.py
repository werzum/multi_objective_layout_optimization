import pandas as pd
from shapely.geometry import LineString, Point

import geometry_utilities, geometry_operations


def parse_point(point_string):
    # Remove the "POINT " prefix and parentheses from the string
    point_string = point_string.replace("POINT (", "").replace(")", "")
    # Split the string into two coordinates
    x, y = point_string.split()
    # Convert the coordinates to floats and create a new Point object
    return Point(float(x), float(y))


def load_sample_gdfs():
    with open("Resources_Organized/tree_gdf_export.csv") as file:
        bestand = pd.read_csv(file)

    with open("Resources_Organized/height_df.csv") as file:
        height = pd.read_csv(file)

    # Apply the function to the 'point' column using the apply method
    bestand["geometry"] = bestand["geometry"].apply(parse_point)
    height["geometry"] = height["geometry"].apply(parse_point)

    return bestand, height


def create_cable_road():
    bestand_gdf, height_gdf = load_sample_gdfs()

    """ the points are saved as strings and therefore have to be loaded differently - no intention to fiddle around with this now """
    possible_line = LineString(
        [bestand_gdf.iloc[1].geometry, bestand_gdf.iloc[100].geometry]
    )
    this_cable_road = geometry_operations.compute_initial_cable_road(
        possible_line, height_gdf
    )

    return this_cable_road
