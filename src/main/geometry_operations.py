from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
import numpy as np
import geopandas as gpd

from src.main import global_vars


def generate_road_points(
    road_geometry: LineString, interval: int
) -> list[BaseGeometry]:
    """Generate a list of points with a given interval along the road geometry

    Args:
        road_geometry (_type_): A list of lines that define the road
        interval: The interval in which a new point is calculated

    Returns:
        _type_: _description_
    """
    # thanks to https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
    distance_delta = interval
    distances = np.arange(0, road_geometry.length, distance_delta)

    return [road_geometry.interpolate(distance) for distance in distances] + [
        Point(road_geometry.coords[1])
    ]


def compute_points_covered_by_geometry(
    points_gdf: gpd.GeoDataFrame, geometry: Polygon, min_trees_covered: int
) -> tuple[set, int]:
    """Return the points covered by geometry in the points_gdf

    Returns:
        set(geometry), int : set of covered points as well as their amount
    """

    contained_points = filter_gdf_by_contained_elements(points_gdf, geometry)

    if len(contained_points) < min_trees_covered:
        raise ValueError("Not enough trees covered by this geometry")
    # filter only those points
    # and return and set of the covered points as well as the amount of trees covered
    return set(contained_points["id"].values), len(contained_points)


def compute_points_covered_per_row(
    points_gdf: gpd.GeoDataFrame,
    row_gdf: gpd.GeoDataFrame,
    buffer_size: int,
    min_trees_covered: int,
):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """

    # already create buffer to avoid having to recreate this object every time
    row_gdf["buffer"] = row_gdf.apply(
        lambda row: row.geometry.buffer(buffer_size), axis=1
    )

    # appply and return the points covered by each buffer
    return row_gdf["buffer"].apply(
        lambda row: compute_points_covered_by_geometry(
            points_gdf, row, min_trees_covered
        )
    )


def filter_gdf_by_contained_elements(gdf: gpd.GeoDataFrame, polygon: Polygon):
    """Return only the points in the gdf which are covered by the polygon

    Args:
        gdf (_type_): The gdf to filter
        polygon (_type_): A polygon geometry

    Returns:
        _type_: the filtered gdf
    """
    # get the points which are contained in the geometry
    coverage_series = gdf.geometry.intersects(polygon)
    # and select only those points from the point_gdf
    contained_points = gdf[coverage_series]

    return contained_points


def compute_distances_facilities_clients(
    tree_gdf: gpd.GeoDataFrame, line_gdf: gpd.GeoDataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Create a numpy matrix with the distance between every tree and line

    Args:
        tree_gdf (_type_): A gdf containing the trees
        line_gdf (_type_): A gdf containing the facilities/lines

    Returns
        tuple[np.ndarray, np.ndarray]: A tuple of two matrices containing the distances and the distance from
    """
    # compute the distance to each tree for every row
    tree_line_distances = []
    carriage_support_distances = []

    for index, line in line_gdf.iterrows():
        # get the distance from all trees to this line
        line_tree_distance = tree_gdf.geometry.distance(line.geometry)

        # get the nearest point between the tree and the cable road for all trees
        # project(tree,line)) gets the distance of the closest point on the line to the tower  -
        # ie. how far along the line do I have to move the carriage to get closest to the tree
        carriage_support_distance = [
            line.geometry.project(tree_geometry) for tree_geometry in tree_gdf.geometry
        ]

        tree_line_distances.append(line_tree_distance)
        carriage_support_distances.append(carriage_support_distance)

    # pivot the table and convert to numpy matrix (solver expects it this way)
    return (
        np.asarray(tree_line_distances).transpose(),
        np.asarray(carriage_support_distances).transpose(),
    )


def fetch_point_elevation(
    point: Point, height_gdf: gpd.GeoDataFrame, max_deviation: float
) -> float:
    """
    Fetches the elevation of a given point.

    Args:
    point (Point): The point for which the elevation is to be fetched.
    height_gdf (GeoDataFrame): A GeoDataFrame containing the elevations.
    max_deviation (float): The maximum deviation allowed while fetching the elevation.

    Returns:
    float: The elevation of the given point.
    """
    # if point.x is None or point.y is None:
    #     print("Point has no coordinates")
    # return height_gdf[
    #     (height_gdf["x"].between(point.x - max_deviation, point.x + max_deviation))
    #     & (height_gdf["y"].between(point.y - max_deviation, point.y + max_deviation))
    # ]["elev"].values[0]

    d, i = global_vars.kdtree.query(point.coords, k=1)
    # Use the final condition to filter the height_gdf and get the elev values
    return height_gdf.iloc[i]["elev"].values[0]


def compute_average_terrain_steepness(
    line_gdf: gpd.GeoDataFrame, height_gdf: gpd.GeoDataFrame
) -> float:
    """
    Computes the average terrain steepness of a given line.

    Args:
    line_gdf (GeoDataFrame): A GeoDataFrame containing the line.
    height_gdf (GeoDataFrame): A GeoDataFrame containing the elevations.

    Returns:
    float: The average terrain steepness of the given line in percent (0-100)
    """

    averages = np.empty(len(line_gdf))

    for index in range(len(line_gdf) - 1):
        cable_road = line_gdf.iloc[index]["Cable Road Object"]
        averages[index] = get_slope(cable_road)

    return sum(averages) / len(averages)


def get_slope(cable_road) -> float:
    """Get the slope of a given cable road
    Args:
        cable_road (classes.Cable_Road): Cable road object
    Returns:
        float: Slope of the cable road in %degrees

    """

    absolute_height_difference = abs(
        cable_road.end_support.total_height - cable_road.start_support.total_height
    )
    cable_road_xy_length = cable_road.b_length_whole_section
    steepness = absolute_height_difference / cable_road_xy_length
    return steepness * 100
