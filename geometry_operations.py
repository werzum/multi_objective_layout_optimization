from shapely.geometry import LineString, Point, Polygon
import numpy as np
import geopandas as gpd


def generate_road_points(road_geometry: LineString, interval: int) -> list[Point]:
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
    road_points = [road_geometry.interpolate(distance) for distance in distances] + [
        Point(road_geometry.coords[1])
    ]

    return road_points


def compute_points_covered_by_geometry(
    points_gdf: gpd.GeoDataFrame, geometry: Polygon, min_trees_covered: int
) -> tuple[set, int]:
    """Return the points covered by geometry in the points_gdf

    Returns:
        set(geometry), int : set of covered points as well as their amount
    """

    contained_points = filter_gdf_by_contained_elements(points_gdf, geometry)

    if len(contained_points) < min_trees_covered:
        return
    # filter only those points
    # and return and set of the covered points as well as the amount of trees covered
    return set(contained_points["id"].values), len(contained_points)


def compute_points_covered_per_row(
    points_gdf: gpd.GeoDataFrame,
    row_gdf: gpd.GeoDataFrame,
    buffer_size: int,
    min_trees_covered: int,
) -> gpd.GeoDataFrame:
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


def filter_gdf_by_contained_elements(
    gdf: gpd.GeoDataFrame, polygon: Polygon
) -> gpd.GeoDataFrame:
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
) -> tuple(np.ndarray, np.ndarray):
    """Create a numpy matrix with the distance between every tree and line

    Args:
        tree_gdf (_type_): A gdf containing the trees
        line_gdf (_type_): A gdf containing the facilities/lines

    Returns:
        _type_: A numpy matrix of the costs/distances
    """
    # compute the distance to each tree for every row
    tree_line_distances = []
    carriage_support_distances = []

    for line in line_gdf.iterrows():
        line_tree_distance = tree_gdf.geometry.distance(line[1].geometry)
        # get the nearest point between the tree and the cable road for all trees
        # project(tree,line)) gets the distance of the closest point on the line
        carriage_support_distance = [
            line[1].geometry.project(Point(tree_geometry.coords[0]))
            for tree_geometry in tree_gdf.geometry
        ]

        tree_line_distances.append(line_tree_distance)
        carriage_support_distances.append(carriage_support_distance)

    # pivot the table and convert to numpy matrix (solver expects it this way)
    return (
        np.asarray(tree_line_distances).transpose(),
        np.asarray(carriage_support_distances).transpose(),
    )
