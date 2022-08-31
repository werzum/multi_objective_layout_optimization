import geometry_utilities

from shapely.geometry import LineString
import numpy as np
import pandas as pd

def generate_possible_lines(road_points, target_trees, anchor_trees, slope_line, max_deviation):
    """ Compute which lines can be made from road_points to anchor_trees without having an angle greater than max_deviation
    Takes buffer size and minimum number of trees covered

    Args:
        road_points (_type_): Array of points
        anchor_trees (_type_): Array of points
        slope_line (_type_): A line
        max_deviation: An int of the max deviation between possible line and slope line

    Returns:
        _type_: _description_
    """
    possible_lines = []
    slope_deviation = []

    for point in road_points:
        for target in target_trees.geometry:
            possible_line = LineString([point, target])
            angle = geometry_utilities.angle_between(possible_line, slope_line)

### here, we should check whether 3 anchor trees within given distance and angle are within reach - 1. compute within distance, 2. check angle

            if geometry_utilities.within_maximum_rotation(angle, max_deviation):
                possible_lines.append(possible_line)
                slope_deviation.append(angle)

    return possible_lines, slope_deviation

def generate_road_points(road_geometry, interval):
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
    road_points = [road_geometry.interpolate(distance) for distance in distances] + [road_geometry.boundary[1]]

    return road_points

def compute_points_covered_by_geometry(points_gdf, geometry, min_trees_covered):
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


def compute_points_covered_per_row(points_gdf, row_gdf, buffer_size, min_trees_covered):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """

    # already create buffer to avoid having to recreate this object every time
    row_gdf["buffer"] = row_gdf.apply(
        lambda row: row.geometry.buffer(buffer_size), axis=1)

    # appply and return the points covered by each buffer
    return row_gdf["buffer"].apply(lambda row: compute_points_covered_by_geometry(points_gdf, row, min_trees_covered))

def filter_gdf_by_contained_elements(gdf, polygon):
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

def compute_distances_facilities_clients(tree_gdf, line_gdf):
    """Create a numpy matrix with the distance between every tree and line

    Args:
        tree_gdf (_type_): A gdf containing the trees
        line_gdf (_type_): A gdf containing the facilities/lines

    Returns:
        _type_: A numpy matrix of the costs/distances
    """    
    # compute the distance to each tree for every row
    distances = []
    for line in line_gdf.iterrows():
        line_tree_distance = tree_gdf.geometry.distance(line[1].geometry)
        distances.append(line_tree_distance)

    # pivot the table and convert to numpy matrix (solver expects it this way)
    return np.asarray(distances).transpose()