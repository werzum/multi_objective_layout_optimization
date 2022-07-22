import numpy as np
import math

from shapely.geometry import LineString, Polygon

def get_contained_elements(gdf, polygon):
    return list(filter(lambda x: polygon.contains(x), gdf.geometry))

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    # extract their coords for vector
    v1 = [v1.coords[0], v1.coords[1]]
    v2 = [v2.coords[0], v2.coords[1]]

    # and recompute as vector
    # thanks for reminding me https://discuss.codechef.com/t/how-to-find-angle-between-two-lines/14516
    v1 = (v1[1][0]-v1[0][0],v1[1][1]-v1[0][1])
    v2 = (v2[1][0]-v2[0][0],v2[1][1]-v2[0][1])

    # get the unit vector, dot product and then the arccos from that
    unit_vector_1, unit_vector_2 = unit_vector(v1), unit_vector(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    # in radians
    angle = np.arccos(dot_product)
    # to degrees - https://stackoverflow.com/questions/9875964/how-can-i-convert-radians-to-degrees-with-python
    return math.degrees(angle)

def generate_possible_lines(road_points,anchor_trees,slope_line,max_deviation):
    """ Compute which lines can be made from road_points to anchor_trees without having an angle greater than max_deviation
    Args:
        road_points (_type_): Array of points
        anchor_trees (_type_): Array of points
        slope_line (_type_): A line
        max_deviation: An int of the max deviation between possible line and slope line

    Returns:
        _type_: _description_
    """    
    possible_lines = []
    for point in road_points:
        for anchor in anchor_trees:
            possible_line = LineString([point,anchor])
            if angle_between(possible_line, slope_line) < max_deviation:
                possible_lines.append(possible_line)
    
    return possible_lines

def gpd_contains(point, area):
    return area.contains(point)

def create_geometry(geometry,buffer_size):
    return geometry.buffer(buffer_size)

def get_points_covered(points_gdf,geometry):

    # get the points which are contained in the geometry
    coverage_series = points_gdf.geometry.apply(lambda row: gpd_contains(row,geometry))
    points_covered = points_gdf.loc[coverage_series,:]

    # filter only those points
    # and return and set of the covered points as well as the amount of trees covered
    return set(points_covered["id"].values),points_covered.size

def compute_points_covered_per_row(points_gdf,row_gdf,buffer_size):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """

    #already create buffer to avoid having to recreate this object every time 
    row_gdf["buffer"] = row_gdf.apply(lambda row: row.geometry.buffer(buffer_size),axis=1)
    
    #appply and return the points covered by each buffer
    return row_gdf["buffer"].apply(lambda row: get_points_covered(points_gdf,row))