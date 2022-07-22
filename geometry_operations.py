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

def compute_points_covered_per_row(points_gdf,row_gdf,buffer_size):
    """Compute how many points are covered per row.geometry in the points_gdf
    Args:
        points_gdf (_type_): A gdf with a list of point geometries
        row_gdf (_type_): The gdf containing lines where we check how many points are covered
        buffer_size: The width added to the row_gdf.geometry entry
    """
    id_covered = np.empty(len(row_gdf), dtype=object)
    amount_covered = np.empty(len(row_gdf), dtype=object)
    for index,row in row_gdf.iterrows():
        buffer = Polygon(row.geometry.buffer(buffer_size))
        # check where the buffer contains the geometry (point) of the tree on a per row basis. The .vectorize constructs a boolean series where gpd_contains
        # is true, and then use .loc to only keep those entries
        points_covered = points_gdf.loc[np.vectorize(gpd_contains)(points_gdf.geometry,buffer),:]
        id_covered[index] = set(points_covered["id"].values)
        amount_covered[index] = points_covered.size
    
    return id_covered,amount_covered