from operator import truediv
import numpy as np
import math
import warnings

from shapely.geometry import LineString, Point


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def value_within_range(min, max, distance):
    if distance > min and distance < max:
        return True
    else:
        return False


def print_point_coordinates(point_list: list[Point]):
    for point in point_list:
        print(point[0], point[1])


def print_line_coordinates(line_list: list[LineString]):
    for line in line_list:
        print_point_coordinates([line.coords[0]])
        print_point_coordinates([line.coords[1]])


def angle_between(v1: LineString, v2: LineString) -> float:
    """Returns the angle between to 2d vectors. Returns 0 to 180 degrees angles - note that the direction of the vector matters!
    Will however not discern between a -20 and 20 rotation wrt the v1.

    Args:
        v1 (_type_): _description_
        v2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    # extract their coords for vector
    v1 = [v1.coords[0], v1.coords[1]]
    v2 = [v2.coords[0], v2.coords[1]]

    # and recompute as vector
    # thanks for reminding me https://discuss.codechef.com/t/how-to-find-angle-between-two-lines/14516
    v1 = (v1[1][0] - v1[0][0], v1[1][1] - v1[0][1])
    v2 = (v2[1][0] - v2[0][0], v2[1][1] - v2[0][1])

    # get the unit vector, dot product and then the arccos from that
    unit_vector_1, unit_vector_2 = unit_vector(v1), unit_vector(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)

    if -1 < dot_product < 1:
        # in radians
        angle = np.arccos(dot_product)
        # to degrees - https://stackoverflow.com/questions/9875964/how-can-i-convert-radians-to-degrees-with-python
        return math.degrees(angle)
    else:
        # we return a large angle?
        return 90


def within_maximum_rotation(angle, max_deviation):
    """Check if the angle between the slope line and possible line is too great.
    This checks several cases, but the angles don't seem to be <20 anyways really.

    Returns:
        Truth Value: If the rotation is within the max deviation
    """
    # if angle is smaller than max_dev or greater than 360-max_dev
    condition1 = True if angle < max_deviation or angle > 360 - max_deviation else False
    # if flipped line is less than max_deviation+180
    condition2 = True if (180 - max_deviation) < angle < 180 + max_deviation else False

    return condition1 or condition2


def area_contains(area, point):
    return area.contains(point)


def create_buffer(geometry, buffer_size):
    return geometry.buffer(buffer_size)


def lineseg_dist(p, a, b):
    """Function lineseg_dist returns the distance the distance from point p to line segment [a,b]. p, a and b are np.arrays.

    Taken from SO https://stackoverflow.com/questions/56463412/distance-from-a-point-to-a-line-segment-in-3d-python

    Args:
        p (_type_): _description_
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def angle_between_3d(v1, v2):
    """
    Calculates the angle between two 3D vectors.

    Args:
        v1: The first 3D vector.
        v2: The second 3D vector.

    Returns:
        The angle between the two vectors in degrees.
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def distance_between_3d_points(point1, point2):
    """Compute distance between two 3d points

    Args:
        point1 (_type_): Numpy array of coordinates
        point2 (_type_): Numpy array of coordinates

    Returns:
        _type_: float of distance
    """
    squared_dist = np.sum((point1 - point2) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist
