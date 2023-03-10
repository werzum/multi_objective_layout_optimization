import numpy as np
import pandas as pd
import math
import pytest

from shapely.geometry import LineString, Point

import geometry_utilities, geometry_operations, helper_functions_tests


def test_angle_between_3d():
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    assert math.isclose(geometry_utilities.angle_between_3d(v1, v2), 90, rel_tol=1e-5)

    v1 = np.array([1, 0, 0])
    v2 = np.array([1, 0, 0])
    assert math.isclose(geometry_utilities.angle_between_3d(v1, v2), 0, rel_tol=1e-5)

    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 0, 1])
    assert math.isclose(geometry_utilities.angle_between_3d(v1, v2), 90, rel_tol=1e-5)

    v1 = np.array([1, 2, 3])
    v2 = np.array([-4, 5, 6])
    assert math.isclose(
        geometry_utilities.angle_between_3d(v1, v2), 43.03, rel_tol=1e-3
    )


def test_distance_between_3d_points():
    point1 = np.array([1, 0, 0])
    point2 = np.array([0, 1, 0])
    assert np.isclose(
        geometry_utilities.distance_between_3d_points(point1, point2), np.sqrt(2)
    )

    point1 = np.array([0, 0, 0])
    point2 = np.array([0, 0, 0])
    assert np.isclose(geometry_utilities.distance_between_3d_points(point1, point2), 0)

    point1 = np.array([1, 2, 3])
    point2 = np.array([-4, 5, 6])
    assert np.isclose(
        geometry_utilities.distance_between_3d_points(point1, point2), np.sqrt(43)
    )

    # Define a function to convert a string to a Shapely Point object


def test_cable_road_creation():
    this_cable_road = helper_functions_tests.create_cable_road()

    # pytest.set_trace()

    assert np.isclose(
        this_cable_road.start_point_height,
        -29.609 + this_cable_road.support_height,
        atol=1,
    )

    assert np.isclose(
        this_cable_road.end_point_height,
        -66.5252 + this_cable_road.support_height,
        atol=1,
    )

    # fetch the floor points along the line
    this_cable_road.points_along_line = generate_road_points(possible_line, interval=2)

    # get the height of those points and set them as attributes to the CR object
    this_cable_road.compute_line_height(height_gdf)

    # generate floor points and their distances
    assert typeof(this_cable_road.floor_points) == list
    assert typeof(this_cable_road.line_to_floor_distances) == list

    # test whether start point != end point and lenght > 0

    # get the rope length
    this_cable_road.b_length_whole_section = this_cable_road.start_point.distance(
        this_cable_road.end_point
    )

    this_cable_road.c_rope_length = geometry_utilities.distance_between_3d_points(
        this_cable_road.line_start_point_array, this_cable_road.line_end_point_array
    )

    print(cable_road)
