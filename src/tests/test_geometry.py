import numpy as np
import pandas as pd
import math
import pytest

from shapely.geometry import LineString, Point

from src import (
    geometry_utilities,
    geometry_operations,
    helper_functions_tests,
    mechanical_computations,
)


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


def test_euler_knicklast():
    # testing different variations from the Tragkraft von Stützenbäumen as per Pestal
    assert np.isclose(mechanical_computations.euler_knicklast(19, 8), 1500, rtol=0.2)

    assert np.isclose(mechanical_computations.euler_knicklast(30, 20), 1500, rtol=0.2)

    assert np.isclose(mechanical_computations.euler_knicklast(50, 20), 12000, rtol=0.20)

    assert np.isclose(mechanical_computations.euler_knicklast(50, 40), 3000, rtol=0.20)

    assert np.isclose(mechanical_computations.euler_knicklast(40, 10), 19000, rtol=0.20)
