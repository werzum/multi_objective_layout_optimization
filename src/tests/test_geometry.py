import numpy as np
import pandas as pd
import math
import pytest

from shapely.geometry import LineString, Point

from src.main import (
    geometry_utilities,
    geometry_operations,
    mechanical_computations,
    classes,
)

from src.tests import helper_functions, test_cable_roads


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


def test_cr_parameter_feasability():
    (
        cable_road,
        line_gdf,
        tree_gdf,
        height_gdf,
    ) = test_cable_roads.main_test_cable_roads()

    tree_0 = tree_gdf[tree_gdf["BHD"] > 40].iloc[0]
    assert (
        len(tree_gdf[tree_gdf["BHD"] > 40]) > 200
    )  # make sure we have enough strong trees
    assert tree_0["max_holding_force"] > 50.000
    assert tree_0["max_supported_force_series"][6] > 50.000

    # test if the cable road works on (simulted) strong anchors
    assert (
        mechanical_computations.check_if_tower_and_anchor_trees_hold(
            cable_road, [50000, 50000, 50000], [cable_road.anchor_triplets], height_gdf
        )
        == True
    )


def test_rotation():
    # ensure that if we rotate the z axis down at x and y, we get a lower z
    v = classes.Point_3D(0, 0, 1)
    v_prime = geometry_utilities.rotate_3d_point_in_z_direction(v, 20)
    np.testing.assert_allclose(v_prime.xyz, np.array([0.3, 0.3, 0.9]), atol=1e-1)

    # ensure that if our point also has a x and y component, it is also rotated
    v = classes.Point_3D(1, 1, 1)
    v_prime = geometry_utilities.rotate_3d_point_in_z_direction(v, 20)
    np.testing.assert_allclose(v_prime.xyz, np.array([1.2, 1.2, 0.5]), atol=1e-1)
