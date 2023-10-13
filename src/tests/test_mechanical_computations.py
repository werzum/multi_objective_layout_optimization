from main import classes_cable_road_computation
from shapely.geometry import Point, LineString
import geopandas as gpd
from itertools import pairwise
import numpy as np

from src.main import (
    geometry_utilities,
    mechanical_computations,
    cable_road_computation,
)

from src.tests import helper_functions


def test_check_if_no_collisions_cable_road(line_gdf: gpd.GeoDataFrame):
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 2)
    assert cable_road.count_segments() == 0

    cable_road.s_current_tension = 20000
    mechanical_computations.check_if_no_collisions_cable_road(cable_road)
    assert cable_road.no_collisions == False

    cable_road.s_current_tension = 120000
    mechanical_computations.check_if_no_collisions_cable_road(cable_road)
    assert cable_road.no_collisions == True


def test_check_if_support_withstands_tension(line_gdf: gpd.GeoDataFrame):
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 3)
    assert cable_road.count_segments() == 2

    # set to a high CR tension and low support force - but not to 80.000, since then we dont have any slack, which determines the force on the support
    cable_road.supported_segments[0].cable_road.s_current_tension = 100000
    cable_road.supported_segments[1].cable_road.s_current_tension = 100000
    cable_road.supported_segments[
        0
    ].end_support.max_supported_force_at_attachment_height = 1000

    support_withstands_tension = (
        mechanical_computations.check_if_support_withstands_tension(
            cable_road.supported_segments[0], cable_road.supported_segments[1]
        )
    )
    assert support_withstands_tension == False

    # set to a low CR tension and high support force
    cable_road.supported_segments[0].cable_road.s_current_tension = 50000
    cable_road.supported_segments[1].cable_road.s_current_tension = 50000
    cable_road.supported_segments[
        0
    ].end_support.max_supported_force_at_attachment_height = 100000

    support_withstands_tension = (
        mechanical_computations.check_if_support_withstands_tension(
            cable_road.supported_segments[0], cable_road.supported_segments[1]
        )
    )
    assert support_withstands_tension == True


# TODO - rework this for 3D
def test_xz_line_from_cr_startpoint_to_centroid(line_gdf: gpd.GeoDataFrame):
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 1)
    start_point = cable_road.xz_left_start_point

    index = len(cable_road.points_along_line) // 2

    line = mechanical_computations.get_xz_line_from_cr_startpoint_to_centroid(
        cable_road,
        start_point,
        move_centroid_left_from_start_point=True,
        sloped=True,
        index=index,
    )

    assert line.coords[0] == start_point.coords[0]
    # ensure that we moved the centroid to the bottom left of the start point, since we passed move_centroid_left=True
    assert line.coords[1][0] < start_point.coords[0][0]
    assert line.coords[1][1] < start_point.coords[0][1]

    # assert that the centroid is about 50% the length of the line long, adding 10% sag to the cable road and 20% tolerance
    assert np.isclose(
        (cable_road.c_rope_length / 2 + +(cable_road.c_rope_length * 0.1)),
        line.length,
        atol=0.2,
    )

    # and now move to the right
    line = mechanical_computations.get_xz_line_from_cr_startpoint_to_centroid(
        cable_road,
        start_point,
        move_centroid_left_from_start_point=False,
        sloped=True,
        index=index,
    )

    assert line.coords[0] == start_point.coords[0]
    # ensure that we moved the centroid to the bottom right of the start point, since we passed move_centroid_left=False
    assert line.coords[1][0] > start_point.coords[0][0]
    assert line.coords[1][1] < start_point.coords[0][1]

    # assert that the centroid is about 50% the length of the line long, adding 10% sag to the cable road and 20% tolerance
    assert np.isclose(
        (cable_road.c_rope_length / 2 + +(cable_road.c_rope_length * 0.1)),
        line.length,
        atol=0.2,
    )


def test_compute_resulting_force_on_cable():
    # go to the bottom right
    straight_line = LineString([(0, 0), (3, 3)])
    sloped_line = LineString([(0, 0), (4, 1.5)])

    resulting_force = mechanical_computations.compute_resulting_force_on_cable(
        straight_line, sloped_line, tension=50000, scaling_factor=10000
    )

    assert np.isclose(resulting_force, 20000, rtol=0.2)


def test_compute_tension_loaded_vs_unloaded_cableroad(
    cable_road: classes_cable_road_computation.Cable_Road,
):
    loaded_cr = cable_road.supported_segments[0].cable_road
    unloaded_cr = cable_road.supported_segments[1].cable_road

    loaded_cr.s_current_tension = 30000
    unloaded_cr.s_current_tension = 30000

    force = mechanical_computations.compute_tension_loaded_vs_unloaded_cableroad(
        loaded_cr, unloaded_cr, scaling_factor=10000
    )

    assert np.isclose(force, 7000, rtol=0.1)
