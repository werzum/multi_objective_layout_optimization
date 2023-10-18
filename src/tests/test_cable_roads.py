import geopandas as gpd
from itertools import pairwise

from src.main import (
    mechanical_computations,
    cable_road_computation,
    classes_cable_road_computation,
)

from src.tests import helper_functions


def main_test_cable_roads():
    line_gdf, tree_gdf, height_gdf = helper_functions.set_up_gdfs()
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 0)

    return cable_road, line_gdf, tree_gdf, height_gdf


def test_cable_road_creation(line_gdf, height_gdf):
    cable_road, line_gdf, tree_gdf, height_gdf = main_test_cable_roads()
    line = line_gdf.iloc[0]

    cable_road_computation.compute_required_supports(
        line["possible_anchor_triples"],
        line["max_holding_force"],
        line["tree_anchor_support_trees"],
        height_gdf,
        tree_gdf,
        from_line=line["line_candidates"],
    )


def test_unsupported_cable_road_parameters(
    line_gdf: gpd.GeoDataFrame, height_gdf: gpd.GeoDataFrame
):
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 1)

    assert min(cable_road.sloped_line_to_floor_distances) > 0
    # assert that this one is at least one meter higher - should always be the case
    assert (
        min(cable_road.unloaded_line_to_floor_distances)
        > min(cable_road.sloped_line_to_floor_distances) + 1
    )

    assert cable_road.c_rope_length > 0
    assert cable_road.b_length_whole_section > 0

    mechanical_computations.check_if_no_collisions_cable_road(cable_road)

    assert cable_road.no_collisions == False
    assert cable_road.anchors_hold == True


def test_supported_cable_road_parameters(
    line_gdf: gpd.GeoDataFrame, height_gdf: gpd.GeoDataFrame, line: gpd.GeoSeries
):
    cable_road = classes_cable_road_computation.load_cable_road(line_gdf, 0)

    for current_segment, next_segment in pairwise(cable_road.supported_segments):
        mechanical_computations.check_if_no_collisions_cable_road(
            current_segment.cable_road
        )
        assert cable_road.no_collisions == False
        assert cable_road.anchors_hold == True

        tower_and_anchors_hold = (
            mechanical_computations.check_if_tower_and_anchor_trees_hold(
                current_segment.cable_road,
                line["max_holding_force"],
                line["possible_anchor_triples"],
                height_gdf,
            )
        )
        assert tower_and_anchors_hold == True

        supports_hold = mechanical_computations.check_if_support_withstands_tension(
            current_segment, next_segment
        )
        assert supports_hold == True

        cable_road_computation.check_support_tension_and_collision(
            current_segment, next_segment
        )


def test_raise_height_and_check_tension(
    cable_road: classes_cable_road_computation.Cable_Road,
):
    segment_1 = cable_road.supported_segments[0]
    segment_2 = cable_road.supported_segments[1]

    cable_road_computation.raise_height_and_check_tension(segment_1, segment_2, 5)
    # assert that the height of the cable changes when we raise it
    segment_1_min_height = min(segment_1.cable_road.sloped_line_to_floor_distances)
    segment_2_min_height = min(segment_2.cable_road.sloped_line_to_floor_distances)

    cable_road_computation.raise_height_and_check_tension(segment_1, segment_2, 6)
    segment_1_min_height_raised = min(
        segment_1.cable_road.sloped_line_to_floor_distances
    )
    segment_2_min_height_raised = min(
        segment_2.cable_road.sloped_line_to_floor_distances
    )

    assert segment_1_min_height_raised > segment_1_min_height
    assert segment_2_min_height_raised > segment_2_min_height


def test_raise_tension_and_check_height(
    cable_road: classes_cable_road_computation.Cable_Road,
):
    assert cable_road.count_segments() == 2

    segment_1 = cable_road.supported_segments[0]
    segment_2 = cable_road.supported_segments[1]

    segment_1_min_height = min(segment_1.cable_road.sloped_line_to_floor_distances)
    segment_2_min_height = min(segment_2.cable_road.sloped_line_to_floor_distances)

    # test lowering the tension to see if the height lowers
    segment_1.cable_road.s_current_tension = 10000
    segment_2.cable_road.s_current_tension = 10000
    segment_1_min_height_low_tension = min(
        segment_1.cable_road.sloped_line_to_floor_distances
    )
    segment_2_min_height_low_tension = min(
        segment_2.cable_road.sloped_line_to_floor_distances
    )
    assert segment_1_min_height_low_tension < segment_1_min_height
    assert segment_2_min_height_low_tension < segment_2_min_height

    # and vice versa with high tension
    segment_1.cable_road.s_current_tension = 100000
    segment_2.cable_road.s_current_tension = 100000
    segment_1_min_height_high_tension = min(
        segment_1.cable_road.sloped_line_to_floor_distances
    )
    segment_2_min_height_high_tension = min(
        segment_2.cable_road.sloped_line_to_floor_distances
    )
    assert segment_1_min_height_high_tension > segment_1_min_height
    assert segment_2_min_height_high_tension > segment_2_min_height
