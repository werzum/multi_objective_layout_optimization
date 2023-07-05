from shapely.geometry import Point, LineString
import geopandas as gpd
from itertools import pairwise

from src.main import (
    classes,
    geometry_utilities,
    mechanical_computations,
    cable_road_computation,
)

from src.tests import helper_functions


def main_test_cable_roads():
    line_gdf, tree_gdf, height_gdf = helper_functions.set_up_gdfs()
    cable_road = classes.load_cable_road(line_gdf, 0)

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
    cable_road = classes.load_cable_road(line_gdf, 1)

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
    cable_road = classes.load_cable_road(line_gdf, 0)

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
