from src.main import (
    classes,
    geometry_utilities,
    mechanical_computations,
    cable_road_computation,
)

from src.tests import helper_functions


def main_test_cable_roads():
    line_gdf, tree_gdf, height_gdf = helper_functions.set_up_gdfs()
    cable_road = helper_functions.load_cable_road(line_gdf, height_gdf, 0)

    return cable_road, line_gdf, tree_gdf, height_gdf


def test_cable_road_creation(line_gdf, height_gdf):
    cable_road, line_gdf, tree_gdf, height_gdf = main_test_cable_roads()
    line = line_gdf.iloc[0]

    cable_road_computation.compute_required_supports(
        line["line_candidates"],
        line["possible_anchor_triples"],
        line["max_holding_force"],
        line["tree_anchor_support_trees"],
        height_gdf,
        0,
        tree_gdf,
        [],
    )


def test_unsupported_cable_road_parameters(line_gdf, height_gdf):
    cable_road = helper_functions.load_cable_road(line_gdf, height_gdf, 1)

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


def test_supported_cable_road_parameters(line_gdf, height_gdf):
    cable_road = helper_functions.load_cable_road(line_gdf, height_gdf, 1)

    for cable_road_segment in cable_road.support_segments:
        mechanical_computations.check_if_no_collisions_cable_road(cable_road_segment)
        assert cable_road.no_collisions == False
        assert cable_road.anchors_hold == True

        tower_and_anchors_hold = (
            mechanical_computations.check_if_tower_and_anchor_trees_hold(
                cable_road_segment, max_supported_forces, anchor_triplets, height_gdf
            )
        )
        assert tower_and_anchors_hold == True

        supports_hold = mechanical_computations.check_if_supports_hold(
            cable_road_segment, tree_anchor_support_trees, height_gdf
        )
        assert supports_hold == True

        cable_road_computation.check_support_tension_and_collision(
            cable_road_segment, height_gdf
        )
