import numpy as np

from src.main import classes_cable_road_computation, geometry_utilities


def pestal_load_path(
    cable_road: "classes_cable_road_computation.Cable_Road", loaded: bool = True
):
    """Calculates the load path of the cable road based on the pestal method

    Args:
        cable_road (classes.Cable_Road): the cable road
        point (Point): the point to calculate the load path for
    Returns:
        float: the deflection of the cable road along the load path
    """
    T_0_basic_tensile_force = cable_road.s_current_tension
    q_s_rope_weight = 1.6
    q_vertical_force = 15000 if loaded else 0

    h_height_difference = abs(
        cable_road.end_support.total_height - cable_road.start_support.total_height
    )

    T_bar_tensile_force_at_center_span = T_0_basic_tensile_force + q_s_rope_weight * (
        (h_height_difference / 2)
    )

    H_t_horizontal_force_tragseil = T_bar_tensile_force_at_center_span * (
        cable_road.b_length_whole_section / cable_road.c_rope_length
    )  # improvised value - need to do the parallelverchiebung here

    distances = geometry_utilities.distance_between_points_from_origin(
        cable_road.points_along_line_xy,
        np.array(
            cable_road.start_support.xy_location
        ).T,  # TODO remove this workaround, doesnt seem to be the updated support version where we have a direct numpy array
    )

    return (
        (distances * (cable_road.b_length_whole_section - distances))
        / (H_t_horizontal_force_tragseil * cable_road.b_length_whole_section)
    ) * (q_vertical_force + ((cable_road.c_rope_length * q_s_rope_weight) / 2))
