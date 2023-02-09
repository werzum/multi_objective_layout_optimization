import math

def calculate_length_unloaded_skyline(this_cable_road):
    # calculate basic length
    this_cable_road.z_mi_height_center_span = (
        this_cable_road.start_point_height + this_cable_road.end_point_height
    ) / 2  # need to adjust this so that first point is reference with z=0)
    this_cable_road.z_mi_height_support = this_cable_road.end_point_height

    # tension at support and center span, 4.6 and 4.7
    this_cable_road.t_i_bar_tensile_force_at_support = (
        this_cable_road.current_tension
        + this_cable_road.z_mi_height_support
        * this_cable_road.q_s_self_weight_center_span
    )

    this_cable_road.t_i_bar_tensile_force_center_span = (
        this_cable_road.current_tension
        + this_cable_road.z_mi_height_center_span
        * this_cable_road.q_s_self_weight_center_span
    )

    h_i_bar_horizontal_tensile_force = (
        this_cable_road.t_i_bar_tensile_force_center_span
        * (this_cable_road.b_length_whole_section / this_cable_road.c_rope_length)
    )

    # 4.10 overlength unloaded skyline
    this_cable_road.delta_s_overlength = (
        (this_cable_road.b_length_whole_section**4)
        * (this_cable_road.q_s_self_weight_center_span**2)
    ) / (24 * this_cable_road.c_rope_length * (h_i_bar_horizontal_tensile_force**2))

    # 4.12 total length of unloaded skyline
    this_cable_road.u_l_total_length = (
        this_cable_road.c_rope_length + this_cable_road.delta_s_overlength
    )


def calculate_length_loaded_skyline(this_cable_road):
    this_cable_road.t_v_j_bar_tensile_force_at_center_span = this_cable_road.tension + (
        this_cable_road.z_mi_height_center_span
        * this_cable_road.q_s_self_weight_center_span
    )
    # deflection as per 4.14
    this_cable_road.y_mi_deflection_at_center_span = (
        this_cable_road.c_rope_length
        / 4
        * this_cable_road.t_v_j_bar_tensile_force_at_center_span
    ) * (
        this_cable_road.q_load
        + (
            this_cable_road.c_rope_length
            * this_cable_road.q_s_self_weight_center_span
            / 2
        )
    )
    # overlength of chords 4.20
    c_delta_chord_length = (
        (2 * this_cable_road.b_length_whole_section**2)
        / (this_cable_road.c_rope_length**3)
    ) * this_cable_road.y_mi_deflection_at_center_span**2
    # and span of chords 4.23
    s_delta_span = (
        (this_cable_road.b_length_whole_section**2)
        * this_cable_road.c_rope_length
        * (this_cable_road.q_s_self_weight_center_span**2)
        / 96
        * this_cable_road.t_v_j_bar_tensile_force_at_center_span
    )

    # sum the different deltas together for overall length of loaded skyline 4.27
    this_cable_road.u_vj_length_loaded_skyline = (
        this_cable_road.c_rope_length + c_delta_chord_length + s_delta_span
    )


def horizontal_force_at_point(this_cable_road, point):
    # extract the x-coords from the point? TBD
    x = point.coords[0]
    # 4.37
    horizontal_force_at_x = (
        this_cable_road.h_mj_horizontal_force_under_load_at_center_span
        * math.sqrt(
            1
            - (
                1
                - (
                    this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support
                    / this_cable_road.h_mj_horizontal_force_under_load_at_center_span
                )
                ** 2
            )
            * (
                1
                - (
                    2
                    * (x / this_cable_road.this_cable_road.b_length_whole_section) ** 2
                )
            )
        )
    )

    return horizontal_force_at_x


def deflection_by_force_and_position(this_cable_road, point, force_at_point):
    x = point.coords[0]
    # 4.36 deflection
    y_x_deflection_at_x = (
        this_cable_road.y_mi_deflection_at_center_span
        * (
            this_cable_road.h_mj_horizontal_force_under_load_at_center_span
            / force_at_point
        )
        * (1 - (1 - (2 * x / this_cable_road.b_length_whole_section) ** 2))
    )

    return y_x_deflection_at_x


def calculate_deflections(this_cable_road):
    """Calculate array of deflections for each point in the skyline according to overlength

    Args:
        this_cable_road (_type_): _description_

    Returns:
        _type_: _description_
    """
    # horizontal components of force as per 4.34 and 4.35
    this_cable_road.h_mj_horizontal_force_under_load_at_center_span = (
        this_cable_road.b_length_whole_section / this_cable_road.c_rope_length
    ) * this_cable_road.t_v_j_bar_tensile_force_at_center_span

    this_cable_road.h_sj_h_mj_horizontal_force_under_load_at_support = (
        this_cable_road.b_length_whole_section / this_cable_road.c_rope_length
    ) * this_cable_road.t_i_bar_tensile_force_at_support

    # are we getting x right?
    horizontal_forces = [
        horizontal_force_at_point(this_cable_road, point)
        for point in this_cable_road.points_along_line
    ]
    # calculate the deflections as per force and position along the CR with 4.36
    y_x_deflections = [
        deflection_by_force_and_position(this_cable_road, point, force)
        for point, force in zip(this_cable_road.points_along_line, horizontal_forces)
    ]

    return y_x_deflections

def lastdurchhang_at_point(
    point, start_point, end_point, c_rope_length, b_whole_section
):
    """
    Calculates the lastdurchhang value at a given point.

    Args:
    point (Point): The point at which the lastdurchhang value is to be calculated.
    start_point (Point): The start point of the section.
    end_point (Point): The end point of the section.
    b_whole_section (float): The length of the whole section.
    H_t_horizontal_force_tragseil (float): The horizontal force of the tragseil.
    q_vertical_force (float): The vertical force.
    c_rope_length (float): The length of the rope.
    q_bar_rope_weight (float): The weight of the rope.
    q_delta_weight_difference_pull_rope_weight (float): The difference in weight between the pull rope and the tragseil.

    Returns:
    float: The lastdurchhang value at the given point.
    """
    H_t_horizontal_force_tragseil = 80000  # improvised value
    q_vertical_force = 15000  # improvised value 30kn?
    q_bar_rope_weight = 1.6  # improvised value 2?
    q_delta_weight_difference_pull_rope_weight = 0.6  # improvised value
    # compute distances and create the corresponding points

    b1_section_1 = start_point.distance(point)
    b2_section_2 = end_point.distance(point)

    lastdurchhang = (
        b1_section_1 * b2_section_2 / (b_whole_section * H_t_horizontal_force_tragseil)
    ) * (
        q_vertical_force
        + (c_rope_length * q_bar_rope_weight / 2)
        + (
            c_rope_length
            * q_delta_weight_difference_pull_rope_weight
            / (4 * b_whole_section)
        )
        * (b2_section_2 - b1_section_1)
    )
    return lastdurchhang