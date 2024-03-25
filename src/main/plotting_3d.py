import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import LineString, Point
from spopt.locate import PMedian

from src.main import (
    mechanical_computations,
    classes_cable_road_computation,
    classes_geometry_objects,
)
from src.tests import helper_functions


def plot_lines(
    this_cable_road: classes_cable_road_computation.Cable_Road,
    pos: list,
):
    """Plot the lines of the cable road.
    Args:

        this_cable_road (classes.Cable_Road): The cable road to plot
        pos (list): The list of positions to plot
    """
    pos.append(
        (
            [point[0] for point in this_cable_road.floor_points],
            [point[1] for point in this_cable_road.floor_points],
            this_cable_road.floor_height_below_line_points
            + this_cable_road.sloped_line_to_floor_distances,
        )
    )


def plot_cr_relief(
    sample_cable_road: classes_cable_road_computation.Cable_Road,
    line_gdf: gpd.GeoDataFrame,
    height_gdf: gpd.GeoDataFrame,
    index: int,
    passed_fig: go.Figure = None,
    show_straight_line: bool = True,
):
    """Plot the relief of a single cable road with a scatterplot of relief, line and floor points. Refers the plot to the type of cable road (supported or unsupported)

    Args:
        sample_cable_road (classes.Cable_Road): _description_
        line_gdf (gpd.GeoDataFrame): _description_
        height_gdf (gpd.GeoDataFrame): _description_
        index (int): the index of the dataframe to extract more data
    """

    if sample_cable_road.supported_segments:
        plot_supported_cr_relief(
            sample_cable_road,
            line_gdf,
            height_gdf,
            index,
            passed_fig,
            show_straight_line=show_straight_line,
        )
    else:
        plot_unsupported_cr_relief(
            sample_cable_road,
            line_gdf,
            height_gdf,
            index,
            passed_fig,
            show_straight_line=show_straight_line,
        )


def plot_unsupported_cr_relief(
    sample_cable_road,
    line_gdf,
    height_gdf,
    index,
    passed_fig: go.Figure = None,
    show_straight_line=False,
):
    x_sample_cr = [point[0] for point in sample_cable_road.floor_points]
    y_sample_cr = [point[1] for point in sample_cable_road.floor_points]
    z_sloped = sample_cable_road.absolute_loaded_line_height

    fig = go.Figure() if passed_fig is None else passed_fig

    if show_straight_line:
        add_straight_line_to_go_figure(sample_cable_road, fig)

    add_relief_to_go_figure(sample_cable_road, fig)

    fig.add_trace(
        go.Scatter3d(
            x=x_sample_cr,
            y=y_sample_cr,
            z=z_sloped,
            mode="lines",
            line=dict(color="green", width=2),
            name="Sloped Distance",
        )
    )

    add_all_anchors_to_go_figure(sample_cable_road, line_gdf, height_gdf, index, fig)

    fig.update_layout(
        title="Detail View of Single Cable Road Path under Load", width=1200, height=800
    )
    # fig.write_html("02_Figures/Cable_Road_3d.html")
    if passed_fig is None:
        fig.update_layout(
            width=1000,
            height=800,
        )
        fig.show("notebook_connected")


def plot_supported_cr_relief(
    sample_cable_road,
    line_gdf,
    height_gdf,
    index,
    passed_fig: go.Figure = None,
    show_straight_line=False,
):
    fig = go.Figure() if passed_fig is None else passed_fig

    add_relief_to_go_figure(sample_cable_road, fig)
    add_all_anchors_to_go_figure(sample_cable_road, line_gdf, height_gdf, index, fig)

    # TODO - iterate over the segments
    # cable_road_segments = helper_functions.create_cable_road_segments(
    #     line_gdf, height_gdf, index
    # )

    # for all individual road segments
    for segment in sample_cable_road.supported_segments:
        cable_road = segment.cable_road
        if show_straight_line:
            add_straight_line_to_go_figure(cable_road, fig)

        x_sample_cr = [point[0] for point in cable_road.floor_points]
        y_sample_cr = [point[1] for point in cable_road.floor_points]
        z_sloped = (
            cable_road.floor_height_below_line_points
            + cable_road.sloped_line_to_floor_distances
        )

        fig.add_trace(
            go.Scatter3d(
                x=x_sample_cr,
                y=y_sample_cr,
                z=z_sloped,
                mode="lines",
                line=dict(width=3),
                name="Cable Road Segment",
            )
        )

    # fig.update_traces(marker={"size": 0.75})
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        width=1000,
        height=800,
        title="Relief Map with possible Cable Roads",
    )

    if passed_fig is None:
        fig.show("notebook_connected")


def add_all_anchors_to_go_figure(
    sample_cable_road: classes_cable_road_computation.Cable_Road,
    line_gdf: gpd.GeoDataFrame,
    height_gdf: gpd.GeoDataFrame,
    index: int,
    fig: go.Figure,
):
    """Add all anchors to the go figure.
    Args:
        sample_cable_road (classes.Cable_Road): _description_
        line_gdf (gpd.GeoDataFrame): _description_
        height_gdf (gpd.GeoDataFrame): _description_
        index (int): _description_
        fig (go.Figure): _description_
    """

    for anchor in line_gdf.loc[index].possible_anchor_triples[0]:
        anchor_point = Point(anchor.coords)
        anchor_line = LineString(
            [anchor_point, sample_cable_road.start_support.xy_location]
        )
        anchor_support = classes_cable_road_computation.Support(
            sample_cable_road.start_support.total_height,
            anchor_point,
            height_gdf,
            80000,
        )
        anchor_cable_road = classes_cable_road_computation.Cable_Road(
            anchor_line,
            height_gdf,
            sample_cable_road.start_support,
            anchor_support,
            pre_tension=sample_cable_road.s_current_tension,
        )

        x_anchor_cr, y_anchor_cr, z_line_to_floor = get_x_y_z_points(anchor_cable_road)

        fig.add_trace(
            go.Scatter3d(
                x=x_anchor_cr,
                y=y_anchor_cr,
                z=z_line_to_floor,
                mode="lines",
                line=dict(color="black", width=2),
                name="Anchor Cable",
            )
        )


def get_x_y_z_points(sample_cable_road):
    """Get the x, y and z points of a cable road - helper function for plotting
    Args:
        sample_cable_road (classes.Cable_Road): The CR
    Returns:
        tuple: list of x, y and z point coordinates
    """
    x_sample_cr = [point[0] for point in sample_cable_road.floor_points]
    y_sample_cr = [point[1] for point in sample_cable_road.floor_points]
    z__sloped = sample_cable_road.absolute_loaded_line_height

    return x_sample_cr, y_sample_cr, z__sloped


def add_relief_to_go_figure(
    sample_cable_road: classes_cable_road_computation.Cable_Road, fig: go.Figure
):
    """Add the relief of a single cable road to a figure.

    Args:
        sample_cable_road (classes.Cable_Road): _description_
        fig (go.Figure): _description_
    """
    # get the relief and plot it
    x_sample_cr, y_sample_cr, z_floor_height = get_x_y_z_points(sample_cable_road)
    z_floor_height = sample_cable_road.floor_height_below_line_points

    fig = fig.add_trace(
        go.Scatter3d(
            x=x_sample_cr,
            y=y_sample_cr,
            z=z_floor_height,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Relief",
        )
    )


def add_straight_line_to_go_figure(
    sample_cable_road: classes_cable_road_computation.Cable_Road, fig: go.Figure
):
    """Add the straight line distance to a figure.

    Args:
        sample_cable_road (classes.Cable_Road): _description_
        fig (go.Figure): _description_
    """
    # get the relief and plot it
    x_sample_cr = [point[0] for point in sample_cable_road.floor_points]
    y_sample_cr = [point[1] for point in sample_cable_road.floor_points]
    z_line_to_floor = (
        sample_cable_road.floor_height_below_line_points
        + sample_cable_road.unloaded_line_to_floor_distances
    )
    fig = fig.add_trace(
        go.Scatter3d(
            x=x_sample_cr,
            y=y_sample_cr,
            z=z_line_to_floor,
            mode="lines",
            line=dict(color="red", width=2),
            name="Straight Line Distance",
        )
    )


def plot_all_cable_roads(height_gdf, line_gdf) -> go.Figure:
    height_gdf_small = height_gdf.iloc[::100]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=height_gdf_small["x"],
                y=height_gdf_small["y"],
                z=height_gdf_small["elev"],
                color="lightgrey",
                opacity=0.5,
            )
        ]
    )
    # fig.update_traces(marker={"size": 0.75})
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        width=2000,
        height=1800,
        title="Relief Map with possible Cable Roads",
    )

    for index, row in line_gdf.iterrows():
        plot_cr_relief(
            row["Cable Road Object"],
            line_gdf,
            height_gdf,
            index,
            fig,
            show_straight_line=False,
        )

    return fig


import plotly.express as px


def plot_3d_model_results(model: PMedian, line_gdf, height_gdf) -> go.Figure:
    line_gdf, cable_road_objects = helper_functions.model_to_line_gdf(model, line_gdf)
    fig = plot_all_cable_roads(height_gdf, line_gdf)
    return fig


def plot_parallelogram(
    cable_road: classes_cable_road_computation.Cable_Road,
    max_holding_force: list[float],
    anchor_triplets: list,
    height_gdf: gpd.GeoDataFrame,
    fig: go.Figure = None,
):
    fig = go.Figure()
    mechanical_computations.check_if_tower_and_anchor_trees_hold(
        cable_road, max_holding_force, anchor_triplets, height_gdf, fig=fig
    )


def plot_tension_loaded_unloaded_cr(loaded_cr, unloaded_cr):
    fig = go.Figure()
    mechanical_computations.compute_tension_loaded_vs_unloaded_cableroad(
        loaded_cr, unloaded_cr, 10000, reverse_direction=False, fig=fig
    )
    mechanical_computations.compute_tension_loaded_vs_unloaded_cableroad(
        unloaded_cr, loaded_cr, 10000, reverse_direction=True, fig=fig
    )


def plot_Linestring_3D(
    line: classes_geometry_objects.LineString_3D, fig: go.Figure, label: str
) -> go.Figure:
    fig.add_trace(
        go.Scatter3d(
            x=[line.start_point.x, line.end_point.x],
            y=[line.start_point.y, line.end_point.y],
            z=[line.start_point.z, line.end_point.z],
            mode="lines",
            line=dict(color="green", width=1),
            name=label,
        )
    )

    return fig
