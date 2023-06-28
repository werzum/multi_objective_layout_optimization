import geopandas as gpd
import numpy as np
import pandas as pd

from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines

from itertools import chain
from shapely.geometry import Point
from vispy.scene import visuals
import vispy

import plotly.graph_objects as go

from src import (
    mechanical_computations,
    cable_road_computation,
    classes,
    geometry_operations,
    test_helper_functions,
)


def plot_gdfs(gdfs: list):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("X coordinate", fontsize=15)
    ax.set_ylabel("Y coordinate", fontsize=15)
    for gdf in gdfs:
        gdf.plot(alpha=0.5, ax=ax)


def plot_gdf_with_anchors_and_supports(gdfs: list, line_gdf: gpd.GeoDataFrame):
    """Plot a gdf with anchors and supports

    Args:
        gdfs (_type_): A list of the gdfs to plot
        line_gdf (_type_): the gdf containing the cable road lines
    """

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlabel("X coordinate", fontsize=15)
    ax.set_ylabel("Y coordinate", fontsize=15)
    for gdf in gdfs:
        gdf.plot(alpha=0.5, ax=ax)

    line_gdf.plot(alpha=0.5, ax=ax)

    for keyword in ["tree_anchor_support_trees"]:
        b = pd.concat(line_gdf[keyword].values)
        b.plot(cmap="tab20", ax=ax)

    for keyword in ["possible_anchor_triples"]:
        b = line_gdf[keyword]
        # c = line_gdf[]
        # double unpacking here
        c = list(chain.from_iterable(b))
        for sublist in c:
            d = gpd.GeoSeries([*sublist])
            d.plot(cmap="tab20", ax=ax)

    for keyword in ["location_of_int_supports"]:
        # filter the empty ones
        b = list(filter(None, line_gdf[keyword]))
        # unpack the list
        c = list(chain.from_iterable(b))
        d = gpd.GeoSeries(c)
        d.plot(cmap="tab20", ax=ax)


def plot_scatter_xy(list):
    xs = [point.x for point in list]
    ys = [point.y for point in list]
    plt.scatter(xs, ys)
    plt.show()


def plot_equal_axis(line):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("X coordinate", fontsize=15)
    ax.set_ylabel("Y coordinate", fontsize=15)
    ax.plot(*line.xy, label="LineString")
    ax.axis("equal")
    return ax


def onclick(event, coords):
    """Return an interactive figure to record mouse clicks on their coordinates
        Modifies a global coords variable to store the clicked points in
    Args:
        event (_type_): the event that is triggered by the mouse click
        coords (_type_): the global coords variable that is modified - its a list of tuples
    """
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords.append((ix, iy))

    if len(coords) == 6:
        print("Coordinates recorded are:", coords)
        coords = []
        print("Array reset")


def plot_p_median_results(
    model, facility_points_gdf, demand_points_gdf, anchor_trees, target_trees, line_gdf
):
    """Plot the results of the P-Median optimization. Based on this https://pysal.org/spopt/notebooks/p-median.html, but heavily shortened.

    Args:
        model (_type_): the P-Median model
        facility_points_gdf (_type_): A gdf with the facilities (ie. factories)
        demand_points_gdf (_type_): A gdf with the demand points (ie. trees)
    """
    arr_points = []
    fac_sites = []
    line_triples = []
    support_trees = []

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(facility_points_gdf)):
        if model.fac2cli[i]:
            # get the corresponding demand points from the fac2cli entry
            geom = demand_points_gdf.iloc[model.fac2cli[i]]["geometry"]
            arr_points.append(geom)
            fac_sites.append(i)
            # get the corresponding anchor triple and support tree from the line_gdf
            line_triples.append(line_gdf.iloc[i]["possible_anchor_triples"][0])
            support_trees.append(line_gdf.iloc[i]["possible_support_trees"])

    fig, ax = plt.subplots(figsize=(12, 12))
    legend_elements = []

    # ugly decomprehension
    unwrapped_triples = []
    for item in line_triples:
        unwrapped_triples.append(gpd.GeoSeries(item))

    # add the trees with respective color to which factory they belong to the map
    for i in range(len(arr_points)):
        gdf = gpd.GeoDataFrame(arr_points[i])
        anchor_lines_gdf = gpd.GeoDataFrame(geometry=unwrapped_triples[i])
        support_trees_gdf = support_trees[i]

        label = f"coverage_points by y{fac_sites[i]}"
        legend_elements.append(Patch(label=label))

        gdf.plot(ax=ax, zorder=3, alpha=0.7, label=label)
        facility_points_gdf.iloc[[fac_sites[i]]].plot(
            ax=ax, marker="*", markersize=200 * 3.0, alpha=0.8, zorder=4, edgecolor="k"
        )

        anchor_lines_gdf.plot(ax=ax, cmap="tab20")
        support_trees_gdf.plot(ax=ax, color="black")

        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                marker="*",
                ms=20 / 2,
                linewidth=0,
                alpha=0.8,
                label=f"y{fac_sites[i]} facility selected",
            )
        )

    anchor_trees.plot(ax=ax)
    target_trees.plot(ax=ax)

    plt.title("Optimized Layout", fontweight="bold")
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))


def plot_pymoo_results(
    model, facility_points_gdf, demand_points_gdf, anchor_trees, target_trees, line_gdf
):
    """Plot the results of the P-Median optimization. Based on this https://pysal.org/spopt/notebooks/p-median.html, but heavily shortened.

    Args:
        model (_type_): the P-Median model
        facility_points_gdf (_type_): A gdf with the facilities (ie. factories)
        demand_points_gdf (_type_): A gdf with the demand points (ie. trees)
    """
    arr_points = []
    fac_sites = []
    line_triples = []

    # extract the cli_assgn_vars and fac_vars from the model/x results of the pymoo optimization
    variable_matrix = model.reshape((len(demand_points_gdf) + 1, len(line_gdf)))

    cli_assgn_vars = variable_matrix[:-1]
    fac_vars = variable_matrix[-1:][0]

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(facility_points_gdf)):
        if fac_vars[i]:
            # get the corresponding demand points from the fac2cli entry
            # get all entries for this column in cli_assgn_vars
            indices = np.where(cli_assgn_vars[:, i])
            # geom = demand_points_gdf.iloc[model.fac2cli[i]]['geometry']
            geom = demand_points_gdf.iloc[indices]["geometry"]
            arr_points.append(geom)
            fac_sites.append(i)
            # get the corresponding anchor triple from the line_gdf
            line_triples.append(line_gdf.iloc[i]["possible_anchor_triples"])

    fig, ax = plt.subplots(figsize=(12, 12))
    legend_elements = []

    # ugly decomprehension
    unwrapped_triples = []
    for item in line_triples:
        unwrapped_triples.append(gpd.GeoSeries(sum(item, [])))

    # add the trees with respective color to which factory they belong to the map
    for i in range(len(arr_points)):
        # each factory belongs to each arr points - ie., we have 516 arr points which takes forever
        gdf = gpd.GeoDataFrame(arr_points[i])
        anchor_lines_gdf = gpd.GeoDataFrame(geometry=unwrapped_triples[i])

        label = f"coverage_points by y{fac_sites[i]}"
        legend_elements.append(Patch(label=label))

        gdf.plot(ax=ax, zorder=3, alpha=0.7, label=label)
        facility_points_gdf.iloc[[fac_sites[i]]].plot(
            ax=ax, marker="*", markersize=200 * 3.0, alpha=0.8, zorder=4, edgecolor="k"
        )

        anchor_lines_gdf.plot(ax=ax)

        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                marker="*",
                ms=20 / 2,
                linewidth=0,
                alpha=0.8,
                label=f"y{fac_sites[i]} facility selected",
            )
        )

    anchor_trees.plot(ax=ax)
    target_trees.plot(ax=ax)

    plt.title("P-Median", fontweight="bold")
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))


def plot_lines(
    this_cable_road: classes.Cable_Road,
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


def plot_vispy_scene(
    height_gdf: gpd.GeoDataFrame, view: vispy.scene.SceneCanvas, pos: list
):
    """Plot the vispy scene for a high performance visualization of the surface
    Args:
        height_gdf (gpd.GeoDataFrame): The gdf with the height data
        view (vispy.scene.SceneCanvas): The vispy scene
        pos (list): The list of positions to plot
    """

    height_gdf_small = height_gdf.iloc[::10, :]
    # pos of lines
    pos_lines = np.hstack((pos)).T
    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos_lines, edge_width=0, face_color=(1, 1, 0.5, 1), size=5)
    view.add(scatter)

    # pos of heightgdf
    pos_height_gdf = np.vstack(
        (height_gdf_small["x"], height_gdf_small["y"], height_gdf_small["elev"])
    ).T
    # create scatter object and fill in the data
    scatter = visuals.Markers()
    scatter.set_data(pos_height_gdf, edge_width=0, face_color=(1, 1, 1, 0.5), size=5)
    view.add(scatter)
    view.camera = "turntable"  # or try 'arcball'
    # add a colored 3D axis for orientation
    axis = visuals.XYZAxis(parent=view.scene)


def plot_cr_relief(
    sample_cable_road: classes.Cable_Road,
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

    if line_gdf.iloc[index]["number_of_supports"]:
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

    cable_road_segments = test_helper_functions.create_cable_road_segments(
        line_gdf, height_gdf, index
    )

    # for all individual road segments
    for cable_road in cable_road_segments:
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

    fig.update_traces(marker={"size": 0.75})
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        width=1000,
        height=800,
        title="Relief Map with possible Cable Roads",
    )

    if passed_fig is None:
        fig.show("notebook_connected")


def add_all_anchors_to_go_figure(
    sample_cable_road: classes.Cable_Road,
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

    for anchor in line_gdf.iloc[index].possible_anchor_triples[0]:
        anchor_point = Point(anchor.coords)
        anchor_line = LineString([anchor_point, sample_cable_road.start_point])
        anchor_cable_road = classes.Cable_Road(
            anchor_line, height_gdf, pre_tension=sample_cable_road.s_current_tension
        )

        x_anchor_cr = [point[0] for point in anchor_cable_road.floor_points]
        y_anchor_cr = [point[1] for point in anchor_cable_road.floor_points]
        z_line_to_floor = (
            anchor_cable_road.floor_height_below_line_points
            + anchor_cable_road.line_to_floor_distances
        )

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


def add_relief_to_go_figure(sample_cable_road: classes.Cable_Road, fig: go.Figure):
    """Add the relief of a single cable road to a figure.

    Args:
        sample_cable_road (classes.Cable_Road): _description_
        fig (go.Figure): _description_
    """
    # get the relief and plot it
    x_sample_cr = [point[0] for point in sample_cable_road.floor_points]
    y_sample_cr = [point[1] for point in sample_cable_road.floor_points]
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
    sample_cable_road: classes.Cable_Road, fig: go.Figure
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


def plot_parallelogram(
    ax: plt.Axes,
    s_max_point: Point,
    s_a_point_force: Point,
    a_3_point: Point,
    a_4_point: Point,
    a_5_point: Point,
    tower_xz_point: Point,
    tower_s_max_radius: Point,
    tower_s_a_radius: Point,
    tower_s_max_x_point: Point,
    s_max_length: float,
):
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-30, -15)
    ax.set_ylim(-15, 15)

    # plot the points
    ax.plot(*s_max_point.xy, "o", color="black")
    ax.plot(*tower_s_max_radius.xy, "o", color="green")
    ax.plot(*s_a_point_force.xy, "o", color="blue")
    ax.plot(*a_3_point.xy, "o", color="red")
    ax.plot(*a_4_point.xy, "o", color="red")
    ax.plot(*a_5_point.xy, "o", color="blue")
    ax.plot(*tower_xz_point.xy, "o", color="orange")

    for lines in [
        [s_max_point, tower_xz_point],
        [s_a_point_force, tower_xz_point],
        [a_3_point, s_max_point],
        [a_4_point, s_a_point_force],
        [a_5_point, tower_xz_point],
        [a_5_point, a_3_point],
        [a_5_point, a_4_point],
        [s_a_point_force, tower_s_max_x_point],
        [tower_s_max_x_point, s_max_point],
        [
            Point([a_3_point.coords[0][0], a_3_point.coords[0][1]]),
            Point(
                [
                    tower_xz_point.coords[0][0],
                    tower_xz_point.coords[0][1] - s_max_length,
                ]
            ),
        ],  # s3 to anchor with length of smax
        [
            Point([a_4_point.coords[0][0], a_4_point.coords[0][1]]),
            Point(
                [
                    tower_xz_point.coords[0][0],
                    a_4_point.coords[0][1]
                    - abs(s_a_point_force.coords[0][1] - tower_xz_point.coords[0][1]),
                ]
            ),
        ],
    ]:
        ax.plot(*LineString(lines).xy, color="black")

    # ax.annotate(
    #     "Force on Cable",
    #     s_max_point.coords[0],
    #     xytext=(3, -15),
    #     fontsize=14,
    #     textcoords="offset points",
    # )
    # ax.annotate(
    #     "Force on Anchor",
    #     s_a_point_force.coords[0],
    #     xytext=(3, -15),
    #     fontsize=14,
    #     textcoords="offset points",
    # )
    # ax.annotate(
    #     "Force on Tower",
    #     a_5_point.coords[0],
    #     xytext=(3, -15),
    #     fontsize=14,
    #     textcoords="offset points",
    # )
    # ax.annotate(
    #     "Buckling Force left",
    #     a_3_point.coords[0],
    #     xytext=(5, -5),
    #     fontsize=14,
    #     textcoords="offset points",
    # )
    # ax.annotate(
    #     "Buckling Force right",
    #     a_4_point.coords[0],
    #     xytext=(3, -15),
    #     fontsize=14,
    #     textcoords="offset points",
    # )
