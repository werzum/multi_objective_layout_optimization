import geopandas as gpd
import numpy as np
import pandas as pd

from spopt.locate import PMedian

from shapely.geometry import Point, LineString

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines

from itertools import chain
from vispy.scene import visuals
import vispy

import plotly.graph_objects as go

from src.main import (
    mechanical_computations,
    classes,
)

from src.tests import helper_functions


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
    tree_anchor = []
    intermediate_supports = []
    line_geometries = []

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(facility_points_gdf)):
        if model.fac2cli[i]:
            line = line_gdf.iloc[i]

            line_geometries.append(line)
            # get the corresponding demand points from the fac2cli entry
            geom = demand_points_gdf.iloc[model.fac2cli[i]]["geometry"]
            arr_points.append(geom)
            fac_sites.append(i)

            # get the corresponding anchor triple and support tree from the line_gdf
            line_triples.append(line["possible_anchor_triples"][0])
            tree_anchor.append(line["tree_anchor_support_trees"])

            # get the intermediate supports
            sub_segments = list(line["Cable Road Object"].get_all_subsegments())
            intermediate_supports.append(
                [subsegment.end_support.xy_location for subsegment in sub_segments][:-1]
            )

    unwrapped_triples = [gpd.GeoSeries(item) for item in line_triples]

    # add the trees with respective color to which factory they belong to the map
    fig = go.Figure()
    colours = ["red", "blue", "green", "orange", "purple", "yellow"]
    for i in range(len(arr_points)):
        # the main CR
        current_line = line_geometries[i]
        fig = add_geometries_to_fig(current_line.geometry, fig, marker_color="black")

        # the trees
        temp_points = arr_points[i]
        xs = [point.x for point in temp_points]
        ys = [point.y for point in temp_points]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker={"color": colours[i]},
                name=f"Trees covered by line {i}",
            )
        )

        # the anchors
        xs, ys = unwrapped_triples[i][0].xy
        for triple in unwrapped_triples[i]:
            fig = add_geometries_to_fig(triple.coords, fig, marker_color="black")

        # the supports
        xs = [point.x for point in intermediate_supports[i]]
        ys = [point.y for point in intermediate_supports[i]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                marker={"color": "black"},
            )
        )

    fig.update_layout(title="P-Median", width=1200, height=800)
    fig.show("notebook_connected")


def add_geometries_to_fig(
    object, fig: go.Figure, marker_color: str, name: str = ""
) -> go.Figure:
    xs, ys = object.xy
    xs, ys = list(xs), list(ys)
    fig.add_trace(
        go.Scatter(
            x=list(xs),
            y=list(ys),
            marker={"color": marker_color},
        )
    )
    return fig


from plotly.subplots import make_subplots


def extract_moo_model_results(
    optimized_model: PMedian,
    line_gdf: gpd.GeoDataFrame,
    fig: go.Figure = None,
    print_results: bool = False,
) -> go.Figure:
    """Extract the results of the P-Median optimization and return them as a plotly table

    Args:
        optimized_model (PMedian): The optimized model
        line_gdf (gpd.GeoDataFrame): The line gdf
        fig (go.Figure): The figure to add the table to
        print_results (bool): Whether to print the results to the console
    Returns:
        go.Figure: The figure with the table
    """

    selected_lines, cable_road_objects = helper_functions.model_to_line_gdf(
        optimized_model, line_gdf
    )

    selected_lines["number_int_supports"] = [
        len(list(cable_road.get_all_subsegments())) - 1
        if list(cable_road.get_all_subsegments())
        else 0
        for cable_road in cable_road_objects
    ]

    columns_to_select = [
        "slope_deviation",
        "angle_between_supports",
        "line_length",
        "line_cost",
        "number_int_supports",
    ]

    table = go.Table(
        header=dict(values=columns_to_select, fill_color="paleturquoise", align="left"),
        cells=dict(
            values=[*[selected_lines[val].astype(int) for val in columns_to_select]],
            fill_color="lavender",
            align="left",
        ),
    )

    total_cost = [int(selected_lines["line_cost"].sum())]
    summary_table = go.Table(
        header=dict(values=["Total cost"], fill_color="lightblue", align="left"),
        cells=dict(
            values=[total_cost],
            fill_color="lavender",
            align="left",
        ),
    )

    if fig:
        fig.add_trace(table)
        fig.add_trace(summary_table)
    elif print_results:
        print(selected_lines[columns_to_select])
        print(total_cost)
    else:
        fig = make_subplots(
            rows=2, cols=1, specs=[[{"type": "table"}], [{"type": "table"}]]
        )
        fig.add_trace(table, row=1, col=1)
        fig.add_trace(summary_table, row=2, col=1)

    return fig


def model_results_comparison(
    model_list: list,
    productivity_cost_matrix: np.ndarray,
    aij: np.ndarray,
    distance_carriage_support: np.ndarray,
):
    """Compare the results of the different models in one table
    Args:
        model_list (list): a list of the models with different tradeoffs
        productivity_cost_matrix (np.ndarray): the productivity cost matrix
        aij (np.ndarray): the distance matrix
        distance_carriage_support (np.ndarray): the distance matrix
    """
    productivity_array = []
    aij_array = []
    distance_carriage_support_array = []

    for model in model_list:
        # get the lines which are active
        fac_vars = np.array([bool(var.value()) for var in model.fac_vars])
        # and the corresponding rows from the distance matrix
        row_sums = []
        for index, row in enumerate(model.fac2cli):
            if row:
                distance_per_this_row = aij[row, index]
                row_sum_distance = distance_per_this_row.sum()
                row_sums.append(row_sum_distance)
        aij_array.append(sum(row_sums))

        row_sums = []
        for index, row in enumerate(model.fac2cli):
            if row:
                productivity_per_row = productivity_cost_matrix[row, index]
                row_sum_distance = productivity_per_row.sum()
                row_sums.append(row_sum_distance)
        productivity_array.append(sum(row_sums))

        row_sums = []
        for index, row in enumerate(model.fac2cli):
            if row:
                distance_per_this_row = distance_carriage_support[row, index]
                row_sum_distance = distance_per_this_row.sum()
                row_sums.append(row_sum_distance)
        distance_carriage_support_array.append(sum(row_sums))

    return pd.DataFrame(
        data={
            "Total distance of trees to cable roads": aij_array,
            "Productivity cost per m3 as per Stampfer": productivity_array,
            "Total distance from carriage to support": distance_carriage_support_array,
        }
    )


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

    for anchor in line_gdf.loc[index].possible_anchor_triples[0]:
        anchor_point = Point(anchor.coords)
        anchor_line = LineString(
            [anchor_point, sample_cable_road.start_support.xy_location]
        )
        anchor_support = classes.Support(
            sample_cable_road.start_support.total_height,
            anchor_point,
            height_gdf,
            80000,
        )
        anchor_cable_road = classes.Cable_Road(
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


def add_relief_to_go_figure(sample_cable_road: classes.Cable_Road, fig: go.Figure):
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


def plot_all_cable_roads(height_gdf, line_gdf) -> go.Figure:
    height_gdf_small = height_gdf.iloc[::100]
    fig = px.scatter_3d(
        x=height_gdf_small["x"], y=height_gdf_small["y"], z=height_gdf_small["elev"]
    )
    fig.update_traces(marker={"size": 0.75})
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        width=1000,
        height=800,
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
    cable_road: classes.Cable_Road,
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
    line: classes.LineString_3D, fig: go.Figure, label: str
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


def plot_moo_objective_function(model_list: list, steps: int):
    """Plot the objective function of the pymoo optimization
    Args:
        model_list (list): a list of the models with different tradeoffs
        steps (int): the number of steps to plot
    """
    # visualize optimization outcome, using matplotlib.pyplot
    plt.figure(figsize=(15, 8))

    obj_1_list = [i for i in range(steps)]

    # Value of objective function
    obj_2_list = [model_list[i].problem.objective.value() for i in range(steps)]

    obj_difference = [obj_2_list[i + 1] - obj_2_list[i] for i in range(steps - 1)]
    obj_difference.append(0)

    plt.plot(obj_1_list, obj_2_list, color="red")

    plt.xlabel("Objective Tradeoff", size=20)
    plt.ylabel("Objective Function Value", size=20)
    # -- add plot title
    plt.title("Combined Objective Function Value", size=32)
    # -- show plot
    plt.show()
