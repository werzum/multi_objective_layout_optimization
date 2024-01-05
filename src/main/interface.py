import pandas as pd
import numpy as np
from typing import Tuple
from random import random

import plotly.graph_objects as go
import plotly.express as px

from src.main import geometry_operations


def create_trees_and_lines_traces(forest_area_gdf, transparent_line):
    # create a trace for the trees
    xs, ys = zip(
        *[
            (row.xy[0][0], row.xy[1][0])
            for row in forest_area_gdf.harvesteable_trees_gdf.geometry
        ]
    )
    trees = go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(color="green"),
        name="Trees",
        hoverinfo="none",
    )

    # Create traces for each line
    individual_lines = [
        go.Scatter(
            x=np.asarray(row.geometry.xy[0]) + random(),
            y=np.asarray(row.geometry.xy[1]) + random(),
            mode="lines",
            line=transparent_line,
            name=str(id),
        )
        for id, row in forest_area_gdf.line_gdf.iterrows()
    ]

    return trees, individual_lines


def plot_pareto_frontier(
    results_df,
    current_indices,
    interactive_layout,
    current_layout_overview_figure,
    current_cable_roads_table_figure,
    current_cable_roads_table,
    forest_area_gdf,
    transparent_line,
    solid_line,
    model_list,
):
    pareto_frontier = go.FigureWidget(
        go.Scatter3d(
            x=results_df["ecological_distances_RNI"],
            y=results_df["ergonomics_distances_RNI"],
            z=results_df["cost_objective_RNI"],
            mode="markers",
        )
    )

    pareto_frontier.update_layout(
        title="""Pareto Frontier""",
        width=600,
        height=600,
        scene=dict(
            xaxis_title="Ecological Optimality",
            yaxis_title="Ergonomics Optimality",
            zaxis_title="Cost Optimality",
        ),
        scene_camera_eye=dict(x=1.7, y=1.7, z=1),
    )
    pareto_frontier.add_annotation(
        dict(
            text="""A Pareto Frontier represents the set of all non-dominated solutions, <br> i.e. solutions where none of the objective functions can be improved in value without degrading some of the other objective values.<br>
        Here, we consider three objectives: cost, relative ergonomical impact, and relative ecological impact. <br>Each point on the Pareto Frontier represents a unique combination of these three objectives that is Pareto optimal.<br> 
        No point on the Pareto Frontier can be improved in one objective without worsening at least one of the other objectives1.""",
            y=-1,
        )
    )

    def selection_fn(trace, points, selector):
        print("recevied click")
        # get index of this point in the trace
        index = points.point_inds[0]

        # first set all traces to lightgrey, ie deactivated:
        interactive_layout.update_traces(line=transparent_line)

        # get the corresponding list of activated cable rows from the dataframe
        active_rows = results_df.iloc[index]["selected_lines"]

        for active_row in active_rows:
            # then, set the traces at the indices of the selected pareto option to black
            interactive_layout.update_traces(
                line=solid_line, selector={"name": str(active_row)}
            )

        # and update the list of current indices to the new ones
        current_indices = active_rows
        update_tables(
            current_cable_roads_table_figure,
            current_cable_roads_table,
            current_layout_overview_figure,
            current_indices,
            interactive_layout,
            forest_area_gdf,
            model_list,
        )
        # as well as the line colors
        update_line_colors_by_indices(current_indices, interactive_layout)

    pareto_frontier.data[0].on_click(selection_fn)
    return pareto_frontier


def update_tables(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    current_layout_overview_figure,
    current_indices,
    interactive_layout,
    forest_area_gdf,
    model_list,
):
    """
    Function to update both tables with the new current_indices based on the selected lines
    """

    # and update the dataframe showing the computed costs
    updated_layout_costs = update_layout_overview(
        current_indices, forest_area_gdf, model_list
    )

    current_layout_overview_figure.data[0].cells.values = (
        updated_layout_costs["Tree_to_line_distance"],
        updated_layout_costs["productivity_cost_overall"],
        updated_layout_costs["line_cost"],
    )

    # set the current_cable_roads_table dataframe rows to show only these CRs
    line_costs, line_lengths, dummy_variable = current_cable_roads_table.loc[
        current_cable_roads_table.index.isin(current_indices)
    ].values.T
    current_cable_roads_table_figure.data[0].cells.values = [
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["wood_volume_per_cr"],
    ]

    # as well as the colour of the corresponding trees
    interactive_layout.data[0].marker.color = [
        px.colors.qualitative.Plotly[integer]
        for integer in updated_layout_costs["tree_to_line_assignment"]
    ]


def create_contour_traces(forest_area_3):
    """Create the contour traces for the given forest area at the given resolution"""
    # only select ~200 points, as everything else is too much and crashes
    small_contour_height_gdf = forest_area_3.height_gdf.iloc[::10000]
    # and only get points in a certain range to preserve our frame of reference
    small_contour_height_gdf = small_contour_height_gdf[
        (small_contour_height_gdf.x.between(-130, 20))
        & (small_contour_height_gdf.y.between(-20, 165))
    ]

    # create the traces
    data = go.Contour(
        z=small_contour_height_gdf.elev.values,
        x=small_contour_height_gdf.x.values,
        y=small_contour_height_gdf.y.values,
        opacity=0.3,
        showscale=False,
        hoverinfo="none",
        colorscale="Greys",
    )
    return data


def update_layout_overview(indices, forest_area_gdf, model_list) -> dict:
    """
    Function to update the cost dataframe with the updated costs for the new configuration based on the selected lines
    Returns distance_trees_to_lines, productivity_cost_overall, line_cost, total_cost, tree_to_line_assignment
    """

    rot_line_gdf = forest_area_gdf.line_gdf[
        forest_area_gdf.line_gdf.index.isin(indices)
    ]

    # Create a matrix with the distance between every tree and line and the distance between the support (beginning of the CR) and the carriage (cloests point on the CR to the tree)
    (
        distance_tree_line,
        distance_carriage_support,
    ) = geometry_operations.compute_distances_facilities_clients(
        forest_area_gdf.harvesteable_trees_gdf, rot_line_gdf
    )

    # assign all trees to their closest line
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)

        # compute the distance of each tree to its assigned line
        distance_trees_to_lines = sum(
            distance_tree_line[
                range(len(tree_to_line_assignment)), tree_to_line_assignment
            ]
        )
    except:
        tree_to_line_assignment = [1 for i in range(len(distance_tree_line))]
        distance_trees_to_lines = sum(distance_tree_line)

    # compute the productivity cost
    productivity_cost_overall = np.sum(
        model_list[0].productivity_cost[
            range(len(tree_to_line_assignment)), tree_to_line_assignment
        ]
    )

    # sum of wood volume per CR
    # here we need to compute the sum of wood per tree to line assignment to return this for the CR table
    # first we get for each class the indices
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(len(rot_line_gdf))
    ]
    wood_volume_per_cr = [
        int(
            sum(
                forest_area_gdf.harvesteable_trees_gdf.iloc[grouped_indices][
                    "cubic_volume"
                ]
            )
        )
        for grouped_indices in grouped_class_indices
    ]

    line_cost = sum(rot_line_gdf["line_cost"])

    # return a dict of the results and convert all results to ints for readability
    return {
        "Tree_to_line_distance": int(distance_trees_to_lines),
        "productivity_cost_overall": int(productivity_cost_overall),
        "line_cost": int(line_cost),
        "tree_to_line_assignment": tree_to_line_assignment,
        "wood_volume_per_cr": wood_volume_per_cr,
    }


def update_line_colors_by_indices(current_indices, interactive_layout):
    """
    Function to set the line colors of the interactive layout based on the current indices
    """
    for indice, integer in zip(current_indices, range(len(current_indices))):
        interactive_layout.data[indice + 2].line.color = px.colors.qualitative.Plotly[
            integer
        ]


def interactive_cr_selection(
    forest_area_gdf, model_list, optimization_result_list, results_df
) -> Tuple[go.FigureWidget, go.FigureWidget, go.FigureWidget, go.FigureWidget]:
    """
    Create an interactive cable road layout visualization.

    Parameters:
    - forest_area_gdf: GeoDataFrame, input forest area data
    - model_list: List, list of models
    - optimization_result_list: List, list of optimization results
    - results_df: DataFrame, optimization results DataFrame

    Returns:
    - Tuple containing four FigureWidgets: interactive_layout, current_cable_roads_table_figure,
      layout_overview_table_figure, pareto_frontier
    """
    # initialize the current indices list we use to keep track of the selected lines
    current_indices = []

    # define the transparent color for CRs once
    color_transparent = "rgba(0,0,0,0.4)"
    transparent_line = dict(color=color_transparent, width=0.5)
    solid_line = dict(color="black", width=5)

    # create traces for the lines and trees
    trees, individual_lines = create_trees_and_lines_traces(
        forest_area_gdf, transparent_line
    )

    # create the traces for a contour plot
    contour_traces = create_contour_traces(forest_area_gdf)

    # create a figure from all individual scatter lines
    interactive_layout = go.FigureWidget([trees, contour_traces, *individual_lines])
    interactive_layout.update_layout(
        title="Interactive Cable Road Layout",
        width=1200,
        height=800,
    )

    # create a dataframe and push it to a figurewidget to display details about our selected lines
    current_cable_roads_table = forest_area_gdf.line_gdf[["line_cost", "line_length"]]
    current_cable_roads_table["current_wood_volume"] = pd.Series(dtype="int")
    current_cable_roads_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(values=["Line Cost", "Line Length", "Wood Volume per CR"]),
                cells=dict(values=[]),
            )
        ]
    )
    current_cable_roads_table_figure.update_layout(title="Current Cable Roads Overview")

    # and for the current layout overview
    layout_overview_df = pd.DataFrame(
        columns=[
            "Tree_to_line_distance",
            "Productivity_cost",
            "Line_cost",
        ]
    )
    layout_overview_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=["Tree to Line Distance", "Productivity Cost", "Line Cost"]
                ),
                cells=dict(values=[layout_overview_df]),
            )
        ]
    )
    layout_overview_table_figure.update_layout(title="Current Layout Overview")

    # get the pareto frontier as 3d scatter plot
    pareto_frontier = plot_pareto_frontier(
        results_df,
        current_indices,
        interactive_layout,
        layout_overview_table_figure,
        current_cable_roads_table_figure,
        current_cable_roads_table,
        forest_area_gdf,
        transparent_line,
        solid_line,
        model_list,
    )

    # create the onclick function to select new CRs
    def selection_fn(trace, points, selector):
        print("recevied click")
        # since the handler is activated for all lines, test if this one has coordinates, ie. is the clicked line
        if points.xs:
            # turn all selected lines to lightgrey
            if trace.line.color != color_transparent:
                interactive_layout.update_traces(
                    line=transparent_line,
                    selector={"name": trace.name},
                )

            # and if this is the clicked line and its not yet activated, turn it black
            elif trace.line.color == color_transparent:
                # update this trace to turn black
                interactive_layout.update_traces(
                    line=solid_line,
                    selector={"name": trace.name},
                )

            # get all active traces  - ie those which are not lightgrey
            active_traces = interactive_layout.select_traces(
                selector=lambda x: True if x.line.color != color_transparent else False
            )

            current_indices = [int(trace.name) for trace in active_traces]

            # color the traces
            # we set the color of the lines in the current indices in consecutive order by choosing corresponding colors from the colorway
            update_line_colors_by_indices(current_indices, interactive_layout)

            # update the tables accordingly

            update_tables(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                current_indices,
                interactive_layout,
                forest_area_gdf,
                model_list,
            )

    # add the onclick function to all line traces
    for trace in interactive_layout.data[2:]:
        trace.on_click(selection_fn)

    return (
        interactive_layout,
        current_cable_roads_table_figure,
        layout_overview_table_figure,
        pareto_frontier,
    )
