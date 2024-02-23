from calendar import c
from functools import partial
import re
from turtle import up
from matplotlib.axis import YAxis
import pandas as pd
import numpy as np
from typing import Tuple
from random import random

import plotly.graph_objects as go
import plotly.express as px
from ipywidgets.widgets import Button, Dropdown, Textarea, Layout
from torch import prod

from src.main import geometry_operations, plotting_3d


def create_trees_and_lines_traces(forest_area_3, transparent_line):
    # create a trace for the trees
    xs, ys = zip(
        *[
            (row.xy[0][0], row.xy[1][0])
            for row in forest_area_3.harvesteable_trees_gdf.geometry
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
        for id, row in forest_area_3.line_gdf.iterrows()
    ]

    return trees, individual_lines


def update_interactive_based_on_indices(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
    transparent_line,
    solid_line,
):
    """
    Function to update the interactive layout based on the current indices as well as the corresponding tables
    """

    # first set all traces to lightgrey, ie deactivated:
    interactive_layout.update_traces(line=transparent_line)

    for active_row in current_indices:
        # then, set the traces at the indices of the selected pareto option to black
        interactive_layout.update_traces(
            line=solid_line, selector={"name": str(active_row)}
        )

    # and update the list of current indices to the new ones as well as the layout
    update_colors_and_tables(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        layout_overview_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
    )


def update_colors_and_tables(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
):
    """
    Wrapper function to update both the colors of the lines and the tables
    """
    update_tables(
        current_cable_roads_table_figure,
        current_cable_roads_table,
        layout_overview_table_figure,
        current_indices,
        interactive_layout,
        forest_area_3,
        model_list,
    )
    update_line_colors_by_indices(current_indices, interactive_layout)


def update_tables(
    current_cable_roads_table_figure,
    current_cable_roads_table,
    layout_overview_table_figure,
    current_indices,
    interactive_layout,
    forest_area_3,
    model_list,
):
    """
    Function to update both tables with the new current_indices based on the selected lines
    """

    # and update the dataframe showing the computed costs
    updated_layout_costs = update_layout_overview(
        current_indices, forest_area_3, model_list
    )

    layout_overview_table_figure.data[0].cells.values = [
        updated_layout_costs["Total Cable Road Costs"],
        updated_layout_costs["Ecolgical Penalty"],
        updated_layout_costs["Ergonomic Penalty"],
        [current_indices],
        updated_layout_costs["Max Yarding Distance"],
        updated_layout_costs["Average Yarding Distance"],
    ]

    # set the current_cable_roads_table dataframe rows to show only these CRs
    line_costs, line_lengths, dummy_variable = current_cable_roads_table.loc[
        current_cable_roads_table.index.isin(current_indices)
    ].values.T

    current_cable_roads_table_figure.data[0].cells.values = [
        line_costs.astype(int),
        line_lengths.astype(int),
        updated_layout_costs["Wood Volume per Cable Road"],
        updated_layout_costs["Supports Amount"],
        updated_layout_costs["Supports Height"],
    ]

    # as well as the colour of the corresponding trees
    interactive_layout.data[0].marker.color = [
        px.colors.qualitative.Plotly[integer]
        for integer in updated_layout_costs["Tree to Cable Road Assignment"]
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
        name="Contour",
    )
    return data


def update_layout_overview(indices, forest_area_3, model_list) -> dict:
    """
    Function to update the cost dataframe with the updated costs for the new configuration based on the selected lines
    Returns distance_trees_to_lines, productivity_cost_overall, line_cost, total_cost, tree_to_line_assignment
    """

    rot_line_gdf = forest_area_3.line_gdf[forest_area_3.line_gdf.index.isin(indices)]

    # Create a matrix with the distance between every tree and line and the distance between the support (beginning of the CR) and the carriage (cloests point on the CR to the tree)
    (
        distance_tree_line,
        distance_carriage_support,
    ) = geometry_operations.compute_distances_facilities_clients(
        forest_area_3.harvesteable_trees_gdf, rot_line_gdf
    )

    # assign all trees to their closest line
    try:
        tree_to_line_assignment = np.argmin(distance_tree_line, axis=1)

        # compute the distance of each tree to its assigned line
        distance_trees_to_selected_lines = distance_tree_line[
            range(len(tree_to_line_assignment)), tree_to_line_assignment
        ]
        distance_trees_to_lines_sum = sum(distance_trees_to_selected_lines)
    except:
        tree_to_line_assignment = [1 for i in range(len(distance_tree_line))]
        distance_trees_to_lines_sum = sum(distance_tree_line)

    # compute the productivity cost
    productivity_cost_overall = np.sum(
        model_list[0].productivity_cost[
            range(len(tree_to_line_assignment)), tree_to_line_assignment
        ]
    )

    # sum of wood volume per CR
    # here we need to compute the sum of wood per tree to line assignment to return this for the CR table
    grouped_class_indices = [
        np.nonzero(tree_to_line_assignment == label)[0]
        for label in range(len(rot_line_gdf))
    ]
    wood_volume_per_cr = [
        int(
            sum(
                forest_area_3.harvesteable_trees_gdf.iloc[grouped_indices][
                    "cubic_volume"
                ]
            )
        )
        for grouped_indices in grouped_class_indices
    ]

    # get the height and amount of supports
    supports_height = [
        (
            [
                segment.start_support.attachment_height
                for segment in cr_object.supported_segments[1:]
            ]
            if cr_object.supported_segments
            else []
        )
        for cr_object in rot_line_gdf["Cable Road Object"]
    ]
    supports_amount = [len(heights) for heights in supports_height]

    # get the max and average yarding distance
    max_yarding_distance = max(distance_trees_to_selected_lines)
    average_yarding_distance = np.mean(distance_trees_to_selected_lines)

    line_cost = sum(rot_line_gdf["line_cost"])

    # total cost = # get the total cable road costs
    total_cable_road_costs = line_cost + productivity_cost_overall

    # calulate the environmental impact of each line beyond 10m lateral distance
    ecological_penalty_threshold = 10
    ecological_penalty_lateral_distances = np.where(
        distance_tree_line > ecological_penalty_threshold,
        distance_tree_line - ecological_penalty_threshold,
        0,
    )

    sum_eco_distances = sum(
        [
            ecological_penalty_lateral_distances[j][i]
            for i, j in zip(
                tree_to_line_assignment,
                range(len(ecological_penalty_lateral_distances)),
            )
        ]
    )

    # double all distances greater than penalty_treshold
    ergonomics_penalty_treshold = 15
    ergonomic_penalty_lateral_distances = np.where(
        distance_tree_line > ergonomics_penalty_treshold,
        (distance_tree_line - ergonomics_penalty_treshold) * 2,
        0,
    )
    sum_ergo_distances = sum(
        [
            ergonomic_penalty_lateral_distances[j][i]
            for i, j in zip(
                tree_to_line_assignment,
                range(len(ergonomic_penalty_lateral_distances)),
            )
        ]
    )

    # return a dict of the results and convert all results to ints for readability
    return {
        "Wood Volume per Cable Road": wood_volume_per_cr,
        "Total Cable Road Costs": int(total_cable_road_costs),
        "Ecolgical Penalty": int(sum_eco_distances),
        "Ergonomic Penalty": int(sum_ergo_distances),
        "Tree to Cable Road Assignment": tree_to_line_assignment,
        "Supports Height": supports_height,
        "Supports Amount": supports_amount,
        "Max Yarding Distance": int(max_yarding_distance),
        "Average Yarding Distance": int(average_yarding_distance),
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
    forest_area_3, model_list, optimization_result_list, results_df
):
    """
    Create an interactive cable road layout visualization.

    Parameters:
    - forest_area_3: GeoDataFrame, input forest area data
    - model_list: List, list of models
    - optimization_result_list: List, list of optimization results
    - results_df: DataFrame, optimization results DataFrame

    Returns:
    - Tuple containing four FigureWidgets: interactive_layout, current_cable_roads_table_figure,
      layout_overview_table_figure, pareto_frontier
    """
    # initialize the current indices list we use to keep track of the selected lines
    current_indices = []

    # initialize the selected cr to none
    selected_cr = 0

    # define the transparent color for CRs once
    color_transparent = "rgba(0, 0, 0, 0.4)"
    transparent_line = dict(color=color_transparent, width=0.5)
    solid_line = dict(color="black", width=5)

    # create traces for the lines and trees
    trees, individual_lines = create_trees_and_lines_traces(
        forest_area_3, transparent_line
    )

    # create the traces for a contour plot
    contour_traces = create_contour_traces(forest_area_3)

    # create a figure from all individual scatter lines
    interactive_layout = go.FigureWidget([trees, contour_traces, *individual_lines])
    interactive_layout.update_layout(
        title="Interactive Cable Road Layout",
        width=1200,
        height=900,
    )

    # create a dataframe and push it to a figurewidget to display details about our selected lines
    current_cable_roads_table = forest_area_3.line_gdf[["line_cost", "line_length"]]
    current_cable_roads_table["current_wood_volume"] = pd.Series(dtype="int")
    current_cable_roads_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(
                    values=[
                        "Cable Road Cost",
                        "Cable Road Length",
                        "Wood Volume per Cable Road",
                        "Supports Amount",
                        "Supports Height",
                    ]
                ),
                cells=dict(values=[]),
            )
        ]
    )
    current_cable_roads_table_figure.update_layout(
        title="Current Cable Roads Overview",
        height=250,
        margin=dict(r=30, l=30, t=30, b=30),
    )

    # and for the current layout overview
    layout_columns = [
        "Total Layout Costs",
        "Ecological Penalty",
        "Ergonomic Penalty",
        "Selected Cable Roads",
        "Max Yarding Distance",
        "Average Yarding Distance",
    ]
    layout_overview_df = pd.DataFrame(columns=layout_columns)
    layout_overview_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(values=layout_columns),
                cells=dict(values=[layout_overview_df]),
            )
        ]
    )
    layout_overview_table_figure.update_layout(
        title="Current Layout Overview", height=150, margin=dict(r=30, l=30, t=30, b=30)
    )

    # as well as the layout comparison table
    layout_comparison_df = pd.DataFrame(columns=layout_columns)
    layout_comparison_table_figure = go.FigureWidget(
        [
            go.Table(
                header=dict(values=layout_columns),
                cells=dict(values=[]),
            )
        ]
    )
    layout_comparison_table_figure.update_layout(title="Layout Comparison")

    def plot_pareto_frontier(
        results_df,
        current_indices,
        interactive_layout,
        layout_overview_table_figure,
        current_cable_roads_table_figure,
        current_cable_roads_table,
        forest_area_3,
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
            width=800,
            height=400,
            scene=dict(
                xaxis_title="Ecological Optimality",
                yaxis_title="Ergonomics Optimality",
                zaxis_title="Cost Optimality",
                xaxis={"autorange": "reversed"},
            ),
            scene_camera_eye=dict(x=1.7, y=1.7, z=1),
            scene_camera_center=dict(x=0, y=0, z=-0.5),
            margin=dict(r=30, l=30, t=30, b=30),
        )

        text = """A Pareto Frontier represents the set of all non-dominated solutions, i.e. solutions where none of the objective functions can be improved in value without degrading some of the other objective values.Here, we consider three objectives: cost, relative ergonomical impact, and relative ecological impact. Each point on the Pareto Frontier represents a unique combination of these three objectives that is Pareto optimal. No point on the Pareto Frontier can be improved in one objective without worsening at least one of the other objectives."""

        # create a widget with the text
        pareto_explanation_widget = Textarea(
            value=text,
            description="Explanation:",
            disabled=True,
            layout=Layout(width="90%", height="50%"),
        )

        def selection_fn(trace, points, selector):
            nonlocal current_indices
            # get index of this point in the trace
            index = points.point_inds[0]

            # get the corresponding list of activated cable rows from the dataframe
            current_indices = results_df.iloc[index]["selected_lines"]

            update_interactive_based_on_indices(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                current_indices,
                interactive_layout,
                forest_area_3,
                model_list,
                transparent_line,
                solid_line,
            )

        pareto_frontier.data[0].on_click(selection_fn)
        return pareto_frontier, pareto_explanation_widget

    # get the pareto frontier as 3d scatter plot
    pareto_frontier, pareto_explanation_widget = plot_pareto_frontier(
        results_df,
        current_indices,
        interactive_layout,
        layout_overview_table_figure,
        current_cable_roads_table_figure,
        current_cable_roads_table,
        forest_area_3,
        transparent_line,
        solid_line,
        model_list,
    )

    # 3d scatter plot for viewing the layout in 3d
    layout_3d_scatter_plot = go.FigureWidget(go.Scatter3d())

    layout_3d_scatter_plot.update_layout(
        title="""3D Plot""",
        width=1000,
        height=600,
        scene_camera_eye=dict(x=0.5, y=0.5, z=1),
    )

    # create the onclick function to select new CRs
    def selection_fn(trace, points, selector):
        # since the handler is activated for all lines, test if this one has coordinates, ie. is the clicked line
        if points.xs:
            # set the selected cr to none by default
            nonlocal selected_cr
            selected_cr = None

            # deactive this if it is active
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

                # and set the selected cr to the name of the trace if it is a new one
                selected_cr = int(trace.name)

            # get all active traces  - ie those which are not lightgrey. very heavy-handed expression, but it works
            active_traces = list(
                interactive_layout.select_traces(
                    selector=lambda x: (
                        True
                        if x.line.color
                        and x.line.color != color_transparent
                        and x.name != "Contour"
                        else False
                    )
                )
            )

            nonlocal current_indices
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
                forest_area_3,
                model_list,
            )

    # add the onclick function to all line traces
    for trace in interactive_layout.data[2:]:
        trace.on_click(selection_fn)

    # add the custom buttons
    def set_current_cr(left=False):
        """
        Function to set the currently selected cr to the next one
        Refers to the nonlocal variables selected_cr and current_indices
        First we get the index of the cr, then we set the current cr to lightgrey, then we increment/decrement the cr, then we set the new cr to black
        And finally we update the tables and the layout
        """
        nonlocal selected_cr
        nonlocal current_indices

        # if there are no current indices, return
        if selected_cr is None:
            return

        # get the index of the currently selected cr
        index_cr = current_indices.index(selected_cr)

        # make this trace lightgrey
        interactive_layout.update_traces(
            line=transparent_line,
            selector={"name": str(selected_cr)},
        )

        # in/decrement the cr
        selected_cr = selected_cr - 1 if left else selected_cr + 1

        # and set the selected_cr on the index
        current_indices[index_cr] = selected_cr

        # update this trace to turn black
        interactive_layout.update_traces(
            line=solid_line,
            selector={"name": str(selected_cr)},
        )

        update_colors_and_tables(
            current_cable_roads_table_figure,
            current_cable_roads_table,
            layout_overview_table_figure,
            current_indices,
            interactive_layout,
            forest_area_3,
            model_list,
        )

    def move_left_callback(button):
        set_current_cr(left=True)

    def move_right_callback(button):
        set_current_cr(left=False)

    def add_to_comparison_callback(button):
        """
        Function to add the current layout to the comparison table
        """
        nonlocal layout_comparison_df
        nonlocal buttons

        # append the current data from the layout overview table to the comparison table
        layout_comparison_df.loc[len(layout_comparison_df) + 1] = (
            layout_overview_table_figure.data[0].cells.values
        )

        # and update the figure accordingly
        layout_comparison_table_figure.data[0].cells.values = (
            layout_comparison_df.values.T
        )

        recreate_dropdown_menu()

    def reset_comparison_table_callback(button):
        """
        Function to reset the comparison table by emptying the figure data
        """
        layout_comparison_table_figure.data[0].cells.values = []
        # reset the dataframe
        nonlocal layout_comparison_df
        layout_comparison_df = pd.DataFrame(columns=[layout_columns])

        # and reset the dropdown menu
        recreate_dropdown_menu()

    def create_dropdown_menu():
        """
        Function to recreate the dropdown menu based on the current indices
        """
        nonlocal layout_comparison_df

        # recreate the dropdown menu with the current indices
        dropdown_menu = Dropdown(
            options=[""],
            description="Load custom Layout",
        )

        return dropdown_menu

    def recreate_dropdown_menu():
        """
        Function to recreate the dropdown menu based on the current indices
        """
        nonlocal layout_comparison_df
        nonlocal buttons

        # recreate the dropdown menu with the current indices and one empty selection
        buttons[5].options = [""] + [
            str(index) for index in range(len(layout_comparison_df))
        ]

    def dropdown_menu_callback(change):
        """
        Function to load a custom layout from the dropdown menu
        """
        if change["type"] == "change" and change["name"] == "value":
            if change["new"] == "":
                return

            nonlocal layout_comparison_df
            nonlocal buttons

            # get the index of the selected layout
            selected_index = int(change.new)

            # get the corresponding list of activated cable rows from the dataframe
            corresponding_indices = layout_comparison_df.iloc[selected_index][
                "Selected Cable Roads"
            ][0]

            update_interactive_based_on_indices(
                current_cable_roads_table_figure,
                current_cable_roads_table,
                layout_overview_table_figure,
                corresponding_indices,
                interactive_layout,
                forest_area_3,
                model_list,
                transparent_line,
                solid_line,
            )

    def view_in_3d_callback(button, scatterplot=layout_3d_scatter_plot):
        """
        Function to view the current layout in 3d. This updates the layout_3d_scatter_plot with the new 3d scatterplot based on the current indices
        """
        nonlocal current_indices

        # reset the scatterplot
        scatterplot.data = []

        # get the new traces
        new_figure_traces = plotting_3d.plot_all_cable_roads(
            forest_area_3.height_gdf, forest_area_3.line_gdf.iloc[current_indices]
        ).data

        scatterplot.add_traces(new_figure_traces)

    def reset_button_callback(button):
        """
        Function to reset the currently selected cable roads
        """
        nonlocal current_indices
        nonlocal selected_cr

        # reset the selected cr to none
        selected_cr = None

        # reset the current indices
        current_indices = []

        # reset the dropdown menu
        recreate_dropdown_menu()

        # reset the layout
        interactive_layout.update_traces(line=transparent_line)

        # reset the tables
        current_cable_roads_table_figure.data[0].cells.values = [
            [],
            [],
            [],
        ]

        layout_overview_table_figure.data[0].cells.values = [
            [],
            [],
            [],
            [],
        ]

    def create_buttons(layout_3d_scatter_plot):
        """
        Define the buttons for interacting with the layout and the comparison table
        """
        move_left_button = Button(description="<-")
        move_right_button = Button(description="->")
        reset_all__CRs_button = Button(description="Reset all CRs")
        view_in_3d_button = Button(description="View in 3D")

        add_layout_to_comparison_button = Button(description="Add layout to comparison")
        reset_comparison_button = Button(description="Reset comparison")
        dropdown_menu = create_dropdown_menu()

        # and bind all the functions to the buttons
        move_left_button.on_click(move_left_callback)
        move_right_button.on_click(move_right_callback)
        reset_all__CRs_button.on_click(reset_button_callback)
        add_layout_to_comparison_button.on_click(add_to_comparison_callback)
        reset_comparison_button.on_click(reset_comparison_table_callback)
        dropdown_menu.observe(dropdown_menu_callback)
        view_in_3d_button.on_click(
            partial(view_in_3d_callback, scatterplot=layout_3d_scatter_plot)
        )

        return (
            move_left_button,
            move_right_button,
            reset_all__CRs_button,
            add_layout_to_comparison_button,
            reset_comparison_button,
            dropdown_menu,
            view_in_3d_button,
        )

    buttons = list(create_buttons(layout_3d_scatter_plot))

    return (
        interactive_layout,
        current_cable_roads_table_figure,
        layout_overview_table_figure,
        pareto_frontier,
        pareto_explanation_widget,
        buttons[0],
        buttons[1],
        buttons[2],
        buttons[3],
        buttons[4],
        buttons[5],
        buttons[6],
        layout_comparison_table_figure,
        layout_3d_scatter_plot,
    )
