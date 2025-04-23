import geopandas as gpd
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from itertools import chain

from itertools import cycle

import plotly.graph_objects as go
from src.main import optimization_compute_quantification, classes_linear_optimization
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.main.classes_linear_optimization import result_object


def plot_gdfs(gdfs: list):
    """Plot a list of gdfs"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel("X coordinate", fontsize=15)
    ax.set_ylabel("Y coordinate", fontsize=15)
    for gdf in gdfs:
        gdf.plot(alpha=0.5, ax=ax)


def plot_gdf_with_anchors_and_supports(gdfs: list, line_gdf: gpd.GeoDataFrame):
    """Plot all elements of a line gdf with anchors and supports

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


def add_geometries_to_fig(
    object, fig: go.Figure, marker_color: str, name: str = ""
) -> go.Figure:
    """Add a scatter of the geometry xs and ys to a plotly figure"""
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


def expert_results_extraction(
    fac2cli: list,
    line_gdf: gpd.GeoDataFrame,
    aij: np.ndarray,
    distance_carriage_support: np.ndarray,
    productivity_cost_matrix: np.ndarray,
    tree_volumes_list: pd.Series,
    ecological_penalty_lateral_distances: pd.Series,
    ergonomic_penalty_lateral_distances: np.ndarray,
    name: str,
):
    """Compare the results of the different models in one table
    Args:
        result_list (list): a list of the models with different tradeoffs
        productivity_cost_matrix (np.ndarray): the productivity cost matrix
        aij (np.ndarray): the distance matrix
        distance_carriage_support (np.ndarray): the distance matrix
    """
    productivity_array = []
    aij_array = []
    distance_carriage_support_array = []
    overall_profit = []
    cable_road_costs = []
    facility_cost = line_gdf["line_cost"].values

    ecological_distances = []
    overall_ergonomic_penalty_lateral_distances = []

    total_profit_per_layout_baseline = 0
    for index, row in enumerate(fac2cli):
        if row:
            profit_per_row = tree_volumes_list.iloc[row] * 80
            profit_this_cr = profit_per_row.sum()
            total_profit_per_layout_baseline += profit_this_cr

    # and the corresponding rows from the distance matrix
    row_sums = []
    for index, row in enumerate(fac2cli):
        if row:
            distance_per_this_row = aij[row, index]
            row_sum_distance = distance_per_this_row.sum()
            row_sums.append(row_sum_distance)
    aij_array.append(sum(row_sums))

    row_sums = []
    for index, row in enumerate(fac2cli):
        if row:
            productivity_per_row = productivity_cost_matrix[row, index]
            row_sum_distance = productivity_per_row.sum()
            row_sums.append(row_sum_distance)
    productivity_array.append(sum(row_sums))

    row_sums = []
    for index, row in enumerate(fac2cli):
        if row:
            distance_per_this_row = distance_carriage_support[row, index]
            row_sum_distance = distance_per_this_row.sum()
            row_sums.append(row_sum_distance)
    distance_carriage_support_array.append(sum(row_sums))

    # subtract the productivity cost from the total profit
    overall_profit.append(total_profit_per_layout_baseline - productivity_array[-1])

    # get the total cable road costs
    total_cable_road_costs = 0
    for index, row in enumerate(fac2cli):
        if row:
            cable_road_cost = facility_cost[index]
            total_cable_road_costs += cable_road_cost
    cable_road_costs.append(total_cable_road_costs)

    (
        ecological_distances_here,
        bad_ergonomic_distance_here,
    ) = optimization_compute_quantification.get_secondary_objective_values_with_fac2cli(
        fac2cli,
        ecological_penalty_lateral_distances,
        ergonomic_penalty_lateral_distances,
    )

    ecological_distances.append(ecological_distances_here)
    overall_ergonomic_penalty_lateral_distances.append(bad_ergonomic_distance_here)

    overall_profit_unscaled = np.array(overall_profit)  # * profit_scaling
    profit_baseline = min(overall_profit_unscaled)
    print(f"Profit baseline is {profit_baseline}")
    profit_comparison = overall_profit_unscaled - profit_baseline

    name_list = name

    return pd.DataFrame(
        data={
            "Total distance of trees to cable roads": aij_array,
            "Productivity cost per m3 as per Stampfer": productivity_array,
            "Total distance from carriage to support": distance_carriage_support_array,
            "overall_profit": overall_profit,
            "cable_road_costs": cable_road_costs,
            "profit_comparison": profit_comparison,
            "name": name_list,
            "ecological_distances": ecological_distances,
            "bad_ergonomics_distance": overall_ergonomic_penalty_lateral_distances,
        }
    )


def plot_optimization_layout(
    result: "result_object",
    line_gdf: gpd.GeoDataFrame,
    harvesteable_trees_gdf: gpd.GeoDataFrame,
):
    arr_points = []
    fac_sites = []
    line_triples = []
    tree_anchor = []
    intermediate_supports = []
    line_geometries = []
    demand_points = harvesteable_trees_gdf.reset_index()

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(result.fac2cli)):
        if result.fac2cli[i]:
            line = line_gdf.iloc[i]

            line_geometries.append(line)
            # get the corresponding demand points from the fac2cli entry
            geom = demand_points.iloc[result.fac2cli[i]]["geometry"]
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
    colours_cycle = cycle(colours)
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
                marker={"color": colours_cycle.__next__()},
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

    fig.update_layout(width=1200, height=800)
    return fig
    # fig.show("notebook_connected")


def plot_NSGA_results(X, F, problem):
    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors="none", edgecolors="r")
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors="none", edgecolors="blue")
    plt.title("Objective Space")
    plt.show()
