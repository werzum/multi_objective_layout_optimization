import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.lines as mlines
from matplotlib_scalebar.scalebar import ScaleBar

def plot_gdfs(gdfs):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X coordinate', fontsize=15)
    ax.set_ylabel('Y coordinate', fontsize=15)
    for gdf in gdfs:
        gdf.plot(alpha=0.5,ax=ax)

def plot_scatter_xy(list):
    xs = [point.x for point in list]
    ys = [point.y for point in list]
    plt.scatter(xs, ys)
    plt.show()

def plot_equal_axis(line):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X coordinate', fontsize=15)
    ax.set_ylabel('Y coordinate', fontsize=15)
    ax.plot(*line.xy, label='LineString')
    ax.axis('equal')
    return ax

def onclick(event, coords):
    """Return an interactive figure to record mouse clicks on their coordinates
        Modifies a global coords variable to store the clicked points in
    Args:
        event (_type_): event fired by fig
    """    
    global ix, iy
    ix, iy = event.xdata, event.ydata

    coords.append((ix, iy))

    if len(coords) == 6:
        print("Coordinates recorded are:",coords)
        coords = []
        print("Array reset")


def plot_p_median_results(model, facility_points_gdf, demand_points_gdf, anchor_trees, target_trees, line_gdf):
    """ Plot the results of the P-Median optimization. Based on this https://pysal.org/spopt/notebooks/p-median.html, but heavily shortened.

    Args:
        model (_type_): the P-Median model
        facility_points_gdf (_type_): A gdf with the facilities (ie. factories)
        demand_points_gdf (_type_): A gdf with the demand points (ie. trees)
    """    
    arr_points = []
    fac_sites = []
    line_triples = []

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(facility_points_gdf)):
        if model.fac2cli[i]:
            # get the corresponding demand points from the fac2cli entry
            geom = demand_points_gdf.iloc[model.fac2cli[i]]['geometry']
            arr_points.append(geom)
            fac_sites.append(i)
            # get the corresponding anchor triple from the line_gdf
            line_triples.append(line_gdf.iloc[i]["possible_anchor_triples"])

    fig, ax = plt.subplots(figsize=(12,12))
    legend_elements = []

    #ugly decomprehension
    unwrapped_triples = []
    for item in line_triples:
        unwrapped_triples.append(gpd.GeoSeries(sum(item, [])))

    # add the trees with respective color to which factory they belong to the map
    for i in range(len(arr_points)):
        gdf = gpd.GeoDataFrame(arr_points[i])
        anchor_lines_gdf = gpd.GeoDataFrame(geometry=unwrapped_triples[i])

        label = f"coverage_points by y{fac_sites[i]}"
        legend_elements.append(Patch(label=label))

        gdf.plot(ax=ax, zorder=3, alpha=0.7, label=label)
        facility_points_gdf.iloc[[fac_sites[i]]].plot(ax=ax,
                                marker="*",
                                markersize=200 * 3.0,
                                alpha=0.8,
                                zorder=4,
                                edgecolor="k")

        anchor_lines_gdf.plot(ax=ax)

        legend_elements.append(mlines.Line2D(
            [],
            [],
            marker="*",
            ms=20 / 2,
            linewidth=0,
            alpha=0.8,
            label=f"y{fac_sites[i]} facility selected",
        ))

    anchor_trees.plot(ax=ax)
    target_trees.plot(ax=ax)

    plt.title("P-Median", fontweight="bold")
    plt.legend(handles = legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))


def plot_pymoo_results(model, facility_points_gdf, demand_points_gdf, anchor_trees, target_trees, line_gdf):
    """ Plot the results of the P-Median optimization. Based on this https://pysal.org/spopt/notebooks/p-median.html, but heavily shortened.

    Args:
        model (_type_): the P-Median model
        facility_points_gdf (_type_): A gdf with the facilities (ie. factories)
        demand_points_gdf (_type_): A gdf with the demand points (ie. trees)
    """    
    arr_points = []
    fac_sites = []
    line_triples = []

    # extract the cli_assgn_vars and fac_vars from the model/x results of the pymoo optimization
    variable_matrix = model[0].reshape((len(demand_points_gdf)+1,len(line_gdf)))

    cli_assgn_vars = variable_matrix[:-1]
    fac_vars = variable_matrix[-1:][0]

    # fill arr_points and fac_sites for non-empty entries in the facilities to clients array
    for i in range(len(facility_points_gdf)):
        if fac_vars[i]:
            # get the corresponding demand points from the fac2cli entry
            # get all entries for this column in cli_assgn_vars
            geom = demand_points_gdf.iloc[cli_assgn_vars[:,i]]['geometry']
            arr_points.append(geom)
            fac_sites.append(i)
            # get the corresponding anchor triple from the line_gdf
            line_triples.append(line_gdf.iloc[i]["possible_anchor_triples"])

    fig, ax = plt.subplots(figsize=(12,12))
    legend_elements = []

    #ugly decomprehension
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
        facility_points_gdf.iloc[[fac_sites[i]]].plot(ax=ax,
                                marker="*",
                                markersize=200 * 3.0,
                                alpha=0.8,
                                zorder=4,
                                edgecolor="k")

        anchor_lines_gdf.plot(ax=ax)

        legend_elements.append(mlines.Line2D(
            [],
            [],
            marker="*",
            ms=20 / 2,
            linewidth=0,
            alpha=0.8,
            label=f"y{fac_sites[i]} facility selected",
        ))

    anchor_trees.plot(ax=ax)
    target_trees.plot(ax=ax)

    plt.title("P-Median", fontweight="bold")
    plt.legend(handles = legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))