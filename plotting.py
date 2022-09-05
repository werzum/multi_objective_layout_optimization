import geopandas
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

def plot_optimization(target_function, results, xy):
    """Plot the figure showing the results of an optimization

    Args:
        target_function (_type_): _description_
        results (_type_): _description_
        xy (_type_): _description_
    """    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(target_function(xy), interpolation='bilinear', origin='lower',
                cmap='gray')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def plot_point(res, marker='o', color=None):
        ax.plot(512+res.x[0], 512+res.x[1], marker=marker, color=color, ms=10)

    # SHGO produces multiple minima, plot them all (with a smaller marker size)
    plot_point(results['shgo'], color='r', marker='+')
    plot_point(results['shgo_sobol'], color='r', marker='x')

    for i in range(results['shgo_sobol'].xl.shape[0]):
        ax.plot(512 + results['shgo_sobol'].xl[i, 0],
                512 + results['shgo_sobol'].xl[i, 1],
                'ro', ms=2)

    ax.set_xlim([-4, 514*2])
    ax.set_ylim([-4, 514*2])
    plt.show()


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
            geom = demand_points_gdf.iloc[model.fac2cli[i]]['geometry']
            arr_points.append(geom)
            fac_sites.append(i)
            line_triples.append(line_gdf.iloc[model.fac2cli[i]]["possible_anchor_triples"])

    fig, ax = plt.subplots(figsize=(12,12))
    legend_elements = []

    # add the trees with respective color to which factory they belong to the map
    for i in range(len(arr_points)):
        gdf = geopandas.GeoDataFrame(arr_points[i])
        anchor_lines_gdf = geopandas.GeoDataFrame(line_triples[i])

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