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

def plot_lcsp_results(model, facility_points_gdf, demand_points_gdf, facility_count, title, p):
    """Plot the results from the location set covering problem.
    The original code can be found here(https://pysal.org/spopt/notebooks/facloc-real-world.html),
    but has been adapted to remove their extra colours and the overlay of all lines/facilities

    Args:
        model (_type_): LSCP/MLCP model
        facility_points_gdf (_type_): line locations
        demand_points_gdf (_type_): tree locations
        facility_count (_type_): _description_
        title (_type_): _description_
        p (_type_): A factor influencing the markersize (_ms)?
    """    
    arr_points = []
    fac_sites = []

    for i in range(facility_count):
        if model.fac2cli[i]:
            geom = demand_points_gdf.iloc[model.fac2cli[i]]["geometry"]
            arr_points.append(geom)
            fac_sites.append(i)

    fig, ax = plt.subplots(figsize=(10, 15))
    legend_elements = []

    demand_points_gdf.plot(
        ax=ax, fc="k", ec="k", marker="s", markersize=7, zorder=2
    )

    legend_elements.append(
        mlines.Line2D(
            [],
            [],
            marker="s",
            markerfacecolor="k",
            markeredgecolor="k",
            ms=3,
            linewidth=0,
            label=f"Demand sites not covered"
        )
    )

    # facility_points_gdf.plot(
    #     ax=ax, fc="brown", marker="*", markersize=80, zorder=8
    # )
    
    legend_elements.append(
        mlines.Line2D(
            [],
            [],
            marker="*",
            markerfacecolor="brown",
            markeredgecolor="brown",
            ms=7,
            lw=0,
            label=f"Store sites ($n$={facility_count})"
        )
    )

    _zo, _ms = 4, 4
    for i in range(len(arr_points)):

        #cset = dv_colors[i]
        fac = fac_sites[i]
        fname = facility_points_gdf.iloc[[fac]]["index_column"].iloc[0]

        gdf = geopandas.GeoDataFrame(arr_points[i])

        label = f"Demand sites covered by {fname}"
        gdf.plot(ax=ax, zorder=_zo, ec="k", markersize=100*_ms)
        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                marker="o",
#                markerfacecolor=cset,
                markeredgecolor="k",
                ms= _ms + 7,
                lw=0,
                label=label
            )
        )

        facility_points_gdf.iloc[[fac]].plot(
            ax=ax, marker="*", markersize=1000, zorder=9, ec="k", lw=2
        )
        legend_elements.append(
            mlines.Line2D(
                [],
                [],
                marker="*",
#                markerfacecolor=cset,
                markeredgecolor="k",
                markeredgewidth=2,
                ms=20,
                lw=0,
                label=fname,
            )
        )

        _zo += 1
        _ms -= (1)*(4/p)

    plt.title(title, fontsize=20)
    kws = dict(loc="upper left", bbox_to_anchor=(1.05, .7), fontsize=15)
    plt.legend(handles=legend_elements, **kws)

    x, y, xyc, arrow_length, c = 0.925, 0.15, "axes fraction", 0.1 , "center"
    xy, xyt = (x, y), (x, y-arrow_length)
    ap = dict(facecolor="black", width=5, headwidth=10)
    kws = dict(arrowprops=ap, ha=c, va=c, fontsize=20)
    plt.annotate("N", xy=xy, xycoords=xyc, xytext=xyt, **kws)

    plt.gca().add_artist(ScaleBar(1))
    plt.show()