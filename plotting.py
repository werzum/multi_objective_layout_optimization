import matplotlib
import matplotlib.pyplot as plt

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