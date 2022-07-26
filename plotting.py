import matplotlib
import matplotlib.pyplot as plt

def plot_gdfs(gdfs):
    fig, ax = plt.subplots(figsize = (20,16))
    ax.set_xlabel('X coordinate', fontsize=15)
    ax.set_ylabel('Y coordinate', fontsize=15)
    for gdf in gdfs:
        gdf.plot(alpha=0.5,ax=ax)

def plot_scatter_xy(list):
    xs = [point.x for point in list]
    ys = [point.y for point in list]
    plt.scatter(xs, ys)

def plot_equal_axis(line):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('X coordinate', fontsize=15)
    ax.set_ylabel('Y coordinate', fontsize=15)
    ax.plot(*line.xy, label='LineString')
    ax.axis('equal')
    return ax

def onclick_coords(data):
    """Return an interactive figure to record mouse clicks on their coordinates
        Creates and modifies a global coords variable to store the clicked points in
    Args:
        data (list): A list of the data to be plotted
    """    
    def onclick(event):
        global ix, iy
        ix, iy = event.xdata, event.ydata

        global coords
        coords.append((ix, iy))
        print("Coordinates recorded are:",coords)

        if len(coords) == 8:
            coords = []
            fig.canvas.mpl_disconnect(cid)
            print("Canvas disconnected due to full array")
    
    # ensure that we start with an empty array
    global coords
    coords = []

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[0],data[1])

    cid = fig.canvas.mpl_connect('button_press_event', onclick)