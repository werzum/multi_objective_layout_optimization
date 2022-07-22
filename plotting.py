import matplotlib
import matplotlib.pyplot as plt

def plot_gdfs(gdfs):
    fig, ax = plt.subplots(figsize = (20,16)) 
    for gdf in gdfs:
        gdf.plot(alpha=0.5,ax=ax)

def plot_scatter_xy(list):
    xs = [point.x for point in list]
    ys = [point.y for point in list]
    plt.scatter(xs, ys)