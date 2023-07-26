import numpy as np
from scipy.spatial import KDTree


def init(height_gdf):
    global kdtree
    kdtree = KDTree(np.column_stack((height_gdf["x"], height_gdf["y"])))
