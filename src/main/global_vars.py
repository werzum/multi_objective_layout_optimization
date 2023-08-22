import numpy as np
from scipy.spatial import KDTree


def init(height_gdf):
    """Initialize the global variables
    Specifically, the KDTree to query points from the height_gdf
    """
    global kdtree
    kdtree = KDTree(np.column_stack((height_gdf["x"], height_gdf["y"])))
