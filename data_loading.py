import rasterio
from shapely.geometry import Polygon
import pandas as pd
from pandas.api.types import is_string_dtype

def read_tif(file_path):
    # Convert tif to geometry
    geom_array = []
    with rasterio.open(file_path) as dataset:

        # Read the dataset's valid data mask as a ndarray.
        mask = dataset.dataset_mask()

        # Extract feature shapes and values from the array.
        for geom, val in rasterio.features.shapes(
                mask, transform=dataset.transform):

            # Transform shapes from the dataset's own coordinate
            # reference system to CRS84 (EPSG:4326).
            # geom = rasterio.warp.transform_geom(
            #     dataset.crs, 'EPSG:4326', geom, precision=6)
            geom_array.append(Polygon(geom["coordinates"][0]))
    return geom_array

def read_csv(file_path):
    with open(file_path) as file:
        bestand = pd.read_csv(file,sep="\t",dtype=str)
    return bestand

def format_csv(csv):
    for column in  ["x","y","z","id"]:
        if is_string_dtype(csv[column]):
            csv[column] = csv[column].str.replace(',', '.').astype(float)
        csv[column] = pd.to_numeric(csv[column])
    return csv
