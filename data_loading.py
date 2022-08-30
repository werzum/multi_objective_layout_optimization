import rasterio
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
from pandas.api.types import is_string_dtype

def read_tif(file_path):
    """Reads a given tif file and extracts its features as raster

    Args:
        file_path (_type_): Filepath

    Returns:
        List: A list of Polygon features
    """    
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
    # ensure the csv columns are parsed correctly - especially with the separators
    for column in  ["x","y","z","id"]:
        if is_string_dtype(csv[column]):
            csv[column] = csv[column].str.replace(',', '.').astype(float)
        csv[column] = pd.to_numeric(csv[column])
    return csv

def load_bestand_and_forest():
    # load the tif
    forest_area_gdf = gpd.GeoDataFrame(pd.DataFrame(
    {"name": ["area1", "area2", "area3", "area4", "area5"]}), 
    geometry=read_tif("Resources_Organized/tif/Bestand3.tif"))

    #load the data and show that we have correctly parsed the CSV
    bestand_3_csv = read_csv("Resources_Organized/csv/Bestand3_h.csv")
    bestand_3_csv = format_csv(bestand_3_csv)

    return forest_area_gdf, bestand_3_csv