import rasterio
from rasterio import features
from shapely.geometry import Polygon, Point
import pandas as pd
import geopandas as gpd
import numpy as np

# from src.main import mechanical_computations


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
        for geom, val in features.shapes(mask, transform=dataset.transform):
            geom_array.append(Polygon(geom["coordinates"][0]))

        # stackoverflow magic to read the height values to an array https://stackoverflow.com/questions/51485603/converting-numpy-structured-array-to-pandas-dataframes
        arr = dataset.read(1)
        mask = arr != dataset.nodata
        elev = arr[mask]
        col, row = np.where(mask)
        x, y = dataset.xy(col, row)
        uid = np.arange(dataset.height * dataset.width).reshape(
            (dataset.height, dataset.width)
        )[mask]

    height_df = pd.DataFrame(
        np.rec.fromarrays([uid, x, y, elev], names=["id", "x", "y", "elev"])
    ).set_index("id")

    return geom_array, height_df


def read_csv(file_path: str) -> pd.DataFrame:
    """Reads a given csv file and returns a dataframe"""
    with open(file_path) as file:
        bestand = pd.read_csv(file, sep="\t", dtype=str)
    return bestand


def format_tree_dataframe(csv: pd.DataFrame) -> pd.DataFrame:
    """
    Format the tree dataframe to the correct format, especially the columns to numeric

    """
    # ensure the csv columns are parsed correctly - especially with the separators
    # convert just columns "a" and "b"
    columns_to_change = ["x", "y", "z", "id", "BHD", "crownVolume", "h"]
    csv[columns_to_change] = (
        csv[columns_to_change].replace(",", ".", regex=True).astype(float)
    )
    csv[columns_to_change] = csv[columns_to_change].apply(pd.to_numeric)
    return csv


def parse_point(point_string):
    # Remove the "POINT " prefix and parentheses from the string
    point_string = point_string.replace("POINT (", "").replace(")", "")
    # Split the string into two coordinates
    x, y = point_string.split()
    # Convert the coordinates to floats and create a new Point object
    return Point(float(x), float(y))


def parse_list_int(list_string):
    removed_stopsigns = (
        list_string.replace("[", "").replace("]", "").replace(",", "").split()
    )
    parsed_numbers = [int(float(x)) for x in removed_stopsigns]
    return parsed_numbers


def parse_list_float(list_string):
    removed_stopsigns = (
        list_string.replace("[", "").replace("]", "").replace(",", "").split()
    )
    parsed_numbers = [float(x) for x in removed_stopsigns]
    return parsed_numbers


def load_processed_gdfs(data_to_load: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the processed dataframes from the csv files
    """
    with open(
        f"03_Data/Resources_Organized/Dataframes_Processed/Bestand_{data_to_load}.csv"
    ) as file:
        bestand = pd.read_csv(file)

    with open(
        f"03_Data/Resources_Organized/Dataframes_Processed/Height_{data_to_load}.csv"
    ) as file:
        height = pd.read_csv(file)

    bestand_gdf = gpd.GeoDataFrame(
        bestand, geometry=gpd.points_from_xy(bestand.x, bestand.y)
    )

    height_gdf = gpd.GeoDataFrame(
        height, geometry=gpd.points_from_xy(height.x, height.y)
    )

    # convert the objectified column back to lists of floats
    for column in ["max_supported_force_series", "height_series", "diameter_series"]:
        bestand[column] = list(map(eval, bestand[column]))

    return bestand_gdf, height_gdf


def load_raw_bestand_forest_height_dfs(tif_to_load: int):
    # load the tif
    tif_shapes, height_df = read_tif(
        f"03_Data/Resources_Organized/tif/Bestand{tif_to_load}.tif"
    )

    forest_area_gdf = gpd.GeoDataFrame(
        pd.DataFrame({"name": ["area1", "area2", "area3", "area4", "area5"]}),
        geometry=tif_shapes,
    )

    bestand_csv = read_csv(
        f"03_Data/Resources_Organized/csv/Bestand{tif_to_load}_h.csv"
    )
    tree_df = format_tree_dataframe(bestand_csv)

    return tree_df, forest_area_gdf, height_df


# def preprocess_raw_dataframes(
#     tree_df, forest_area_gdf, height_df, data_to_load: int
# ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Preprocess the raw dataframes by cleaning them up and adding the height series and diameter series
#     """

#     # clean up the data
#     tree_df.dropna(inplace=True)
#     tree_df = tree_df[tree_df["h"] > 0]
#     # dont do that - we want to keep the trees with no crown volume
#     # TODO - reprocess everything
#     # tree_df = tree_df[tree_df["crownVolume"].astype(int) > 0]
#     tree_df.reset_index(drop=True, inplace=True)

#     # # recreate the height and diameter series
#     durchmesser_csv = pd.read_csv(
#         f"03_Data/Resources_Organized/Dataframes_Processed/Diameter_Series_{data_to_load}.csv"
#     )

#     # only get the integer heights
#     durchmesser_csv = durchmesser_csv[durchmesser_csv.hoehe % 1 == 0]

#     # get the unique ids
#     kes = tree_df.id.unique()
#     id_dict = {}
#     dm_dict = {}
#     # get a corresponding dict of diameters at each height
#     for ke in kes:
#         id_dict[ke] = durchmesser_csv[durchmesser_csv["tree.id"] == ke][
#             "hoehe"
#         ].to_list()
#         dm_dict[ke] = durchmesser_csv[durchmesser_csv["tree.id"] == ke][
#             "durchmesser"
#         ].to_list()

#     tree_df["height_series"] = tree_df["id"].map(id_dict)
#     tree_df["diameter_series"] = tree_df["id"].map(dm_dict)

#     # get the euler forces
#     list_of_euler_max_force_lists = []
#     for index, row in tree_df.iterrows():
#         temp_list = [
#             mechanical_computations.euler_knicklast(bhd, height)
#             for bhd, height in zip(row["diameter_series"], row["height_series"])
#         ]
#         list_of_euler_max_force_lists.append(temp_list)

#     tree_df["max_supported_force_series"] = list_of_euler_max_force_lists

#     tree_df["max_holding_force"] = (((tree_df["BHD"] * 0.1) ** 2) / 3) * 10000

#     return tree_df, forest_area_gdf, height_df


# one off for loading and preprocessing raw data
def load_and_preprocess_raw_data(data_to_load: int):
    """
    One off function to load and preprocess the raw data
    Args:
        data_to_load (int): Which dataframe to load

    """
    # load raw data
    tree_df, forest_area_gdf, height_df = load_raw_bestand_forest_height_dfs(
        data_to_load
    )

    # preprocess raw data
    tree_df, forest_area_gdf, height_df = preprocess_raw_dataframes(
        tree_df, forest_area_gdf, height_df, data_to_load
    )

    # write them to disk
    tree_df.to_csv(
        f"03_Data/Resources_Organized/Dataframes_Processed/Bestand_{data_to_load}.csv"
    )
    height_df.to_csv(
        f"03_Data/Resources_Organized/Dataframes_Processed/Height_{data_to_load}.csv"
    )
