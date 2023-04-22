from typing import Tuple
from pandas import DataFrame
import pandas as pd
import numpy as np
import geopandas as gpd

COLUMNS_TO_ENCODE = [
    "town",
    "mrt_type",
    "mrt_codes",
    "distance_to_mrt_bins",
    "male_female_ratio_bins",
    "adult_children_ratio_bins",
    "population_bins",
    "type_commerical",
]
COLUMNS_TO_DROP = [
    "town",
    "block",
    "flat_type",
    "street_name",
    "storey_range",
    "flat_model",
    "eco_category",
    "lease_commence_date",
    "elevation",
    "subzone",
    "planning_area",
    "region",
    "mrt_name",
    "mrt_type",
    "mrt_codes",
    "codes",
    "codes_name",
    "distance_to_mrt_bins",
    "male_female_ratio_bins",
    "adult_children_ratio_bins",
    "population_bins",
    "type_commerical",
    "opening_year",
]


def preprocess_month(data: DataFrame):
    data["rebased_month"] = (data["datetime"].dt.year -
                             2000) * 12 + data["datetime"].dt.month


def preprocess_lease_commence_date(data: DataFrame):
    data["remaining_lease"] = data["lease_commence_date"] + \
        99 - data["datetime"].dt.year


def preprocess_flat_type(train: DataFrame, test: DataFrame) -> Tuple[DataFrame, DataFrame]:
    # correct inconsistent values
    train["flat_type"] = train["flat_type"].replace(
        to_replace="\s", value="-", regex=True)
    test["flat_type"] = test["flat_type"].replace(
        to_replace="\s", value="-", regex=True)

    # one-hot
    oneHot = pd.get_dummies(train["flat_type"])
    train_processed = train.join(oneHot)
    oneHot = pd.get_dummies(test["flat_type"])
    test_processed = test.join(oneHot)

    # target encoding
    mapping = train.groupby("flat_type")["resale_price"].mean().to_dict()
    train_processed["flat_type_price"] = train["flat_type"].map(mapping)
    test_processed["flat_type_price"] = test["flat_type"].map(mapping)

    mapping = train.groupby("flat_type")["price_psm"].mean().to_dict()
    train_processed["flat_type_psm"] = train["flat_type"].map(mapping)
    test_processed["flat_type_psm"] = test["flat_type"].map(mapping)

    # label encoding
    mapping = {
        "1-room": 1,
        "2-room": 2,
        "3-room": 3,
        "4-room": 4,
        "5-room": 5,
        "multi-generation": 6,
        "executive": 7,
    }
    train_processed["flat_type_number"] = train["flat_type"].map(mapping)
    test_processed["flat_type_number"] = test["flat_type"].map(mapping)

    return train_processed, test_processed


def preprocess_storey_range(train: DataFrame, test: DataFrame):
    # merge to range of 3. 1-based index
    train["storey_range_start"] = train["storey_range"].str[:2].astype("int")
    train["storey_range_processed"] = np.ceil(
        (train["storey_range_start"] - 1) / 3) + 1
    test["storey_range_start"] = test["storey_range"].str[:2].astype("int")
    test["storey_range_processed"] = np.ceil(
        (test["storey_range_start"] - 1) / 3) + 1

    # target encoding
    mapping = train.groupby(["storey_range_processed"])[
        "resale_price"].mean().to_dict()
    train["storey_range_price"] = train["storey_range_processed"].map(mapping)
    test["storey_range_price"] = test["storey_range_processed"].map(mapping)

    mapping = train.groupby(["storey_range_processed"])[
        "price_psm"].mean().to_dict()
    train["storey_range_price_psm"] = train["storey_range_processed"].map(
        mapping)
    test["storey_range_price_psm"] = test["storey_range_processed"].map(
        mapping)


def preprocess_flat_model(train: DataFrame, test: DataFrame):
    mapping = train.groupby(["flat_model"])["resale_price"].mean().to_dict()
    train["flat_model_price"] = train["flat_model"].map(mapping)
    test["flat_model_price"] = test["flat_model"].map(mapping)

    mapping = train.groupby(["flat_model"])["price_psm"].mean().to_dict()
    train["flat_model_psm"] = train["flat_model"].map(mapping)
    test["flat_model_psm"] = test["flat_model"].map(mapping)


def before_preprocess_lat_lon(data: DataFrame):
    # init and tell geopandas this is lat, long coordinates
    geodata = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326",
    )
    x_data = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(data["longitude"], data["min_lat"]),
        crs="EPSG:4326",
    )
    y_data = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(data["min_lon"], data["latitude"]),
        crs="EPSG:4326",
    )
    # transform coordinate so that can exact distance (in km) can be calculated
    geodata.to_crs(epsg=3414, inplace=True)
    x_data.to_crs(epsg=3414, inplace=True)
    y_data.to_crs(epsg=3414, inplace=True)
    # calculate distance
    # adjust grid size if necessary
    data["x_grid"] = (geodata.distance(y_data) / 1000).astype("int")
    data["y_grid"] = (geodata.distance(x_data) / 1000).astype("int")


def preprocess_lat_lon(train: DataFrame, test: DataFrame):
    before_preprocess_lat_lon(train)
    before_preprocess_lat_lon(test)

    mapping = train.groupby(["x_grid", "y_grid"])[
        "resale_price"].mean().to_dict()
    train["grid_price"] = train[["x_grid", "y_grid"]].agg(
        tuple, 1).map(mapping)
    test["grid_price"] = test[["x_grid", "y_grid"]].agg(tuple, 1).map(mapping)

    mapping = train.groupby(["x_grid", "y_grid"])["price_psm"].mean().to_dict()
    train["grid_price_psm"] = train[["x_grid", "y_grid"]].agg(
        tuple, 1).map(mapping)
    test["grid_price_psm"] = test[["x_grid", "y_grid"]].agg(
        tuple, 1).map(mapping)

    train.drop(labels=["x_grid", "y_grid"], axis=1, inplace=True)
    test.drop(labels=["x_grid", "y_grid"], axis=1, inplace=True)


def preprocess_zone_street_region(train: DataFrame, test: DataFrame):
    mapping = train.groupby(["subzone"])["resale_price"].mean().to_dict()
    train["subzone_price"] = train["subzone"].map(mapping)
    test["subzone_price"] = test["subzone"].map(mapping)

    mapping = train.groupby(["subzone"])["price_psm"].mean().to_dict()
    train["subzone_price_psm"] = train["subzone"].map(mapping)
    test["subzone_price_psm"] = test["subzone"].map(mapping)

    # train["street_name"] = train["street_name"].str.lower()
    # test["street_name"] = test["street_name"].str.lower()
    # mapping = train.groupby(['street_name'])['resale_price'].mean().to_dict()
    # train["street_name_price"] = train['street_name'].map(mapping)
    # test["street_name_price"] = test['street_name'].map(mapping)

    # mapping = train.groupby(['street_name'])['price_psm'].mean().to_dict()
    # train["street_name_price_psm"] = train['street_name'].map(mapping)
    # test["street_name_price_psm"] = test['street_name'].map(mapping)

    mapping = train.groupby(["planning_area"])["resale_price"].mean().to_dict()
    train["planning_area_price"] = train["planning_area"].map(mapping)
    test["planning_area_price"] = test["planning_area"].map(mapping)

    mapping = train.groupby(["planning_area"])["price_psm"].mean().to_dict()
    train["planning_area_price_psm"] = train["planning_area"].map(mapping)
    test["planning_area_price_psm"] = test["planning_area"].map(mapping)

    mapping = train.groupby(["region"])["resale_price"].mean().to_dict()
    train["region_price"] = train["region"].map(mapping)
    test["region_price"] = test["region"].map(mapping)

    mapping = train.groupby(["region"])["price_psm"].mean().to_dict()
    train["region_price_psm"] = train["region"].map(mapping)
    test["region_price_psm"] = test["region"].map(mapping)


def target_encode(name: str, train: DataFrame, test: DataFrame):
    mapping = train.groupby([name])["resale_price"].mean().to_dict()
    new_column_name = name + "_price"
    train[new_column_name] = train[name].map(mapping)
    test[new_column_name] = test[name].map(mapping)


def preprocess_train_test(
    train_raw: DataFrame, test_raw: DataFrame
) -> Tuple[DataFrame, DataFrame]:
    """Preprocess train and test data. The final output is a tuple of train and test dataframe."""
    train = train_raw.copy().reset_index(drop=True)
    test = test_raw.copy().reset_index(drop=True)
    # temporary attributes
    train["price_psm"] = train["resale_price"] / train["floor_area_sqm"]

    train["datetime"] = pd.to_datetime(train.month)
    test["datetime"] = pd.to_datetime(test.month)

    min_lat = train["latitude"].min()
    min_lon = train["longitude"].min()
    train["min_lat"] = min_lat
    train["min_lon"] = min_lon
    test["min_lat"] = min_lat
    test["min_lon"] = min_lon

    # start preprocessing
    preprocess_month(train)
    preprocess_month(test)

    preprocess_lease_commence_date(train)
    preprocess_lease_commence_date(test)

    train_processed, test_processed = preprocess_flat_type(train, test)
    preprocess_storey_range(train_processed, test_processed)
    preprocess_flat_model(train_processed, test_processed)
    preprocess_lat_lon(train_processed, test_processed)
    preprocess_zone_street_region(train_processed, test_processed)

    for column_name in COLUMNS_TO_ENCODE:
        target_encode(column_name, train_processed, test_processed)

    train_processed.drop(
        labels=["price_psm", "datetime", "min_lat", "min_lon", "month"], axis=1, inplace=True
    )
    test_processed.drop(
        labels=["datetime", "min_lat", "min_lon", "month"], axis=1, inplace=True
    )
    train_processed.drop(columns=COLUMNS_TO_DROP,
                         inplace=True, errors='ignore')
    test_processed.drop(columns=COLUMNS_TO_DROP, inplace=True, errors='ignore')

    train_drop_duplicated = train_processed.drop_duplicates(
        keep="first", inplace=False)
    # print(test_processed[test_processed.isnull().any(axis=1)])

    return train_drop_duplicated, test_processed
