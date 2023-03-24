import numpy as np
import pandas as pd
import geopandas
from pandas import DataFrame


def preprocess_flat_type(data: DataFrame):
    data["flat_type"] = data.flat_type.replace(to_replace="\s", value="-", regex=True)
    # one-hot
    oneHot = pd.get_dummies(data["flat_type"])
    data = data.join(oneHot)
    # target encoding
    data["flat_type_price"] = data.groupby(["flat_type"])["resale_price"].transform(
        "mean"
    )
    data["flat_type_psm"] = data.groupby(["flat_type"])["price_psm"].transform("mean")
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
    data["flat_type_number"] = data["flat_type"].map(mapping)


def preprocess_storey_range(data: DataFrame):
    # merge to range of 3
    data["storey_range_start"] = data["storey_range"].str[:2].astype("int") - 1
    data["storey_range_processed"] = np.ceil(data["storey_range_start"] / 3) + 1
    # target encoding
    data["storey_range_price"] = data.groupby(["storey_range"])[
        "resale_price"
    ].transform("mean")
    data["storey_range_price_psm"] = data.groupby(["storey_range"])[
        "price_psm"
    ].transform("mean")


def preprocess_flat_model(data: DataFrame):
    data["flat_model_price"] = data.groupby(["flat_model"])["resale_price"].transform(
        "mean"
    )
    data["flat_model_psm"] = data.groupby(["flat_model"])["price_psm"].transform("mean")


def preprocess_lat_lon(data: DataFrame):
    # init and tell geopandas this is lat, long coordinates
    geodata = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(data["longitude"], data["latitude"]),
        crs="EPSG:4326",
    )
    x_data = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(data["longitude"], data["min_lat"]),
        crs="EPSG:4326",
    )
    y_data = geopandas.GeoDataFrame(
        geometry=geopandas.points_from_xy(data["min_lon"], data["latitude"]),
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

    data["grid_price"] = data.groupby(["x_grid", "y_grid"])["resale_price"].transform(
        "mean"
    )
    data["grid_price_psm"] = data.groupby(["x_grid", "y_grid"])["price_psm"].transform(
        "mean"
    )


def preprocess_zone_street_region(data: DataFrame):
    data["subzone_price"] = data.groupby(["subzone"])["resale_price"].transform("mean")
    data["subzone_price_psm"] = data.groupby(["subzone"])["price_psm"].transform("mean")

    data["street_name_price"] = data.groupby(["street_name"])["resale_price"].transform(
        "mean"
    )
    data["street_name_price_psm"] = data.groupby(["street_name"])[
        "price_psm"
    ].transform("mean")

    data["planning_area_price"] = data.groupby(["planning_area"])[
        "resale_price"
    ].transform("mean")
    data["planning_area_price_psm"] = data.groupby(["planning_area"])[
        "price_psm"
    ].transform("mean")

    data["region_price"] = data.groupby(["region"])["resale_price"].transform("mean")
    data["region_price_psm"] = data.groupby(["region"])["price_psm"].transform("mean")


def preprocess(data: DataFrame):
    # temporary attributes
    data["price_psm"] = data["resale_price"] / data["floor_area_sqm"]
    data["datetime"] = pd.to_datetime(data.month)
    data["min_lat"] = data["latitude"].min()
    data["min_lon"] = data["longitude"].min()

    # rebase month to 2000
    data["month"] = (data["datetime"].dt.year - 2000) * 12 + data["datetime"].dt.month

    # get remaining lease
    data["remaining_lease"] = data["lease_commence_date"] + 99 - 2023
    data = data.drop(columns="lease_commence_date")

    preprocess_flat_type(data)
    preprocess_storey_range(data)
    preprocess_flat_model(data)
    preprocess_lat_lon(data)
    preprocess_zone_street_region(data)