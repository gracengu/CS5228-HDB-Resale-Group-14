import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import LabelEncoder

from shapely.geometry import Point
import geopandas as gpd
import pygeos
import geopy
from geopy import distance

import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo


# Global Variables
pio.renderers.default = 'notebook_connected'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DATA_DIRECTORY = "../data/"
DATAPREPROCESSING_DIRECTORY = "../results/data_preprocessing/"
DATAPREPROCESSINGIMG_DIRECTORY = "../img/data_preprocessing/"

# Helper functions


def get_distance_km(row):
    return geopy.distance.geodesic((row['lat_left'], row['lng_left']), (row['lat_right'], row['lng_right'])).km


def preprocess_traintest_data(train, test):
    '''Preprocess train and test data and merge them. The final output is a geopandas dataframe.'''

    # Preprocess train and test data
    train_df = train.rename(columns={"longitude": "lng", "latitude": "lat"})
    train_df["train"] = 1
    test_df = test.rename(columns={"longitude": "lng", "latitude": "lat"})
    test_df["train"] = 0
    full_data = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    full_data["train"] = full_data["train"].astype("str")
    full_data = full_data.drop_duplicates(keep="first", inplace=False)
    full_data["flat_type"] = [
        i.replace(" ", "-").lower() for i in full_data["flat_type"]]
    full_data["street_name"] = [i.lower() for i in full_data["street_name"]]

    # Generate geometry column that will be helpful for merging data with auxiliary data later
    full_data["geometry"] = full_data.apply(
        lambda row: Point(row['lng'], row['lat']), axis=1)
    gdf_full = gpd.GeoDataFrame(full_data, geometry='geometry')

    return gdf_full


def preprocess_trainstdata(gdf_full, train_stations, save_results=False):
    '''Preprocess train stations data and merge with train/test data.'''

    # Merge train stations data with full data based on nearest train stations
    train_stations["geometry"] = train_stations.apply(
        lambda row: Point(row['lng'], row['lat']), axis=1)
    gdf_train = gpd.GeoDataFrame(train_stations, geometry='geometry')
    trainstation_w_hdb_data = gdf_full.sjoin_nearest(gdf_train)
    assert trainstation_w_hdb_data.shape[0] == gdf_full.shape[0]

    # Feature Engineer: Distance to mrt (Euclidean distance)
    trainstation_w_hdb_data = trainstation_w_hdb_data.rename(
        columns={"name": "mrt_name"})
    trainstation_w_hdb_data['geometry_left'] = trainstation_w_hdb_data.apply(
        lambda row: Point(row['lng_left'], row['lat_left']), axis=1)
    trainstation_w_hdb_data['geometry_right'] = trainstation_w_hdb_data.apply(
        lambda row: Point(row['lng_right'], row['lat_right']), axis=1)
    trainstation_w_hdb_data["distance_to_mrt"] = gpd.GeoSeries(
        trainstation_w_hdb_data['geometry_left']).distance(gpd.GeoSeries(trainstation_w_hdb_data['geometry_right']))

    # Feature Engineer: Distance to mrt (km)
    trainstation_w_hdb_data["distance_to_mrt_km"] = trainstation_w_hdb_data[[
        "lng_left", "lat_left", "lng_right", "lat_right"]].apply(get_distance_km, axis=1)

    # Feature Engineer: MRT duplicated
    # TODO: Double check mrt that have slash '/' in them
    mrt_duplicated = list(train_stations.loc[train_stations.duplicated(
        subset='name', keep=False), "name"].unique())
    trainstation_w_hdb_data["mrt_counts"] = np.where(
        trainstation_w_hdb_data.mrt_name.isin(mrt_duplicated), 2, 1)

    # Feature Engineer: Distance to mrt in bins
    le = LabelEncoder()
    trainstation_w_hdb_data['distance_to_mrt_bins'] = pd.cut(
        trainstation_w_hdb_data['distance_to_mrt'], 3)
    trainstation_w_hdb_data['distance_to_mrt_bins'] = le.fit_transform(
        trainstation_w_hdb_data['distance_to_mrt_bins'])

    # Feature Engineer: Split out alphanumerical codes into alphabet codes only
    trainstation_w_hdb_data['codes_name'] = trainstation_w_hdb_data['codes'].str[:2]

    if save_results:
        trainstation_w_hdb_data.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, "merge_hdbtrain.csv"), index=False)

    return trainstation_w_hdb_data


def preprocess_popdata(pop_demographics, save_results=False):
    '''Preprocess population demographics data and merge with train/test data'''

    # Create aggregated binnings
    children_groups = ['0-4', '5-9']
    teenager_groups = ['10-14', '15-19']
    youngadult_groups = ['20-24', '25-29']
    adult_groups = ['30-34', '35-39', '40-44',
                    '45-49', '50-54', '55-59', '60-64']
    seniorcit_groups = ['70-74', '75-79', '80-84', '85+']

    pop_demo_updated = pop_demographics.copy()
    pop_demo_updated["group_assign"] = np.where(
        pop_demo_updated.age_group.isin(['0-4', '5-9']), 'children', None)
    pop_demo_updated["group_assign"] = np.where(pop_demo_updated.age_group.isin(
        ['10-14', '15-19']), 'teenager', pop_demo_updated.group_assign)
    pop_demo_updated["group_assign"] = np.where(pop_demo_updated.age_group.isin(
        ['20-24', '25-29']), 'young_adult', pop_demo_updated.group_assign)
    pop_demo_updated["group_assign"] = np.where(pop_demo_updated.age_group.isin(
        ['30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64']), 'adult', pop_demo_updated.group_assign)
    pop_demo_updated["group_assign"] = np.where(pop_demo_updated.age_group.isin(
        ['70-74', '75-79', '80-84', '85+']), 'senior_citizen', pop_demo_updated.group_assign)

    # Create pivot table for the population count based on age_group
    age_count = pd.DataFrame(pop_demo_updated.groupby(
        ["plannin_area", "subzone", "age_group"])["count"].sum()).reset_index()
    age_count = age_count.rename(columns={"plannin_area": "planning_area"})

    # Create pivot table for the population count based on 'group_assign'
    agegroup_count = pd.DataFrame(pop_demo_updated.groupby(
        ["plannin_area", "subzone", "group_assign"])["count"].sum()).reset_index()
    agegroup_count_pivot = pd.DataFrame(agegroup_count.pivot(
        index=['plannin_area', 'subzone'], columns='group_assign', values='count')).reset_index()
    agegroup_count_pivot.columns = ["plannin_area", "subzone"]+[
        i+"_count" for i in agegroup_count_pivot.columns if i not in ["plannin_area", "subzone"]]
    agegroup_count_pivot = agegroup_count_pivot.rename(
        columns={"plannin_area": "planning_area"})

    # Create pivot table for the population count based on 'gender'
    gender_count = pd.DataFrame(pop_demo_updated.groupby(
        ["plannin_area", "subzone", "sex"])["count"].sum()).reset_index()
    gender_count_pivot = pd.DataFrame(gender_count.pivot(
        index=['plannin_area', 'subzone'], columns='sex', values='count')).reset_index()
    gender_count_pivot.columns = ["plannin_area",
                                  "subzone", "female_count", "male_count"]
    gender_count_pivot = gender_count_pivot.rename(
        columns={"plannin_area": "planning_area"})

    # Create pivot table for population count based on only 'planning_area' and 'subzone'
    pop_count = pd.DataFrame(pop_demo_updated.groupby(
        ["plannin_area", "subzone"])["count"].sum()).reset_index()
    pop_count.columns = ["plannin_area", "subzone", "population_count"]
    pop_count = pop_count.rename(columns={"plannin_area": "planning_area"})

    # Save data
    if save_results:
        age_count.to_csv(os.path.join(DATAPREPROCESSING_DIRECTORY,
                         "age_count_pivot_table.csv"), index=False)
        agegroup_count_pivot.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, "agegroup_pivot_table.csv"), index=False)
        gender_count_pivot.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, "gender_count_pivot_table.csv"), index=False)
        pop_count.to_csv(os.path.join(DATAPREPROCESSING_DIRECTORY,
                         "pop_count_pivot_table.csv"), index=False)

    return age_count, agegroup_count_pivot, gender_count_pivot, pop_count


def merge_hdbtrain_popdemo(trainstation_w_hdb_data, agegroup_count_pivot, gender_count_pivot, pop_count, save_results=False):
    '''Merge previously merge hdbtrain data with population demographics data'''

    # Merge data
    merge_data = pd.merge(trainstation_w_hdb_data, pop_count.rename(columns={
                          "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")
    merge_data = pd.merge(merge_data, agegroup_count_pivot.rename(columns={
                          "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")
    merge_data = pd.merge(merge_data, gender_count_pivot.rename(columns={
                          "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")

    # Feature Engineer: Male Female Ratio
    le = LabelEncoder()
    merge_data["male_female_ratio"] = merge_data["male_count"] / \
        merge_data["female_count"]
    merge_data['male_female_ratio_bins'] = pd.cut(
        merge_data['male_female_ratio'], 3)
    merge_data['male_female_ratio_bins'] = le.fit_transform(
        merge_data['male_female_ratio_bins'])

    # Feature Engineer: Adult Children Ratio
    # TODO: Checkout the infinity, instead of removing them, process them
    merge_data["adult_children_ratio"] = merge_data[["young_adult_count", "adult_count", "senior_citizen_count"]].sum(
        axis=1)/merge_data[["children_count", "teenager_count"]].sum(axis=1)
    # merge_data = merge_data.loc[merge_data.adult_children_ratio!=np.inf, :]
    # merge_data['adult_children_ratio_bins'] = pd.cut(merge_data['adult_children_ratio'], 2)
    # merge_data['adult_children_ratio_bins'] = le.fit_transform(merge_data['adult_children_ratio_bins'])

    # Feature Engineer: Population Bins
    le = LabelEncoder()
    merge_data['population_bins'] = pd.cut(merge_data['population_count'], 3)
    merge_data['population_bins'] = le.fit_transform(
        merge_data['population_bins'])

    # Save data
    if save_results:
        merge_data.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, "merge_hdbtrainpop.csv"), index=False)

    return merge_data
