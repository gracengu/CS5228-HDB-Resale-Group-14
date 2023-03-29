# HDB Resale Price Prediction
# Author: Grace Ngu Sook Ern, Hu Dong Yue, Cao Sheng, Guo Wei
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import plotly.offline as pyo

import os
import pandas as pd
from pandas import DataFrame
from pandas_profiling import ProfileReport
import numpy as np

import sklearn
from sklearn.preprocessing import LabelEncoder

from scipy.spatial import cKDTree, KDTree
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm

from shapely.geometry import Point
import geopandas as gpd
import pygeos
import geopy
from geopy import distance

import preprocessing_train_test

import warnings
warnings.filterwarnings("ignore")

# Global Variables
pio.renderers.default = 'notebook_connected'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DATA_DIRECTORY = "../data/"
DATAPREPROCESSING_DIRECTORY = "../results/data_preprocessing/"
DATAPREPROCESSINGIMG_DIRECTORY = "../img/data_preprocessing/"
MODELPREPROCESSING_DIRECTORY = "../model/data_preprocessing/"

# Helper functions

def df_to_gdf(data_df):
    data_gdf = gpd.GeoDataFrame(data_df, geometry=gpd.points_from_xy(data_df.lng, data_df.lat), crs='EPSG:4326')
    data_gdf = data_gdf.to_crs(epsg=3414)
    return data_gdf

def merge_auxiliary_data(data: DataFrame, commerical_df: str, market_df, population_df, primary_School_df, secondary_School_df, mall_df, train_station_df) -> DataFrame:
    '''Merge all the auxiliary dataset into train or test dataset'''
    # transfer dataframe to geo_dataframe
    data_gdf = df_to_gdf(data.rename(columns={"longitude": "lng", "latitude": "lat"}))

    data_gdf = merge_trainST_and_population(data_gdf, train_station_df, population_df)

    data_gdf = merge_commerical_and_market(data_gdf, commerical_df, market_df)

    data_gdf = merge_school_and_mall(data_gdf, primary_School_df, secondary_School_df, mall_df)

    return data_gdf


def merge_trainST_and_population(data_gdf, train_station_df, population_df):
    '''preprocess and merge dataset: train_station_df and population_df into train or test dataset'''
    hdbtrain_data = preprocess_trainstdata(data_gdf, train_station_df, train=True, save_results=False)
    age_count, agegroup_count_pivot, gender_count_pivot, pop_count = preprocess_popdata(population_df, save_results=False)
    data_gdf = merge_hdbtrain_popdemo(hdbtrain_data, agegroup_count_pivot, gender_count_pivot, pop_count, train=True, save_results=True)
    
    return data_gdf


def merge_commerical_and_market(data_gdf, commerical_df, market_df):
    '''preprocess and merge dataset: commerical_df and market_df into train or test dataset'''
    commerical_gdf = df_to_gdf(commerical_df)
    market_gdf = df_to_gdf(market_df)

    data_gdf = merge_commerical(data_gdf, commerical_gdf)
    data_gdf = merge_marketa(data_gdf, market_gdf) 

    return data_gdf

def merge_school_and_mall(data_gdf, primary_School_df, secondary_School_df, mall_df):
    '''preprocess and merge dataset: primary_School_df, secondary_School_df and mall_df into train or test dataset'''
    primary_School_gdf = df_to_gdf(primary_School_df)
    secondary_School_gdf = df_to_gdf(secondary_School_df)
    mall_gdf = df_to_gdf(mall_df)

    data_gdf = merge_primary_School(data_gdf, primary_School_gdf)
    data_gdf = merge_secondary_School(data_gdf, secondary_School_gdf)
    data_gdf = merge_mall_School(data_gdf, mall_gdf)
    
    return data_gdf

################｜ for merge_trainST_and_population ｜##################

def get_distance_km(row):
    return geopy.distance.geodesic((row['lat_left'], row['lng_left']), (row['lat_right'], row['lng_right'])).km

def preprocess_trainstdata(data_gdf, train_station_df, train=False, save_results=False, fname="train"):
    '''Preprocess train stations data and merge with train/test data.'''
    # gdf_df, train_stations
    # Merge train stations data with full data based on nearest train stations
    train_station_gdf = df_to_gdf(train_station_df)
    # train_stations["geometry"] = train_stations.apply(
    #     lambda row: Point(row['lng'], row['lat']), axis=1)
    # gdf_train = gpd.GeoDataFrame(train_stations, geometry='geometry')
    trainstation_w_hdb_data = data_gdf.sjoin_nearest(train_station_gdf)
    assert trainstation_w_hdb_data.shape[0] == data_gdf.shape[
        0], "Merged data does not have the same shape as input data. Please check."

    # # Feature Engineer: Distance to mrt (Euclidean distance)
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

    # Feature Engineer: MRT-LRT Links
    mrt_lrt_links_list = list(train_station_df.loc[train_station_df.duplicated(
        subset='name', keep=False), "name"].unique())
    trainstation_w_hdb_data["mrt_lrt_links"] = np.where(
        trainstation_w_hdb_data.mrt_name.isin(mrt_lrt_links_list), 1, 0)

    # Feature Engineer: MRT Interchange count, NS16/NS16 - not counted as interchange
    mrt_interchange_list = [
        i for i in train_station_df.codes if ('/' in i) & ("NS16" not in i)]
    trainstation_w_hdb_data["mrt_interchange_flag"] = np.where(
        trainstation_w_hdb_data.codes.isin(mrt_interchange_list), 1, 0)
    trainstation_w_hdb_data["mrt_interchange_count"] = [
        len(set(i.split("/")))-1 for i in trainstation_w_hdb_data["codes"]]

    # Feature Engineer: Generate alphabetical train codes name from the alphanumeric codes name
    # TODO: Taking only the first code, check with team for opinion
    trainstation_w_hdb_data['codes_name'] = trainstation_w_hdb_data['codes'].str[:2]

    # Feature Engineer: Distance to mrt in bins
    trainstation_w_hdb_data['distance_to_mrt_bins'] = pd.cut(
        trainstation_w_hdb_data['distance_to_mrt'], 3)
    if train:
        le = LabelEncoder()
        trainstation_w_hdb_data['distance_to_mrt_bins'] = le.fit_transform(
            trainstation_w_hdb_data['distance_to_mrt_bins'])
        np.save(os.path.join(MODELPREPROCESSING_DIRECTORY,
                'le_distance_mrt_bins.npy'), le.classes_)

    else:
        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(MODELPREPROCESSING_DIRECTORY,
                                           'le_distance_mrt_bins.npy'), allow_pickle=True)
        trainstation_w_hdb_data['distance_to_mrt_bins'] = le.transform(
            trainstation_w_hdb_data['distance_to_mrt_bins'])

    if save_results:
        trainstation_w_hdb_data.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, f"merge_hdb_{fname}.csv"), index=False)

    return trainstation_w_hdb_data


def preprocess_popdata(pop_demographics, save_results=False, fname="train"):
    '''Preprocess population demographics data and merge with both train and test data'''

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
                         f"age_count_pivot_table_{fname}.csv"), index=False)
        agegroup_count_pivot.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, f"agegroup_pivot_table.csv"), index=False)
        gender_count_pivot.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, "gender_count_pivot_table.csv"), index=False)
        pop_count.to_csv(os.path.join(DATAPREPROCESSING_DIRECTORY,
                         f"pop_count_pivot_table_{fname}.csv"), index=False)

    return age_count, agegroup_count_pivot, gender_count_pivot, pop_count


def merge_hdbtrain_popdemo(trainstation_w_hdb_data, agegroup_count_pivot, gender_count_pivot, pop_count, train=False, save_results=False, fname="train"):
    '''Merge previously merge hdbtrain data with population demographics data'''

    # Merge data
    merge_data = pd.merge(trainstation_w_hdb_data, pop_count.rename(columns={
        "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")
    merge_data = pd.merge(merge_data, agegroup_count_pivot.rename(columns={
        "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")
    merge_data = pd.merge(merge_data, gender_count_pivot.rename(columns={
        "plannin_area": "planning_area"}), on=["planning_area", "subzone"], how="left")

    # Fill missing data with zero
    columns_fill_missing = ['population_count', 'adult_count', 'children_count', 'senior_citizen_count',
                            'teenager_count', 'young_adult_count', 'female_count', 'male_count']
    merge_data[columns_fill_missing] = merge_data[columns_fill_missing].fillna(
        value=0)

    # Feature Engineer: Male Female Ratio
    merge_data["male_female_ratio"] = merge_data["male_count"] / \
        merge_data["female_count"]
    merge_data.loc[(merge_data.male_female_ratio.isna()) | (
        merge_data.male_female_ratio == np.Inf), "male_female_ratio"] = 0
    merge_data['male_female_ratio_bins'] = pd.cut(
        merge_data['male_female_ratio'], 3)
    if train:
        le = LabelEncoder()
        merge_data['male_female_ratio_bins'] = le.fit_transform(
            merge_data['male_female_ratio_bins'])
        np.save(os.path.join(MODELPREPROCESSING_DIRECTORY,
                'le_male_female_ratio_bins.npy'), le.classes_)
    else:
        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(MODELPREPROCESSING_DIRECTORY,
                                           'le_male_female_ratio_bins.npy'), allow_pickle=True)
        merge_data['male_female_ratio_bins'] = le.transform(
            merge_data['male_female_ratio_bins'])

    # Feature Engineer: Adult Children Ratio
    merge_data["adult_children_ratio"] = merge_data[["young_adult_count", "adult_count", "senior_citizen_count"]].sum(
        axis=1)/merge_data[["children_count", "teenager_count"]].sum(axis=1)
    merge_data.loc[(merge_data.adult_children_ratio.isna()) | (
        merge_data.adult_children_ratio == np.Inf), "adult_children_ratio"] = 0
    merge_data['adult_children_ratio_bins'] = pd.cut(
        merge_data['adult_children_ratio'], 2)
    if train:
        le = LabelEncoder()
        merge_data['adult_children_ratio_bins'] = le.fit_transform(
            merge_data['adult_children_ratio_bins'])
        np.save(os.path.join(MODELPREPROCESSING_DIRECTORY,
                'le_adult_children_ratio_bins.npy'), le.classes_)
    else:
        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(MODELPREPROCESSING_DIRECTORY,
                                           'le_adult_children_ratio_bins.npy'), allow_pickle=True)
        merge_data['adult_children_ratio_bins'] = le.transform(
            merge_data['adult_children_ratio_bins'])

    # Feature Engineer: Population Bins
    merge_data['population_bins'] = pd.cut(merge_data['population_count'], 3)

    if train:
        le = LabelEncoder()
        merge_data['population_bins'] = le.fit_transform(
            merge_data['population_bins'])
        np.save(os.path.join(MODELPREPROCESSING_DIRECTORY,
                'le_population_bins.npy'), le.classes_)
    else:
        le = LabelEncoder()
        le.classes_ = np.load(os.path.join(MODELPREPROCESSING_DIRECTORY,
                                           'le_population_bins.npy'), allow_pickle=True)
        merge_data['population_bins'] = le.transform(
            merge_data['population_bins'])

    # drop columns
    drop_cols = ['lat_left', 'lat_right', 'lng_left', 'lng_right',
                 'geometry_right', 'index_right', 'geometry_left', 'geometry_right']
    selected_cols = [i for i in merge_data.columns if i not in drop_cols]
    merge_data = merge_data[selected_cols]

    # Save data
    if save_results:
        merge_data.to_csv(os.path.join(
            DATAPREPROCESSING_DIRECTORY, f"merge_hdbtrainstpop_{fname}.csv"), index=False)

    return merge_data


################｜ for merge_commerical_and_market ｜##################


def merge_commerical(data_gdf, commerical_gdf):
    data_gdf = ckdnearest1(data_gdf, commerical_gdf, 2000, 5)
    data_gdf = data_gdf.rename(columns={'name': 'name_commerical', 'type': 'type_commerical', 'dist': 'nearest_dist_commerical', 'inRangeCount': 'inRangeCount_commerical'})
    return data_gdf

def merge_marketa(data_gdf, market_gdf):
    data_gdf = ckdnearest1(data_gdf, market_gdf, 1000, 10)
    data_gdf = data_gdf.rename(columns={'name': 'name_market', 'dist': 'nearest_dist_market', 'inRangeCount': 'inRangeCount_market'})
    return data_gdf


def calculateCount (df, dist, k, distRange, inRangeCount):
    df[inRangeCount] = 0
    for i in range(k):
        df[inRangeCount] = np.where(dist[:, i-1]<distRange, df[inRangeCount] + 1, df[inRangeCount])
    
    return df


def ckdnearest1(gdA, gdB, distRange, k):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    
    kNearest = range(1, k+1)
    
    dist, idx = btree.query(nA, k=kNearest)
    gdB_nearest = gdB.iloc[idx[:,0]].drop(columns={'lat', 'lng', "geometry"}).reset_index(drop=True)
    
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist[:,0], name='dist'),
#             pd.Series(dist[:,1], name='dist2'),
#             pd.Series(dist[:,2], name='dist3'),
#             pd.Series(dist[:,3], name='dist4'),
#             pd.Series(dist[:,4], name='dist5'),
        ], 
        axis=1)

    gdf = calculateCount(gdf, dist, k, distRange, 'inRangeCount')
    
    return gdf


################｜ for merge_school_and_mall ｜##################

def ckdnearest2(gdA, gdB, k, distance_threshold):

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    
    kNearest = range(1, k+1)
    
#     dist is the distance array, idx is the index array
    dist, idx = btree.query(nA, kNearest)

    gdB_nearest = gdB.iloc[idx[:,0]].drop(columns="geometry").reset_index(drop=True)
    
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(list(idx), name='closedIndex'),
            pd.Series(dist[:,0], name='dist')
        ], 
        axis=1)

    gdf['inRangeCount'] = 0
    distRange = 1
    for i in range(k):
        gdf['inRangeCount'] = np.where(dist[:, i-1]< distance_threshold, gdf['inRangeCount'] + 1, gdf['inRangeCount'])
        
    gdf['closedIndex'] = gdf.apply(
        lambda row: row['closedIndex'][0: row['inRangeCount']], axis=1)
   
    return gdf


def merge_primary_School(data_gdf, primary_School_gdf):
    primary_School_gdf = primary_School_gdf.reset_index()
    primary_School_gdf['id'] = primary_School_gdf['index']
    primary_School_gdf = primary_School_gdf.drop('index', axis=1)
    data_gdf = ckdnearest2(data_gdf, primary_School_gdf, 10, 0.010)
    data_gdf = data_gdf.rename(columns = {"closedIndex" : "closePrimaryID", 
                                   "inRangeCount" : "nearPrimaryCount", 
                                   "dist": "MinPrimaryDist"})
    
    return data_gdf


def merge_secondary_School(data_gdf, secondary_School_gdf):
    secondary_School_gdf = secondary_School_gdf.reset_index()
    secondary_School_gdf['id'] = secondary_School_gdf['index']
    secondary_School_gdf = secondary_School_gdf.drop('index', axis=1)
    data_gdf = ckdnearest2(data_gdf, secondary_School_gdf, 5, 500)
    data_gdf = data_gdf.rename(columns = {"closedIndex" : "closeSecondID", 
                                   "inRangeCount" : "nearSecondCount", 
                                   "dist": "MinSecDist"})
    return data_gdf


def merge_mall_School(data_gdf, mall_gdf):
    mall_gdf = mall_gdf.reset_index()
    mall_gdf['id'] = mall_gdf['index']
    mall_gdf = mall_gdf.drop('index', axis=1)
    data_gdf = ckdnearest2(data_gdf, mall_gdf, 5, 500)
    data_gdf = data_gdf.rename(columns = {"closedIndex" : "closeShopID", 
                                   "inRangeCount" : "nearShopCount", 
                                   "dist": "MinShopDist"})
    return data_gdf