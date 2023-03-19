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


def get_distance_km(row):
    return geopy.distance.geodesic((row['lat_left'], row['lng_left']), (row['lat_right'], row['lng_right'])).km


def preprocess_traintest_data(data):
    '''Preprocess train and test data and merge them. The final output is a geopandas dataframe.'''

    # Preprocess train and test data
    data_renamed = data.rename(columns={"longitude": "lng", "latitude": "lat"})
    data_renamed["flat_type"] = [
        i.replace(" ", "-").lower() for i in data_renamed["flat_type"]]
    data_renamed["street_name"] = [i.lower()
                                   for i in data_renamed["street_name"]]

    # Generate geometry column that will be helpful for merging data with auxiliary data later
    data_renamed["geometry"] = data_renamed.apply(
        lambda row: Point(row['lng'], row['lat']), axis=1)
    geopandas_df = gpd.GeoDataFrame(data_renamed, geometry='geometry')

    # Drop duplicated rows
    geopandas_df_drop_duplicated = geopandas_df.drop_duplicates(
        keep="first", inplace=False)

    return geopandas_df_drop_duplicated


def preprocess_trainstdata(gdf_df, train_stations, train=False, save_results=False, fname="train"):
    '''Preprocess train stations data and merge with train/test data.'''

    # Merge train stations data with full data based on nearest train stations
    train_stations["geometry"] = train_stations.apply(
        lambda row: Point(row['lng'], row['lat']), axis=1)
    gdf_train = gpd.GeoDataFrame(train_stations, geometry='geometry')
    trainstation_w_hdb_data = gdf_df.sjoin_nearest(gdf_train)
    assert trainstation_w_hdb_data.shape[0] == gdf_df.shape[
        0], "Merged data does not have the same shape as input data. Please check."

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

    # Feature Engineer: MRT-LRT Links
    mrt_lrt_links_list = list(train_stations.loc[train_stations.duplicated(
        subset='name', keep=False), "name"].unique())
    trainstation_w_hdb_data["mrt_lrt_links"] = np.where(
        trainstation_w_hdb_data.mrt_name.isin(mrt_lrt_links_list), 1, 0)

    # Feature Engineer: MRT Interchange count, NS16/NS16 - not counted as interchange
    mrt_interchange_list = [
        i for i in train_stations.codes if ('/' in i) & ("NS16" not in i)]
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


def calculateCount(df, dist, k, distRange, inRangeCount):
    '''Calculate nearest auxiliary building in close proximity range.'''

    df[inRangeCount] = 0
    for i in range(k):
        df[inRangeCount] = np.where(
            dist[:, i-1] < distRange, df[inRangeCount] + 1, df[inRangeCount])

    return df


def haversine(p1, p2):
    '''Calculate haversein distances to be similar to represent Earth's distance better.'''

    # Convert coordinates to radians
    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c  # Earth's radius is approximately 6,367 km

    return km


def ckdnearest(gdA, gdB, k):
    '''Find nearest buildings and merge dataframe.'''

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))

    # Compute distance matrix using the custom distance metric
    distances = np.zeros((len(nA), len(nB)))
    for i, p1 in enumerate(nA):
        for j, p2 in enumerate(nB):
            distances[i, j] = haversine(p1, p2)
    max_distance = 1000  # kilometers
    distances[distances > max_distance] = np.inf
    btree = cKDTree(nB)
    btree.data_distances = distances

    kNearest = range(1, k+1)

    dist, idx = btree.query(nA, k=kNearest)
    gdB_nearest = gdB.iloc[idx[:, 0]].drop(
        columns={"geometry"}).reset_index(drop=True)

    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist[:, 0], name='dist'),
        ],
        axis=1)

    return gdf, dist


def generate_pandasprofiling(data, important_cols, save_results=False, fname="train"):
    '''Generate pandas profiling for the data.'''

    hdbtrainpop_profile = ProfileReport(data[important_cols])

    if save_results:
        hdbtrainpop_profile.to_file(os.path.join(
            DATAPREPROCESSING_DIRECTORY, f"hdbtrainstpop_{fname}.html"))

    return hdbtrainpop_profile


def chisquared_calc(data, categorical_column_lst):
    '''Run chi-squared correlation for categorical data'''

    chi2_list = []
    pval_list = []
    for category_col in categorical_column_lst:
        contingency_table = pd.crosstab(
            data[category_col], data['resale_price'])
        chi2, pval, _, _ = chi2_contingency(contingency_table)
        chi2_list.append(chi2)
        pval_list.append(pval)

    final_chi2 = pd.DataFrame(zip(categorical_column_lst, chi2_list, pval_list), columns=[
        "Categorical_Column", "Chi2_Statistic", "Pvalue"])

    return final_chi2


def anova_calc(data, categorical_column_lst):
    '''Run anove correlation for categorical data'''

    anova_list = []
    pval_list = []
    col_list = []

    for col in categorical_column_lst:
        if data[col].nunique() > 1:
            model = sm.formula.ols(
                'resale_price ~ {}'.format(col), data=data).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            f_stat = anova_table['F'][0]
            p_val = anova_table['PR(>F)'][0]
            anova_list.append(f_stat)
            pval_list.append(p_val)
            col_list.append(col)
        else:
            print(f"{col} does not have enough levels to compute ANOVA.")

    final_anova = pd.DataFrame(zip(col_list, anova_list, pval_list), columns=[
        "Categorical_Column", "F_statistics", "Pvalue"])

    return final_anova
