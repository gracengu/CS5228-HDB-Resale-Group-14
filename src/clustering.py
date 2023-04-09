# HDB Resale Price Prediction
# Author: Grace Ngu Sook Ern, Hu Dong Yue, Cao Sheng, Guo Wei

from preprocessing_merge import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import scipy as sp


# Selected columns from mrt and population data
selected_columns = ['floor_area_sqm', 'distance_to_mrt_km', 'population_count', 'adult_count', 'children_count',
                    'senior_citizen_count', 'teenager_count', 'female_count', 'male_count']


def principal_component_analysis(data, selected_columns, plot=False):
    '''Generate 1st and 2nd principal components of data based on selected columns.'''

    X = data.loc[:, selected_columns].values
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents, columns=[
                               'principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, data[['resale_price']]], axis=1)

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        ax.scatter(finalDf['principal component 1'],
                   finalDf['principal component 2'])
        ax.grid()

    return finalDf


def kmeans_clustering(data, k=2, plot=False):
    '''Kmeans clustering on principal component 1 and principal component 2. Not selected due to poor clustering outcome, \
        but is used to get representative data sample for dbscan and hierarchical clustering.'''

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data[['principal component 1', 'principal component 2']])
    labels = kmeans.labels_
    data["Kmeans_cluster"] = labels

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        targets = [0, 1]
        colors = ['r', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = data['Kmeans_cluster'] == target
            ax.scatter(data.loc[indicesToKeep, 'principal component 1'],
                       data.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()

    return data, kmeans


def sample_data(data, perc_sample=0.6):
    '''Use Kmeans cluster as proxy for sampling data.'''

    sampled_data = data.groupby('Kmeans_cluster').sample(
        n=int(data.shape[0]*perc_sample))

    return sampled_data


def hierarchical_clustering(data, k=2, plot=False):
    '''Hierarchical clustering on principal component 1 and principal component 2. Not selected due to poor clustering outcome.'''

    hcluster = AgglomerativeClustering(n_clusters=k)
    hcluster.fit(data[['principal component 1', 'principal component 2']])
    labels = hcluster.labels_
    data["Hierarchical_cluster"] = labels

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        targets = [0, 1]
        colors = ['r', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = data['Hierarchical_cluster'] == target
            ax.scatter(data.loc[indicesToKeep, 'principal component 1'],
                       data.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()

    return data, hcluster


def dbscan_clustering(data, eps=0.5, min_samples=10, plot=False):
    '''DBSCAN clustering on principal component 1 and principal component 2. '''

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data[['principal component 1', 'principal component 2']])
    labels = dbscan.labels_
    data["DBSCAN_cluster"] = labels

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        targets = [0, 1]
        colors = ['r', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = data['DBSCAN_cluster'] == target
            ax.scatter(data.loc[indicesToKeep, 'principal component 1'],
                       data.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()

    return data, dbscan


def dbscan_predict(dbscan_model, data, metric=sp.spatial.distance.cosine, plot=True):
    '''Use dbscan to perform prediction'''

    # Iterate all input samples for a label
    nr_samples = data.shape[0]
    cluster_new = np.ones(shape=nr_samples, dtype=int) * -1
    data = data[['principal component 1', 'principal component 2']]

    for i in range(nr_samples):
        diff = dbscan_model.components_ - list(data.iloc[i, :])
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)
        if dist[shortest_dist_idx] < dbscan_model.eps:
            cluster_new[i] = dbscan_model.labels_[
                dbscan_model.core_sample_indices_[shortest_dist_idx]]

    data['DBSCAN_cluster'] = cluster_new

    if plot:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)

        targets = [0, 1]
        colors = ['r', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = data['DBSCAN_cluster'] == target
            ax.scatter(data.loc[indicesToKeep, 'principal component 1'],
                       data.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()

    return data


def compute_clustering(data, selected_columns):
    '''Final clustering with DBSCAN.'''

    # Generate principal component 1 & 2
    pca_data = principal_component_analysis(data, selected_columns, plot=False)

    # Use k-means clusters as proxy for sampling data
    kmeans_cluster_data, kmeans_model = kmeans_clustering(
        pca_data, k=2, plot=False)
    sampled_data = sample_data(kmeans_cluster_data, perc_sample=0.1)

    # Train dbscan model on sample data
    _, dbscan_model = dbscan_clustering(
        sampled_data, eps=1.5, min_samples=50, plot=False)

    # Run dbscan prediction on full data
    data = dbscan_predict(dbscan_model, pca_data, plot=True)

    return data
