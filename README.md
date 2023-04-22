# CS5228-HDB-Resale-Group-14

HDB Flat Resale Price Prediction Based on Flat and Location Attributes
CS5228 Group 14 Project Report
Author: CAO Sheng, Grace NGU Sook Ern, GUO Wei, HU Dongyue
Github Repo: https://github.com/gracengu/CS5228-HDB-Resale-Group-14
Kaggle Group Name: Group 14

To reproduce envrionment: 

```{bash}
conda env create -f win_environment.yml
```

## Code Run Process

Code Structure:

![image](./CodeStructure.png)

1. Run preprocessing.ipynb which connects to preprocessing_merge.py and preprocessing_train_test.py
- src/preprocessing.ipynb: This part include preprocessing auxiliary datasets from mall, markets, commercial centres, train station, population demographics, etc... and perform feature engineering for this datasets. Then it merges the dataset to the train-test dataset provided. This includes correlation analysis that checks the correlation of the features to resale price. Here we include pearson correlation and ANOVA for numerical features numerical target, and categorical features numerical target via ANOVA and Chi-square. Besides, we also include a portion of data analysis work in this notebook. 
- src/data_exploration_train_data.ipynb: This part includes the data exploratory work for train test dataset
- src/clustering_kmeans_kmedoids_hiearchical.ipynb: demonstrate our attempt to try out kmeans, kmedoids, hierarchical clustering and DBSCAN for principal components. This connects to clustering.py
- src/pca_clustering_final.ipynb: the final PCA and clustering which is used later on in the model training. 
- src/preprocessing_train_test.py: This part includes preprocessing of train-test dataset, including the auxiliary datasets preprocessed from preprocessing.ipynb


2. Model training 
- src/linearregression_svm.ipynb: Performs linear regression and SVM modelling and k-fold cross validation for hyperparameter tuning
- src/neural_network_mlp_training.ipynb and src/neural_network_mlp_evaluation.ipynb: Performs MLP modelling and k-fold cross validation for hyperparameter tuning
- src/random_forest.ipynb: Performs RandomForest modelling and k-fold cross validation for hyperparameter tuning
- src/xgbmodel.ipynb: Performs XGBoost modelling and k-fold cross validation for hyperparameter tuning


3. Final Ensemble Notebook
- src/final_ensemble_prediction.ipynb: This includes the code script for the final model training used for kaggle submission. This include feature importance plots from our best xgboost and random forest regression as well. 


 