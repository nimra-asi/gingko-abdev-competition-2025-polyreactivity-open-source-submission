# Databricks notebook source
!pip install -r requirements_regression.txt

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# Enables autoreload
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

import pickle
import json
from pathlib import Path
from typing import Dict, List, Literal, Any, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# add path to the utils folder if needed
# path_to_add = ""

# if path_to_add not in sys.path:
#     sys.path.append(path_to_add)

# COMMAND ----------

# import functions to add precomputed features for sequences
from utils.utils import create_feature_space, create_feature_space_test_set

# COMMAND ----------

output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC # read in data

# COMMAND ----------

file_path = "data/GDPa1_v1.2_20250814.csv"
df = pd.read_csv(file_path)
print(df.shape)
df.head(2)

# formatting data frame
df = df.rename(columns=({"SEC %Monomer": "SEC_perc_Monomer",
                         "AC-SINS_pH6.0": "AC_SINS_pH6.0",
                         "AC-SINS_pH7.4": "AC_SINS_pH7.4",
                         "vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

df.columns = [x.lower() for x in df.columns]
df = df.dropna(subset=["pr_cho"]) # remove missing values
df.head(2)

# COMMAND ----------

df.isna().sum()
df.shape

# COMMAND ----------

print(df.shape)
df['pr_cho'].isna().sum()

# COMMAND ----------

# examine the distribution of pr_cho values
df["pr_cho"].describe()

# COMMAND ----------

print("removing duplicated antibody lucatumumab") 
# the rationale behind this removal / exploration for duplication is in the notebook: pr_cho_data_exploration.ipynb

df = df.loc[df["antibody_name"]!="lucatumumab"].reset_index(drop=True)
print(df.shape)
df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # read in the features

# COMMAND ----------

features_list = ["esm2", "esm2_cls", "ablang2", "moe", "tap"]
ds = create_feature_space(df, features_list)
print(ds.shape)
ds.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # perform feature selection

# COMMAND ----------

import mlflow
mlflow.autolog(disable=True)

# COMMAND ----------

ds.head(2)

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Custom Spearman correlation scorer
def spearman_scorer(y_true, y_pred):
    """Spearman correlation coefficient scorer"""
    corr, _ = spearmanr(y_true, y_pred)
    return corr

spearman_score = make_scorer(spearman_scorer, greater_is_better=True)

# Define your variables
target_col = 'pr_cho' 
group_col = 'hierarchical_cluster_igg_isotype_stratified_fold'
non_feature_cols = ['antibody_id', 'antibody_name', 'titer', 'purity', 'sec_perc_monomer',
       'smac', 'hic', 'hac', 'pr_cho', 'pr_ova', 'ac_sins_ph6.0',
       'ac_sins_ph7.4', 'tonset', 'tm1', 'tm2', 'hc_subtype', 'lc_subtype',
       'highest_clinical_trial_asof_feb2025', 'est_status_asof_feb2025', 'vh',
       'hc_protein_sequence', 'hc_dna_sequence', 'vl', 'lc_protein_sequence',
       'lc_dna_sequence', 'hierarchical_cluster_fold', 'random_fold',
       'hierarchical_cluster_igg_isotype_stratified_fold', 'light_aligned_aho',
       'heavy_aligned_aho']
feature_cols = [col for col in ds.columns if col not in non_feature_cols]

# Prepare data
X = ds[feature_cols].values
y = ds[target_col].values
groups = ds[group_col].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -- remove invariant features
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X_scaled)

# Get boolean mask of kept features
mask = selector.get_support()

# Get kept and dropped column names
kept_features = [feature_cols[i] for i, selected in enumerate(mask) if selected]
dropped_features_var = [feature_cols[i] for i, selected in enumerate(mask) if not selected]

print(f"Dropped features variance threshold: {dropped_features_var}")

# ------------ remove highly correlated features

def remove_correlated_features(X, columns, threshold=0.9):
    corr_matrix = np.corrcoef(X.T)
    upper = np.triu(np.abs(corr_matrix), k=1)
    to_drop = [i for i in range(upper.shape[0]) if any(upper[:, i] > threshold)]
    
    dropped_names = [columns[i] for i in to_drop]
    remaining_cols = [col for i, col in enumerate(columns) if i not in to_drop]
    
    return np.delete(X, to_drop, axis=1), dropped_names, remaining_cols

# Usage
X_uncorr, dropped_features_corr, remaining_names = remove_correlated_features(X_scaled, feature_cols,threshold=0.9)
print(f"Dropped features corr: {dropped_features_corr}")

to_drop = dropped_features_var + dropped_features_corr

dss = ds.drop(columns=to_drop)

feature_cols = [col for col in dss.columns if col not in non_feature_cols]

# Prepare data
X = dss[feature_cols].values
y = dss[target_col].values
groups = dss[group_col].values

# Standardize features (important for Ridge)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Ridge': Ridge(alpha=1.0, random_state=42),
    }

# Group CV splitter
cv = GroupKFold(n_splits=ds[group_col].nunique())

# Store results
results = {}

# Run RFECV for each model
print("Running RFE with cross-validation")
print("="*60)

for model_name, model in models.items():
    print(f"\nProcessing {model_name}...")
    
    # RFECV with custom scorer and group CV
    rfecv = RFECV(
        estimator=model,
        step=1,  # Remove 1 feature at a time (can increase for speed)
        cv=cv,
        scoring=spearman_score,
        n_jobs=-1,
        verbose=0
    )
    
    # Fit RFECV
    rfecv.fit(X_scaled, y, groups=groups)
    
    # Store results
    results[model_name] = {
        'rfecv': rfecv,
        'n_features': rfecv.n_features_,
        'support': rfecv.support_,
        'ranking': rfecv.ranking_,
        'cv_scores': rfecv.cv_results_['mean_test_score'],
        'selected_features': [feature_cols[i] for i, selected in enumerate(rfecv.support_) if selected]
    }
    
    print(f"  Optimal number of features: {rfecv.n_features_}")
    print(f"  Best CV Spearman correlation: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    print(f"  Selected {len(results[model_name]['selected_features'])} features")


# COMMAND ----------

# Examine the selected features
selected_features = results['Ridge']['selected_features']
print(len(selected_features))
selected_features

# COMMAND ----------

# save the selected features
sel_df = pd.DataFrame({'selected_features_rfe_ridge_preprocessing': selected_features})
print(sel_df.shape)
sel_df.to_csv('outputs/pr_cho_selected_features_rfe_ridge_preprocessing.csv', index=False)
sel_df.head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC # run regression pipeline

# COMMAND ----------

s = pd.read_csv("outputs/pr_cho_selected_features_rfe_ridge_preprocessing.csv")
print(s.shape)
s.head()

selected_features = s['selected_features_rfe_ridge_preprocessing'].to_list()
print(s)

# COMMAND ----------

# import regressor to test different regression models
from utils.antibody_regressor_multi_embedding import AntibodyRegressorMultiEmbedding

# COMMAND ----------

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA # Import for Dimensionality Reduction
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# TabPFN
try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# COMMAND ----------

# hugging face login for tabpfn
import huggingface_hub
huggingface_hub.login()

# COMMAND ----------

cols = ['antibody_id', 'antibody_name', 'titer', 'purity', 'sec_perc_monomer',
       'smac', 'hic', 'hac', 'pr_cho', 'pr_ova', 'ac_sins_ph6.0',
       'ac_sins_ph7.4', 'tonset', 'tm1', 'tm2', 'hc_subtype', 'lc_subtype',
       'highest_clinical_trial_asof_feb2025', 'est_status_asof_feb2025', 'vh',
       'hc_protein_sequence', 'hc_dna_sequence', 'vl', 'lc_protein_sequence',
       'lc_dna_sequence', 'hierarchical_cluster_fold', 'random_fold',
       'hierarchical_cluster_igg_isotype_stratified_fold', 'light_aligned_aho',
       'heavy_aligned_aho'] + selected_features

print(len(cols))

data = ds[cols].copy()

# COMMAND ----------

# Define Parameters
group_col = 'hierarchical_cluster_igg_isotype_stratified_fold'
target_name = 'pr_cho'
feature_type = 'esm2_esm2_cls_ablang2_moe_tap_imputation'
scaler = 'standard'
dr = False              
dr_n_components = None

output_dir = "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_folder = output_dir

print(f"Data shape: {data.shape}")
print(f"Output directory: {output_dir}")
print("-" * 30)

# COMMAND ----------

PARAMS = {
    'df': data,
    'output_folder': output_dir,
    'group_col': group_col,     
    'target_name': target_name,
    'feature_type': feature_type,        
    'scaler': scaler,        
    'dr': dr,      
    'dr_n_components': dr_n_components,           
    }
    
try:
    regressor = AntibodyRegressorMultiEmbedding(**PARAMS)
    results = regressor.run_regression_pipeline()
        
    # Display a snippet of the averaged results
    results_df = pd.DataFrame(results)
except Exception as e:
    print(f"Error occurred: {e}")

# COMMAND ----------

print("\n--- Summary of Grouped CV Results (Averaged Across Folds) ---")
summary = results_df.groupby('model')[['spearman', 'rmse', 'mae']].mean().sort_values(by='spearman', ascending=False)
print(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # hyperparameter tuning for best model

# COMMAND ----------

# decided to move further with Ridge Regression for fine tuning

# COMMAND ----------

file_path = "data/GDPa1_v1.2_20250814.csv"
df = pd.read_csv(file_path)
print(df.shape)

df = df.rename(columns=({"SEC %Monomer": "SEC_perc_Monomer",
                         "AC-SINS_pH6.0": "AC_SINS_pH6.0",
                         "AC-SINS_pH7.4": "AC_SINS_pH7.4",
                         "vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

df.columns = [x.lower() for x in df.columns]
df = df.dropna(subset=["pr_cho"])
print(df.shape)
print("removing duplicated antibody lucatumumab") 
df = df.loc[df["antibody_name"]!="lucatumumab"].reset_index(drop=True)
print(df.shape)
df.head(2)

# COMMAND ----------

features_list = ["esm2", "esm2_cls", "ablang2", "moe", "tap"]
ds = create_feature_space(df, features_list)
print(ds.shape)
ds.head(2)

# COMMAND ----------

file_path = "outputs/pr_cho_selected_features_rfe_ridge_preprocessing.csv"
s = pd.read_csv(file_path)
print(s.shape)
s.head()

selected_features = s['selected_features_rfe_ridge_preprocessing'].to_list()

# COMMAND ----------

selected_features

# COMMAND ----------

X = ds[selected_features]
y = ds[target_name]
groups = ds[group_col]

# COMMAND ----------

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
import numpy as np

# Split your data first
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42
)

# Define custom Spearman correlation scorer
def spearman_scorer(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

spearman_score = make_scorer(spearman_scorer)

# Create a pipeline with scaling and Ridge regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Define hyperparameter grid
param_grid = {
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Set up GroupKFold
group_kfold = GroupKFold(n_splits=5)

# Set up GridSearchCV with GroupKFold
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=group_kfold,
    scoring=spearman_score,
    n_jobs=-1
)

# Fit the model (groups should be an array indicating group membership for each sample)
grid_search.fit(X_train, y_train, groups=groups_train)


# Get best parameters and score
print(f"Best alpha: {grid_search.best_params_['ridge__alpha']}")
print(f"Best Spearman r: {grid_search.best_score_:.4f}")


# Use the best model
best_model = grid_search.best_estimator_

# Evaluate on held-out test set
y_pred = best_model.predict(X_test)
test_spearman = spearmanr(y_test, y_pred)[0]
print(f"Spearman r on test set: {test_spearman:.4f}")

# COMMAND ----------

best_params = grid_search.best_params_
best_params

# COMMAND ----------

# MAGIC %md
# MAGIC # get final submission files

# COMMAND ----------

# import class for final evaluation
from utils.get_final_results_ridge import FinalModelEvaluator

# COMMAND ----------

from sklearn.linear_model import Ridge
target_col = "pr_cho"
group_col = "hierarchical_cluster_igg_isotype_stratified_fold"

cols = ['antibody_name', 'antibody_id', 'vh', 'vl', target_col, group_col] + selected_features
print(cols)

input_df = ds[cols].copy()
print(input_df.shape)

best_params_ridge = {'alpha': 10}


# COMMAND ----------

final_evaluator = FinalModelEvaluator(
    output_folder='outputs',
    input_df=input_df,
    group_column=group_col,
    value_column=target_col,
    model_class=Ridge,
    model_params=best_params_ridge,
    n_splits=5,
    dr_method=None,
    n_components=None 
)

final_submission_df, metrics_df = final_evaluator.run_submission_pipeline()

# COMMAND ----------

# MAGIC %md
# MAGIC ## get final fitted model

# COMMAND ----------

X_train = ds[selected_features].copy()
y_train = ds[target_col].copy()

# COMMAND ----------

import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# Define Your Output Directory and Constants ---
MODEL_DIR = Path('outputs/pr_cho_final_fitted_models')
MODEL_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist


# Fit and Save StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("StandardScaler saved to scaler.pkl")

# --- 4. Train and Save Final Model ---
# BEST_PARAMS holds the optimal dictionary for the final model
final_model = Ridge(**best_params_ridge)
final_model.fit(X_train_scaled, y_train)

with open(MODEL_DIR / 'final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
print("Final model saved to final_model.pkl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## get the predictions for the missing cols

# COMMAND ----------

file_path = "data/GDPa1_v1.2_20250814.csv"
df = pd.read_csv(file_path)
print(df.shape)
df.head(2)

df = df.rename(columns=({"SEC %Monomer": "SEC_perc_Monomer",
                         "AC-SINS_pH6.0": "AC_SINS_pH6.0",
                         "AC-SINS_pH7.4": "AC_SINS_pH7.4",
                         "vh_protein_sequence": "vh",
                         "vl_protein_sequence": "vl"}))

# print(df.isna().sum())
df.columns = [x.lower() for x in df.columns]

df.head(2)

# COMMAND ----------

test = df.loc[df["antibody_name"]=="lucatumumab"]
test

# COMMAND ----------

# getting a prediction for lucatumumab as was not in the training set
missing_df = df.loc[((df[target_col].isna()) | (df['antibody_name']=='lucatumumab'))].reset_index(drop=True)
print(missing_df.shape)
missing_df.head(2)


# COMMAND ----------

# --- 1. Define Constants and Paths (ADJUST THESE) ---
MODEL_DIR = Path('outputs/pr_cho_final_fitted_models') # Directory where fitted models are stored

# Load the fitted transformers and the final model
try:
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing fitted model file. Did you save 'scaler.pkl' and 'final_model.pkl' to {MODEL_DIR}? Error: {e}")

# COMMAND ----------

ds = create_feature_space(missing_df, features_list)
print(ds.shape)

# COMMAND ----------

# Get the list of PLM features
X_test_raw = ds[selected_features].values


# Apply StandardScaler
X_test_scaled = scaler.transform(X_test_raw)


# Generate predictions
testset_y = final_model.predict(X_test_scaled)

# Create and Save Submission CSV
missing_preds = missing_df[['antibody_name', 'vh', 'vl', group_col]].copy().rename(columns=({'vh': 'vh_protein_sequence', 'vl': 'vl_protein_sequence'}))

# Add the predicted pr cho column
missing_preds[target_col] = testset_y

print(missing_preds.shape)

# COMMAND ----------

# read in the previously generated predictions and concatenate the two
dt = pd.read_csv("outputs/pr_cho_Ridge_OOF_submission.csv")
print(dt.shape)
dt.head(2)

# COMMAND ----------

final_submission = pd.concat([dt, missing_preds], axis=0)
print(final_submission.shape)
print(final_submission['antibody_name'].nunique())
final_submission.head(2)

# COMMAND ----------

# checking pred for lucutumumab
test = final_submission.loc[final_submission['antibody_name']=='lucatumumab']
test

# COMMAND ----------

final_submission.to_csv("outputs/pr_cho_11172025_OOF_submission.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## get predictions for test set

# COMMAND ----------

file_path = "data/heldout-set-sequences.csv"
test_df = pd.read_csv(file_path)

print(test_df.shape)
test_df.head(2)

# COMMAND ----------

# Define Constants and Paths
MODEL_DIR = Path('outputs/pr_cho_final_fitted_models')
OUTPUT_PATH = 'outputs/pr_cho_11172025_testset_submission.csv'

# Load the fitted scaler and the final model
try:
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'final_model.pkl', 'rb') as f:
        final_model = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing fitted model file. Did you save 'scaler.pkl' and 'final_model.pkl' to {MODEL_DIR}? Error: {e}")


# COMMAND ----------

from utils.utils import create_feature_space_test_set

# COMMAND ----------

features_list
ds = create_feature_space_test_set(test_df, features_list)
print(ds.shape)
ds.head(2)

# COMMAND ----------

selected_features

# COMMAND ----------

X_test_raw = ds[selected_features].values
X_test_scaled = scaler.transform(X_test_raw)

testset_y = final_model.predict(X_test_scaled)

# Create and save submission CSV
testset_submission = ds[['antibody_name', 'vh_protein_sequence', 'vl_protein_sequence']].copy()

# Add the predictions
testset_submission[target_col.upper()] = testset_y

# Save the file
testset_submission.to_csv(OUTPUT_PATH, index=False)

print(f"Final predictions generated and saved to: {OUTPUT_PATH}")
print(f"Submission shape: {testset_submission.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### check that pipeline above recreated orginal submitted results

# COMMAND ----------

testset_submission.head(2)

# COMMAND ----------

# read in results submitted for competition
file_path = "submitted_11172025_pr_cho/pr_cho_11172025_testset_submission.csv"
submitted = pd.read_csv(file_path)

print(submitted.shape)
submitted.head(2)

# COMMAND ----------

combined = testset_submission.merge(submitted[['antibody_name', 'pr_cho']], on='antibody_name', how='left')
print(combined.isna().sum())

# COMMAND ----------

combined['comparison'] = combined['PR_CHO'].round(4) == combined['pr_cho'].round(4)

print(combined['comparison'].sum())
