from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

class AntibodyRegressorMultiEmbedding:
    """Main regressor for antibody sequences."""

    # Updated __init__ to accept dr_n_components
    def __init__(self, 
                 df: pd.DataFrame,
                 output_folder: str,
                 group_col: str,
                 target_name: str,
                 feature_type: str,
                 scaler: str, 
                 dr: bool,
                 dr_n_components: Optional[int] = None,
                 ):
        self.df = df
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.group_col = group_col
        self.target_name = target_name
        self.feature_type = feature_type
        self.scaler = scaler
        self.dr = dr
        self.dr_n_components = dr_n_components
        self.results = {}

    def get_models(self) -> Dict[str, Any]:
        """Get dictionary of regression models to train."""
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
            'Huber': HuberRegressor(),
            'RandomForest': RandomForestRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, eval_metric='rmse'),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'KNN': KNeighborsRegressor(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'KernelRidge': KernelRidge(),
            'GaussianProcess': GaussianProcessRegressor(random_state=42),
            'SVR': SVR(),
            'MLP': MLPRegressor(random_state=42, max_iter=1000),
            'PLS': PLSRegression(n_components=10)
        }
        if TABPFN_AVAILABLE:
            # Note: For small datasets, TabPFN might not need explicit DR
            models['TabPFN'] = TabPFNRegressor(device='cpu')

        return models

    def get_features(self):
        """Reads in data and gets the right set of features based on feature type"""
        df = self.df
        non_feature_cols = ['antibody_id', 'antibody_name', 'titer', 'purity', 'sec_perc_monomer',
                            'smac', 'hic', 'hac', 'pr_cho', 'pr_ova', 'ac_sins_ph6.0',
                            'ac_sins_ph7.4', 'tonset', 'tm1', 'tm2', 'hc_subtype', 'lc_subtype',
                            'highest_clinical_trial_asof_feb2025', 'est_status_asof_feb2025', 'vh',
                            'hc_protein_sequence', 'hc_dna_sequence', 'vl', 'lc_protein_sequence',
                            'lc_dna_sequence', 'hierarchical_cluster_fold', 'random_fold',
                            'hierarchical_cluster_igg_isotype_stratified_fold', 'light_aligned_aho',
                            'heavy_aligned_aho']

        feature_cols = [x for x in df.columns if x not in non_feature_cols]

        return df, feature_cols

    def apply_transformations(self, X_train_outer, X_test_outer):
        """Apply the required transformations to X and y such as scaling, PCA etc"""
        
        # Get appropriate scaling
        if self.scaler == "standard":
            scaler = StandardScaler()
        elif self.scaler == "robust":
            scaler = RobustScaler()
        
        pca = PCA(n_components=self.dr_n_components)

        if self.dr:
            pipeline = make_pipeline(scaler, pca)
        else:
            pipeline = make_pipeline(scaler)

        # Fit the entire transformation on the training data
        pipeline.fit(X_train_outer) 

        # Apply the fitted transformations to the training and test data
        X_train_transformed = pipeline.transform(X_train_outer)
        X_test_transformed = pipeline.transform(X_test_outer)

        return X_train_transformed, X_test_transformed

    def run_regression_pipeline(self):
        """Run the complete regression pipeline with grouped cross-validation."""
        print("Starting Antibody Regression Pipeline with K-Fold CV and PCA")
        print("=" * 50)
        
        # Read in the data
        print("Reading in data...")
        df, feature_cols = self.get_features()

        X = df[feature_cols].values 
        y = df[self.target_name].values
        groups = df[self.group_col].values

        models = self.get_models()
        all_results = []

        # Define the cross-validator for ungrouped data
        outer_cv = GroupKFold(n_splits=len(np.unique(groups)))

        # Loop through folds using the KFold object
        print("\nPerforming K-Fold Cross-Validation...")
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
            print(f"\nProcessing Fold: {fold+1}/{outer_cv.get_n_splits()}")

            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]

            test_groups = groups[test_idx]
            unique_test_groups = np.unique(test_groups)
    
            # Define models that require scaling
            models_requiring_scaling = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'Huber',
                                        'KNN', 'KernelRidge', 'GaussianProcess', 'SVR', 'MLP', 'PLS'
                                        ]

            for model_name, model_template in models.items():

                # Apply scaling + PCA for all features
                X_train_for_model, X_test_for_model = self.apply_transformations(X_train_outer, X_test_outer)
        
                try:
                    model_template.fit(X_train_for_model, y_train_outer)
                    y_pred = model_template.predict(X_test_for_model)

                    mae = mean_absolute_error(y_test_outer, y_pred)
                    mse = mean_squared_error(y_test_outer, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test_outer, y_pred)
                    spearman_rho, p_val = spearmanr(y_test_outer, y_pred)

                    result_row = {
                        'model': model_name,
                        'feature_type': self.feature_type,
                        'target_name': self.target_name,
                        'fold': fold + 1,
                        'num_test_groups': len(unique_test_groups),
                        'spearman': spearman_rho,
                        'spearman_pvalue': p_val,
                        'mae': mae,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'num_features': X_train_for_model.shape[1],
                        'dr_n_components': self.dr_n_components
                        }
                    all_results.append(result_row)

                except Exception as e:
                    print(f"    Error training/evaluating {model_name} for fold {fold+1}: {str(e)}")
                    continue

        self.save_results(all_results)
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {self.output_folder}")
        return all_results
    
    def save_results(self, all_results: List[Dict]):
        """Save all results and generate visualizations for regression."""
        print("Saving regression results...")
        results_df = pd.DataFrame(all_results)
        if self.dr:
            results_name =  f"regression_results_{self.feature_type}_pca{self.dr_n_components}_{self.scaler}scaler_groupedcv.csv"
        else:
            results_name =  f"regression_results_{self.feature_type}_no_pca_{self.scaler}scaler_groupedcv.csv"
        results_path = self.output_folder / results_name
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to: {results_path}")

        return results_df

        return