import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Literal, Optional, List
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

class FinalModelEvaluator:
    """
    Runs and saves OOF predictions for a single, pre-tuned model using Grouped CV
    and handles StandardScaler and PCA within each fold.
    """

    def __init__(self, 
                 output_folder: str, 
                 input_df: pd.DataFrame, 
                 group_column: str, 
                 value_column: str, 
                 model_class: Any, model_params: Dict[str, Any],
                 n_components: Optional[int],
                 dr_method: Optional[Literal['pca']] = 'pca',  
                 n_splits: int = 5):
        
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.group_column = group_column
        self.value_column = value_column
        self.input_df = input_df
        
        self.model_class = model_class
        self.model_params = model_params
        
        self.dr_method = dr_method
        self.n_components = n_components
        self.n_splits = n_splits

    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the antibody dataset."""
        df = self.input_df
        # Ensure 'vh', 'vl' and all required columns exist and drop NaNs
        required_cols = ['vh', 'vl', self.value_column, self.group_column]
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        
        # Check if metadata columns exist for the submission file (adjust names as needed)
        self.submission_cols = [
            col for col in ['antibody_name', 'vh', 'vl', self.group_column] 
            if col in df.columns
        ]
        
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features."""
        non_feature_cols = ['antibody_id', 'antibody_name', 'titer', 'purity', 'sec_perc_monomer',
       'smac', 'hic', 'hac', 'pr_cho', 'pr_ova', 'ac_sins_ph6.0',
       'ac_sins_ph7.4', 'tonset', 'tm1', 'tm2', 'hc_subtype', 'lc_subtype',
       'highest_clinical_trial_asof_feb2025', 'est_status_asof_feb2025', 'vh',
       'hc_protein_sequence', 'hc_dna_sequence', 'vl', 'lc_protein_sequence',
       'lc_dna_sequence', 'hierarchical_cluster_fold', 'random_fold',
       'hierarchical_cluster_igg_isotype_stratified_fold', 'light_aligned_aho',
       'heavy_aligned_aho']
        
        feature_cols = [x for x in df.columns if x not in non_feature_cols] 
        return df[feature_cols]

    def apply_feature_processing(self, 
                                 X_train: np.ndarray, 
                                 X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler, Optional[PCA]]:
        """Applies scaling and PCA, fitting only on X_train."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        dr_model = None
        if self.dr_method == 'pca':
            n_comp = min(self.n_components, X_train_scaled.shape[1], X_train_scaled.shape[0] - 1)
            dr_model = PCA(n_components=n_comp, random_state=42)
            
            X_train_processed = dr_model.fit_transform(X_train_scaled)
            X_test_processed = dr_model.transform(X_test_scaled)
        else:
            X_train_processed = X_train_scaled
            X_test_processed = X_test_scaled

        return X_train_processed, X_test_processed, scaler, dr_model

    def run_submission_pipeline(self):
        """Runs Grouped CV, collects OOF predictions, and saves the submission file."""
        
        df = self.load_and_preprocess_data()
        
        # 1. Prepare Data
        features = self.extract_features(df)
        X = features.values
        y = df[self.value_column].values
        groups = df[self.group_column].values
        
        outer_cv = GroupKFold(n_splits=self.n_splits)
        model_name = self.model_class.__name__
        
        all_test_indices = []
        all_test_predictions = []
        all_metrics = []

        print(f"Starting {self.n_splits}-Fold Grouped CV for {model_name}...")
        print(f"Model Params: {self.model_params}")
        print(f"DR: {self.dr_method} with {self.n_components} components")

        # 2. Grouped CV Loop
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
            
            X_train_outer, X_test_outer = X[train_idx], X[test_idx]
            y_train_outer, y_test_outer = y[train_idx], y[test_idx]
            
            # Apply scaling and PCA (fitted only on train data)
            X_train_processed, X_test_processed, _, _ = self.apply_feature_processing(
                X_train_outer, X_test_outer
            )

            # Initialize and fit the best model
            model = self.model_class(**self.model_params)
            model.fit(X_train_processed, y_train_outer)
            y_pred = model.predict(X_test_processed)

            # 3. Collect OOF Predictions and Metrics
            all_test_indices.extend(test_idx)
            all_test_predictions.extend(y_pred)
            
            # Calculate fold metrics
            spearman_corr, _ = spearmanr(y_pred, y_test_outer)
            mae = mean_absolute_error(y_test_outer, y_pred)
            all_metrics.append({'fold': fold + 1, 'SpearmanR': spearman_corr, 'MAE': mae})
        
        # 4. Final Submission File Generation (OOF)
        
        # Map predictions back to the original (reset) DataFrame index
        pred_series = pd.Series(all_test_predictions, index=all_test_indices)
        pred_series.sort_index(inplace=True) 

        # Create submission DataFrame using the processed data frame structure
        submission_df = df.iloc[pred_series.index][self.submission_cols].copy()
        
        # Add the Out-Of-Fold predictions
        submission_df[self.value_column] = pred_series.values
        
        # 5. Save Results
        avg_metrics_df = pd.DataFrame(all_metrics)
        print("\n--- OOF Performance Summary ---")
        print(avg_metrics_df.mean(numeric_only=True))
        
        submission_filename = f'{self.value_column}_{model_name}_OOF_submission.csv'
        submission_path = self.output_folder / submission_filename
        submission_df = submission_df.rename(columns=({"vh": "vh_protein_sequence",
                                                       "vl": "vl_protein_sequence"}))
        submission_df.to_csv(submission_path, index=False)
        
        print(f"\nSubmission file saved successfully to: {submission_path}")
        return submission_df, avg_metrics_df
