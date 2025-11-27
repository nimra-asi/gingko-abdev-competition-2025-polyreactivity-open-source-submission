"""
Utility functions for improved antibody polyreactivity prediction.

This module contains advanced feature engineering, modeling, and evaluation
functions to boost prediction performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr, zscore
import warnings
warnings.filterwarnings('ignore')

# Try to import BioPython
try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False


class SequenceFeatureExtractor:
    """Extract physicochemical features from antibody sequences."""
    
    @staticmethod
    def extract_features(sequence: str, chain_name: str = '') -> Dict[str, float]:
        """
        Extract comprehensive physicochemical features from sequence.
        
        Args:
            sequence: Amino acid sequence
            chain_name: Prefix for feature names (e.g., 'vh', 'vl')
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Basic composition
        features[f'{chain_name}_length'] = len(sequence)
        
        # Amino acid counts
        aa_counts = {aa: sequence.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        
        # Charge-related features
        features[f'{chain_name}_positive_charge'] = (
            aa_counts.get('K', 0) + aa_counts.get('R', 0) + aa_counts.get('H', 0)
        )
        features[f'{chain_name}_negative_charge'] = (
            aa_counts.get('D', 0) + aa_counts.get('E', 0)
        )
        features[f'{chain_name}_net_charge'] = (
            features[f'{chain_name}_positive_charge'] - 
            features[f'{chain_name}_negative_charge']
        )
        features[f'{chain_name}_charge_density'] = (
            features[f'{chain_name}_net_charge'] / len(sequence)
        )
        
        # Hydrophobic residues
        hydrophobic = sum(aa_counts.get(aa, 0) for aa in 'AILMFWYV')
        features[f'{chain_name}_hydrophobic_ratio'] = hydrophobic / len(sequence)
        
        # Polar uncharged
        polar = sum(aa_counts.get(aa, 0) for aa in 'STNQ')
        features[f'{chain_name}_polar_ratio'] = polar / len(sequence)
        
        # Aromatic residues
        aromatic = sum(aa_counts.get(aa, 0) for aa in 'FWY')
        features[f'{chain_name}_aromatic_ratio'] = aromatic / len(sequence)
        
        # Surface patches
        features[f'{chain_name}_max_consecutive_hydrophobic'] = (
            SequenceFeatureExtractor._max_consecutive(sequence, 'AILMFWYV')
        )
        features[f'{chain_name}_max_consecutive_charged'] = (
            SequenceFeatureExtractor._max_consecutive(sequence, 'DEKRH')
        )
        features[f'{chain_name}_max_consecutive_positive'] = (
            SequenceFeatureExtractor._max_consecutive(sequence, 'KRH')
        )
        
        # Special sites
        features[f'{chain_name}_glycosylation_sites'] = (
            SequenceFeatureExtractor._count_glycosylation_sites(sequence)
        )
        features[f'{chain_name}_aggregation_score'] = (
            SequenceFeatureExtractor._calculate_aggregation_score(sequence)
        )
        
        # BioPython features
        if BIO_AVAILABLE:
            try:
                pa = ProteinAnalysis(sequence)
                features[f'{chain_name}_pI'] = pa.isoelectric_point()
                features[f'{chain_name}_instability_index'] = pa.instability_index()
                features[f'{chain_name}_gravy'] = pa.gravy()
                features[f'{chain_name}_aromaticity'] = pa.aromaticity()
                
                ss = pa.secondary_structure_fraction()
                features[f'{chain_name}_helix_fraction'] = ss[0]
                features[f'{chain_name}_turn_fraction'] = ss[1]
                features[f'{chain_name}_sheet_fraction'] = ss[2]
            except:
                pass
        
        return features
    
    @staticmethod
    def _max_consecutive(sequence: str, aa_set: str) -> int:
        """Count maximum consecutive residues from aa_set."""
        max_count = 0
        current_count = 0
        for aa in sequence:
            if aa in aa_set:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count
    
    @staticmethod
    def _count_glycosylation_sites(sequence: str) -> int:
        """Count N-glycosylation sites (N-X-S/T pattern)."""
        count = 0
        for i in range(len(sequence) - 2):
            if (sequence[i] == 'N' and 
                sequence[i+1] != 'P' and 
                sequence[i+2] in 'ST'):
                count += 1
        return count
    
    @staticmethod
    def _calculate_aggregation_score(sequence: str) -> float:
        """Simple aggregation propensity score."""
        window_size = 5
        hydrophobic_aa = 'VILMFWY'
        max_score = 0
        
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            score = sum(1 for aa in window if aa in hydrophobic_aa)
            max_score = max(max_score, score)
        
        return max_score / window_size if sequence else 0


class FeatureFusion:
    """Combine multiple embedding types into unified feature space."""
    
    @staticmethod
    def create_fusion_features(
        df: pd.DataFrame,
        embedding_cols: Dict[str, List[str]],
        moe_cols: Optional[List[str]] = None,
        use_pca: bool = True,
        n_components: int = 50
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create fused feature space from multiple embeddings.
        
        Args:
            df: DataFrame with embeddings
            embedding_cols: Dict mapping embedding names to column lists
            moe_cols: List of MOE feature columns
            use_pca: Whether to reduce each embedding with PCA
            n_components: Number of PCA components per embedding
            
        Returns:
            Fused feature matrix and feature names
        """
        feature_sets = []
        feature_names = []
        
        # Process each embedding
        for emb_name, emb_cols in embedding_cols.items():
            X_emb = df[emb_cols].values
            
            if use_pca and X_emb.shape[1] > n_components:
                print(f"Reducing {emb_name}: {X_emb.shape[1]} → {n_components} dims")
                pca = PCA(n_components=n_components, random_state=42)
                X_emb_reduced = pca.fit_transform(X_emb)
                print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.3f}")
                feature_sets.append(X_emb_reduced)
                feature_names.extend([f"{emb_name}_pc{i}" for i in range(n_components)])
            else:
                feature_sets.append(X_emb)
                feature_names.extend([f"{emb_name}_{i}" for i in range(X_emb.shape[1])])
        
        # Add MOE features
        if moe_cols:
            X_moe = df[moe_cols].values
            feature_sets.append(X_moe)
            feature_names.extend(moe_cols)
        
        # Concatenate
        X_combined = np.hstack(feature_sets)
        
        return X_combined, feature_names
    
    @staticmethod
    def add_interaction_features(
        X_base: np.ndarray,
        X_moe: np.ndarray,
        n_top_features: int = 10
    ) -> np.ndarray:
        """
        Create interaction features between embeddings and MOE.
        
        Args:
            X_base: Base embedding features
            X_moe: MOE features
            n_top_features: Number of top base features to use
            
        Returns:
            Interaction feature matrix
        """
        # Select top features by variance
        variances = np.var(X_base, axis=0)
        top_indices = np.argsort(variances)[-n_top_features:]
        
        interactions = []
        for i in top_indices:
            for j in range(X_moe.shape[1]):
                interactions.append(X_base[:, i] * X_moe[:, j])
        
        if interactions:
            return np.column_stack(interactions)
        else:
            return np.array([]).reshape(X_base.shape[0], 0)


class TargetPreparation:
    """Prepare and transform target variable."""
    
    @staticmethod
    def prepare_target(
        y: np.ndarray,
        method: str = 'power',
        remove_outliers: bool = True,
        outlier_threshold: float = 3.5
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
        """
        Prepare target with outlier removal and transformation.
        
        Args:
            y: Target values
            method: Transformation method ('power', 'quantile', 'log', 'none')
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outliers
            
        Returns:
            Transformed target, mask, transformer object
        """
        mask = np.ones(len(y), dtype=bool)
        
        # Remove outliers
        if remove_outliers:
            z_scores = np.abs(zscore(y))
            mask = z_scores < outlier_threshold
            n_outliers = (~mask).sum()
            if n_outliers > 0:
                print(f"Removing {n_outliers} outliers (|z| > {outlier_threshold})")
        
        y_clean = y[mask]
        
        # Transform
        transformer = None
        if method == 'power':
            transformer = PowerTransformer(method='yeo-johnson')
            y_trans = transformer.fit_transform(y_clean.reshape(-1, 1)).ravel()
        elif method == 'quantile':
            transformer = QuantileTransformer(output_distribution='normal')
            y_trans = transformer.fit_transform(y_clean.reshape(-1, 1)).ravel()
        elif method == 'log':
            y_trans = np.log1p(y_clean - y_clean.min() + 1)
        else:
            y_trans = y_clean
        
        return y_trans, mask, transformer


class ModelEvaluator:
    """Evaluate models with cross-validation."""
    
    @staticmethod
    def evaluate_models_cv(
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        models_dict: Dict[str, Any],
        cv_splits: int = 5,
        scale_features: bool = True
    ) -> pd.DataFrame:
        """
        Comprehensive cross-validation evaluation.
        
        Args:
            X: Feature matrix
            y: Target values
            groups: Group labels for GroupKFold
            models_dict: Dictionary of models to evaluate
            cv_splits: Number of CV splits
            scale_features: Whether to scale features
            
        Returns:
            DataFrame with detailed results
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        results = []
        gkf = GroupKFold(n_splits=cv_splits)
        scaler = StandardScaler() if scale_features else None
        
        print("="*60)
        print("CROSS-VALIDATION EVALUATION")
        print("="*60)
        print(f"Splits: {cv_splits}")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print("="*60 + "\n")
        
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            fold_results = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), 1):
                # Split
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale
                if scaler:
                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                
                # Train
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_val)
                
                # Metrics
                spearman_corr, spearman_p = spearmanr(y_val, y_pred)
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                fold_results.append({
                    'model': model_name,
                    'fold': fold_idx,
                    'spearman': spearman_corr,
                    'spearman_p': spearman_p,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'n_train': len(train_idx),
                    'n_val': len(val_idx)
                })
                
                print(f"  Fold {fold_idx}: Spearman={spearman_corr:.4f}, "
                      f"MAE={mae:.4f}, R²={r2:.4f}")
            
            results.extend(fold_results)
            
            # Summary
            avg_spearman = np.mean([r['spearman'] for r in fold_results])
            std_spearman = np.std([r['spearman'] for r in fold_results])
            print(f"  ✓ Average: {avg_spearman:.4f} ± {std_spearman:.4f}")
        
        return pd.DataFrame(results)


def get_recommended_models():
    """
    Get a dictionary of recommended models for antibody property prediction.
    
    Returns:
        Dictionary of model instances
    """
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    
    models = {
        'ridge': Ridge(alpha=10.0),
        'lasso': Lasso(alpha=0.1, max_iter=2000),
        'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        'huber': HuberRegressor(epsilon=1.35, alpha=0.1),
        'rf': RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        ),
        'lgb': lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        'gbr': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42
        ),
    }
    
    return models


def create_stacking_ensemble(base_models: Dict[str, Any], meta_model: str = 'ridge'):
    """
    Create a stacking ensemble.
    
    Args:
        base_models: Dictionary of base models
        meta_model: Type of meta-learner ('ridge', 'elastic')
        
    Returns:
        StackingRegressor instance
    """
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    
    estimators = [(name, model) for name, model in base_models.items()]
    
    if meta_model == 'ridge':
        final_estimator = Ridge(alpha=1.0)
    elif meta_model == 'elastic':
        final_estimator = ElasticNet(alpha=0.1, l1_ratio=0.5)
    else:
        final_estimator = Ridge(alpha=1.0)
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    
    return stacking


if __name__ == "__main__":
    print("Antibody Polyreactivity Prediction Utilities")
    print("=" * 60)
    print("Available classes:")
    print("  - SequenceFeatureExtractor: Extract features from sequences")
    print("  - FeatureFusion: Combine multiple embeddings")
    print("  - TargetPreparation: Transform target variable")
    print("  - ModelEvaluator: Cross-validation evaluation")
    print("\nAvailable functions:")
    print("  - get_recommended_models(): Get pre-configured models")
    print("  - create_stacking_ensemble(): Create ensemble model")
