"""
LightGBM model wrapper for PathSnap candidate scoring.

Family: Boosting (gradient boosting decision trees)
Owner: Member D

This wrapper provides:
- fit() with early stopping on validation fold
- predict_proba() for probability predictions
- save() and load() for model persistence
- Training and inference timing
- Group-aware training considerations
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss


class LightGBMModel:
    """
    LightGBM binary classifier for candidate scoring.
    
    Uses early stopping on validation data to prevent overfitting.
    Designed for tabular feature data from GPS candidate generation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM model from config.
        
        Args:
            config: Configuration dictionary from configs/models/lgbm.yaml
        """
        self.config = config
        self.model = None
        self.best_iteration = None
        self.best_score = None
        self.train_time_s = None
        self.infer_time_s = None
        self.feature_names = None
        
        # Default parameters
        default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': -1,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }
        
        # Override with config params
        self.params = {**default_params, **config.get('params', {})}
        
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        trip_ids_train: Optional[np.ndarray] = None,
        trip_ids_val: Optional[np.ndarray] = None,
    ) -> 'LightGBMModel':
        """
        Fit the LightGBM model with optional early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels (0/1 for candidate correctness)
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            trip_ids_train: Trip IDs for training samples (for group awareness)
            trip_ids_val: Trip IDs for validation samples
            
        Returns:
            self
        """
        start_time = time.time()
        
        # Store feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
            
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names,
            free_raw_data=False
        )
        
        # Early stopping setup
        callbacks = []
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            val_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                reference=train_data,
                free_raw_data=False
            )
            
            callbacks.append(lgb.early_stopping(
                stopping_rounds=self.config.get('early_stopping_rounds', 10),
                verbose=self.config.get('verbose_early_stopping', False)
            ))
            
            callbacks.append(lgb.log_evaluation(
                period=self.config.get('log_evaluation_period', 0)  # 0 = silent
            ))
        
        # Train model
        self.model = lgb.train(
            params={
                'objective': self.params['objective'],
                'metric': self.params['metric'],
                'boosting_type': self.params['boosting_type'],
                'num_leaves': self.params['num_leaves'],
                'max_depth': self.params['max_depth'],
                'learning_rate': self.params['learning_rate'],
                'min_child_samples': self.params['min_child_samples'],
                'subsample': self.params['subsample'],
                'colsample_bytree': self.params['colsample_bytree'],
                'reg_alpha': self.params['reg_alpha'],
                'reg_lambda': self.params['reg_lambda'],
                'random_state': self.params['random_state'],
                'n_jobs': self.params['n_jobs'],
                'verbose': self.params['verbose'],
            },
            train_set=train_data,
            num_boost_round=self.params['n_estimators'],
            valid_sets=[train_data, val_data] if X_val is not None else [train_data],
            valid_names=['train', 'valid'] if X_val is not None else ['train'],
            callbacks=callbacks,
        )
        
        self.best_iteration = self.model.best_iteration
        self.best_score = self.model.best_score
        
        self.train_time_s = time.time() - start_time
        
        return self
    
    def predict_proba(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict probabilities for candidate correctness.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 2) with [P(incorrect), P(correct)]
        """
        start_time = time.time()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Get probabilities for positive class (correct candidate)
        proba_positive = self.model.predict(X, num_iteration=self.best_iteration)
        
        # Create 2D array with [P(0), P(1)]
        proba = np.zeros((len(proba_positive), 2))
        proba[:, 0] = 1 - proba_positive
        proba[:, 1] = proba_positive
        
        self.infer_time_s = time.time() - start_time
        
        return proba
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict binary labels (0/1).
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model (will add .txt extension)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(filepath.with_suffix('.txt')))
        
        # Save metadata
        metadata = {
            'config': self.config,
            'best_iteration': self.best_iteration,
            'best_score': self.best_score,
            'train_time_s': self.train_time_s,
            'feature_names': self.feature_names,
            'params': self.params,
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'LightGBMModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file (without extension)
            
        Returns:
            Loaded LightGBMModel instance
        """
        filepath = Path(filepath)
        
        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(metadata['config'])
        instance.best_iteration = metadata['best_iteration']
        instance.best_score = metadata['best_score']
        instance.train_time_s = metadata['train_time_s']
        instance.feature_names = metadata['feature_names']
        instance.params = metadata['params']
        
        # Load LightGBM model
        instance.model = lgb.Booster(
            model_file=str(filepath.with_suffix('.txt'))
        )
        
        return instance
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: 'gain', 'split', or 'cover'
            
        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.model.feature_importance(importance_type=importance_type)
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()
    
    def get_timing(self) -> Dict[str, Optional[float]]:
        """Get training and inference timing."""
        return {
            'train_time_s': self.train_time_s,
            'infer_time_s': self.infer_time_s,
        }
