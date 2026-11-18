"""
MLP (Multi-Layer Perceptron) model wrapper for PathSnap candidate scoring.

Family: Neural network
Owner: Member D

This wrapper provides:
- fit() with epoch-wise training and validation loss tracking
- predict_proba() for probability predictions
- save() and load() for model persistence
- Training/validation loss history saved to CSV (ONLY model with epoch histories)
- StandardScaler integration (fit on training only)
- Training and inference timing
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
import joblib


class MLPModel:
    """
    MLP binary classifier for candidate scoring.
    
    Uses sklearn's MLPClassifier with custom loss tracking.
    StandardScaler is fit on training data only and applied to all predictions.
    Training and validation loss curves are saved for visualization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP model from config.
        
        Args:
            config: Configuration dictionary from configs/models/mlp.yaml
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.train_time_s = None
        self.infer_time_s = None
        self.feature_names = None
        self.n_epochs_trained = None
        
        # Default parameters
        default_params = {
            'hidden_layer_sizes': (64, 32),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 256,
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 100,
            'tol': 1e-4,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'random_state': 42,
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
        loss_history_path: Optional[Union[str, Path]] = None,
    ) -> 'MLPModel':
        """
        Fit the MLP model with epoch-wise loss tracking.
        
        Args:
            X_train: Training features
            y_train: Training labels (0/1 for candidate correctness)
            X_val: Validation features (for loss tracking)
            y_val: Validation labels
            trip_ids_train: Trip IDs for training samples (unused, for interface consistency)
            trip_ids_val: Trip IDs for validation samples (unused)
            loss_history_path: Path to save loss history CSV
            
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
            
        # Fit StandardScaler on training data only
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Scale validation data if provided
        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize model
        # Disable sklearn's early_stopping to use our manual validation tracking
        params_no_early_stop = self.params.copy()
        params_no_early_stop['early_stopping'] = False
        
        self.model = MLPClassifier(**params_no_early_stop)
        
        # Manual epoch-wise training with loss tracking
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Use partial_fit for epoch-wise control
        classes = np.unique(y_train)
        
        # Warm start for incremental training
        self.model.warm_start = True
        self.model.max_iter = 1  # Train one epoch at a time
        
        max_epochs = self.params['max_iter']
        n_iter_no_change = self.params['n_iter_no_change']
        tol = self.params['tol']
        
        best_val_loss = np.inf
        epochs_without_improvement = 0
        
        for epoch in range(max_epochs):
            # Train one epoch
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate training loss
            train_proba = self.model.predict_proba(X_train_scaled)
            train_loss = log_loss(y_train, train_proba)
            self.train_loss_history.append(train_loss)
            
            # Calculate validation loss if provided
            if X_val is not None:
                val_proba = self.model.predict_proba(X_val_scaled)
                val_loss = log_loss(y_val, val_proba)
                self.val_loss_history.append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss - tol:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= n_iter_no_change:
                    if self.config.get('verbose', False):
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                # If no validation set, use training loss for early stopping
                if train_loss < best_val_loss - tol:
                    best_val_loss = train_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                if epochs_without_improvement >= n_iter_no_change:
                    if self.config.get('verbose', False):
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        self.n_epochs_trained = len(self.train_loss_history)
        self.train_time_s = time.time() - start_time
        
        # Save loss history to CSV
        if loss_history_path is not None:
            self.save_loss_history(loss_history_path)
        
        return self
    
    def predict_proba(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predict probabilities for candidate correctness.
        
        Args:
            X: Feature matrix (unscaled)
            
        Returns:
            Array of shape (n_samples, 2) with [P(incorrect), P(correct)]
        """
        start_time = time.time()
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Apply scaler transformation
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        proba = self.model.predict_proba(X_scaled)
        
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
    
    def save_loss_history(self, filepath: Union[str, Path]) -> None:
        """
        Save training and validation loss history to CSV.
        
        This is the ONLY model with epoch-wise histories, so this file
        is used to generate loss curve plots.
        
        Args:
            filepath: Path to save CSV file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            'epoch': range(1, len(self.train_loss_history) + 1),
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history if self.val_loss_history else [None] * len(self.train_loss_history),
        })
        
        df.to_csv(filepath, index=False)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save model and scaler to files.
        
        Args:
            filepath: Base path for saving (extensions added automatically)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn MLP model
        joblib.dump(self.model, filepath.with_suffix('.joblib'))
        
        # Save StandardScaler
        joblib.dump(self.scaler, filepath.with_suffix('.scaler'))
        
        # Save metadata
        metadata = {
            'config': self.config,
            'train_time_s': self.train_time_s,
            'feature_names': self.feature_names,
            'params': self.params,
            'n_epochs_trained': self.n_epochs_trained,
        }
        
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MLPModel':
        """
        Load model and scaler from files.
        
        Args:
            filepath: Base path for loading (without extension)
            
        Returns:
            Loaded MLPModel instance
        """
        filepath = Path(filepath)
        
        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            metadata = json.load(f)
        
        # Create instance
        instance = cls(metadata['config'])
        instance.train_time_s = metadata['train_time_s']
        instance.feature_names = metadata['feature_names']
        instance.params = metadata['params']
        instance.n_epochs_trained = metadata['n_epochs_trained']
        
        # Load sklearn MLP model
        instance.model = joblib.load(filepath.with_suffix('.joblib'))
        
        # Load StandardScaler
        instance.scaler = joblib.load(filepath.with_suffix('.scaler'))
        
        return instance
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.params.copy()
    
    def get_timing(self) -> Dict[str, Optional[float]]:
        """Get training and inference timing."""
        return {
            'train_time_s': self.train_time_s,
            'infer_time_s': self.infer_time_s,
        }
    
    def get_loss_history(self) -> pd.DataFrame:
        """
        Get training and validation loss history as DataFrame.
        
        Returns:
            DataFrame with epoch, train_loss, and val_loss columns
        """
        return pd.DataFrame({
            'epoch': range(1, len(self.train_loss_history) + 1),
            'train_loss': self.train_loss_history,
            'val_loss': self.val_loss_history if self.val_loss_history else [None] * len(self.train_loss_history),
        })
