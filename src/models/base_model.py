"""
Base model classes for maritime anomaly detection.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.
    
    All anomaly detection models should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the base anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.model_name = self.__class__.__name__
        
        logger.info(f"Initialized {self.model_name}")
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BaseAnomalyDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            X: Training features
            y: Training labels (optional for unsupervised methods)
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0: normal, 1: anomaly)
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities/scores.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores/probabilities
        """
        pass
    
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit the model and predict on the same data.
        
        Args:
            X: Input features
            y: Training labels (optional)
            
        Returns:
            Binary predictions
        """
        return self.fit(X, y).predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate model score (accuracy by default).
        
        Args:
            X: Input features
            y: True labels
            
        Returns:
            Model score
        """
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names,
            'model_name': self.model_name,
            'is_fitted': self.is_fitted
        }
        
        try:
            # Try joblib first (better for sklearn models)
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            # Fallback to pickle
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"Model saved to {filepath} using pickle")
            except Exception as e2:
                logger.error(f"Failed to save model: {e2}")
                raise
    
    def load_model(self, filepath: Union[str, Path]) -> 'BaseAnomalyDetector':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self for method chaining
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            # Try joblib first
            model_data = joblib.load(filepath)
        except Exception:
            # Fallback to pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_names = model_data['feature_names']
        self.model_name = model_data['model_name']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return self
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores if available.
        
        Returns:
            Feature importance scores or None
        """
        if not self.is_fitted:
            logger.warning("Model not fitted yet")
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).flatten()
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        else:
            return self.config
    
    def set_params(self, **params) -> 'BaseAnomalyDetector':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        
        # Update config
        self.config.update(params)
        return self
    
    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess input data.
        
        Args:
            X: Input features
            
        Returns:
            Validated input features
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        
        # Ensure 2D array
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Check for NaN values
        if np.isnan(X).any():
            logger.warning("Input contains NaN values, filling with 0")
            X = np.nan_to_num(X, nan=0.0)
        
        # Check for infinite values
        if np.isinf(X).any():
            logger.warning("Input contains infinite values, clipping")
            X = np.clip(X, -1e10, 1e10)
        
        return X
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.model_name}(fitted={self.is_fitted})"
    
    def __str__(self) -> str:
        """String representation of the model."""
        return self.__repr__()


class SupervisedAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for supervised anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.classes_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupervisedAnomalyDetector':
        """
        Fit the supervised anomaly detection model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        pass
    
    def validate_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Validate and preprocess labels.
        
        Args:
            y: Input labels
            
        Returns:
            Validated labels
        """
        # Convert to numpy array if needed
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)
        
        # Ensure 1D array
        if y.ndim > 1:
            y = y.flatten()
        
        # Check for valid binary labels
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            raise ValueError(f"Expected binary labels, got {len(unique_labels)} unique values")
        
        # Store classes
        self.classes_ = unique_labels
        
        return y


class UnsupervisedAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for unsupervised anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'UnsupervisedAnomalyDetector':
        """
        Fit the unsupervised anomaly detection model.
        
        Args:
            X: Training features
            y: Ignored (for compatibility)
            
        Returns:
            Self for method chaining
        """
        pass
    
    def set_threshold(self, threshold: float) -> 'UnsupervisedAnomalyDetector':
        """
        Set the decision threshold for anomaly detection.
        
        Args:
            threshold: Decision threshold
            
        Returns:
            Self for method chaining
        """
        self.threshold_ = threshold
        return self
    
    def find_optimal_threshold(self, X: np.ndarray, y: np.ndarray, 
                             metric: str = 'f1') -> float:
        """
        Find optimal threshold based on validation data.
        
        Args:
            X: Validation features
            y: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        scores = self.predict_proba(X)
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        
        best_score = -1
        best_threshold = 0.5
        
        metric_func = {
            'f1': f1_score,
            'precision': precision_score,
            'recall': recall_score
        }[metric]
        
        for threshold in thresholds:
            predictions = (scores >= threshold).astype(int)
            try:
                score = metric_func(y, predictions, zero_division=0)
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            except:
                continue
        
        self.threshold_ = best_threshold
        logger.info(f"Optimal threshold: {best_threshold:.4f} ({metric}: {best_score:.4f})")
        
        return best_threshold


class EnsembleAnomalyDetector(BaseAnomalyDetector):
    """
    Base class for ensemble anomaly detection models.
    """
    
    def __init__(self, base_models: list = None, config: Dict[str, Any] = None):
        super().__init__(config)
        self.base_models = base_models or []
        self.weights = None
        self.meta_model = None
    
    def add_model(self, model: BaseAnomalyDetector, weight: float = 1.0) -> 'EnsembleAnomalyDetector':
        """
        Add a base model to the ensemble.
        
        Args:
            model: Base anomaly detection model
            weight: Weight for the model in ensemble
            
        Returns:
            Self for method chaining
        """
        self.base_models.append(model)
        if self.weights is None:
            self.weights = [weight]
        else:
            self.weights.append(weight)
        
        return self
    
    def set_weights(self, weights: list) -> 'EnsembleAnomalyDetector':
        """
        Set weights for base models.
        
        Args:
            weights: List of weights for base models
            
        Returns:
            Self for method chaining
        """
        if len(weights) != len(self.base_models):
            raise ValueError("Number of weights must match number of base models")
        
        self.weights = weights
        return self
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'EnsembleAnomalyDetector':
        """
        Fit the ensemble model.
        
        Args:
            X: Training features
            y: Training labels (optional)
            
        Returns:
            Self for method chaining
        """
        pass
    
    def get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Args:
            X: Input features
            
        Returns:
            Array of shape (n_samples, n_models) with base model predictions
        """
        if not self.base_models:
            raise ValueError("No base models added to ensemble")
        
        predictions = []
        for model in self.base_models:
            if not model.is_fitted:
                raise ValueError(f"Base model {model.model_name} is not fitted")
            
            pred = model.predict_proba(X)
            predictions.append(pred)
        
        return np.column_stack(predictions)
    
    def __len__(self) -> int:
        """Return number of base models."""
        return len(self.base_models)
    
    def __getitem__(self, index: int) -> BaseAnomalyDetector:
        """Get base model by index."""
        return self.base_models[index] 