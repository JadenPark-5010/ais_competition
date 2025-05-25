"""
Ensemble anomaly detection models for maritime data.

This module implements ensemble approaches that combine multiple base models
for robust and accurate anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .base_model import EnsembleAnomalyDetector, BaseAnomalyDetector
from .traisformer import TrAISformerAnomalyDetector
from .clustering_model import CombinedClusteringAnomalyDetector, StatisticalAnomalyDetector

logger = logging.getLogger(__name__)


class WeightedEnsembleAnomalyDetector(EnsembleAnomalyDetector):
    """
    Weighted ensemble anomaly detector.
    
    Combines predictions from multiple base models using weighted averaging.
    """
    
    def __init__(self, base_models: List[BaseAnomalyDetector] = None, config: Dict[str, Any] = None):
        """
        Initialize weighted ensemble.
        
        Args:
            base_models: List of base anomaly detection models
            config: Configuration dictionary
        """
        super().__init__(base_models, config)
        
        # Ensemble configuration
        ensemble_config = self.config.get('models', {}).get('ensemble', {})
        
        # Default weights if not provided
        if self.weights is None:
            default_weights = ensemble_config.get('weights', [])
            if default_weights and len(default_weights) == len(self.base_models):
                self.weights = default_weights
            else:
                # Equal weights
                self.weights = [1.0 / len(self.base_models)] * len(self.base_models)
        
        # Normalize weights
        if self.weights:
            weight_sum = sum(self.weights)
            self.weights = [w / weight_sum for w in self.weights]
        
        logger.info(f"Weighted Ensemble initialized with {len(self.base_models)} models")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'WeightedEnsembleAnomalyDetector':
        """
        Fit all base models.
        
        Args:
            X: Training features
            y: Training labels (optional)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting weighted ensemble models")
        
        # Fit each base model
        for i, model in enumerate(self.base_models):
            logger.info(f"Fitting base model {i + 1}/{len(self.base_models)}: {model.model_name}")
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Failed to fit model {model.model_name}: {e}")
                # Set weight to 0 for failed models
                self.weights[i] = 0
        
        # Renormalize weights after removing failed models
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
        else:
            raise ValueError("All base models failed to fit")
        
        self.is_fitted = True
        logger.info("Weighted ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using weighted ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using weighted ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Weighted ensemble scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Get predictions from all base models
        all_scores = []
        valid_weights = []
        
        for i, model in enumerate(self.base_models):
            if self.weights[i] > 0:  # Only use models with positive weights
                try:
                    scores = model.predict_proba(X)
                    
                    # Handle different output formats
                    if scores.ndim == 2 and scores.shape[1] == 2:
                        # Binary classification probabilities
                        scores = scores[:, 1]  # Use anomaly class probability
                    elif scores.ndim == 2 and scores.shape[1] == 1:
                        scores = scores.flatten()
                    
                    all_scores.append(scores)
                    valid_weights.append(self.weights[i])
                    
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model.model_name}: {e}")
                    continue
        
        if not all_scores:
            raise ValueError("No valid predictions from base models")
        
        # Weighted average
        all_scores = np.array(all_scores)
        valid_weights = np.array(valid_weights)
        
        # Normalize weights
        valid_weights = valid_weights / valid_weights.sum()
        
        # Calculate weighted average
        ensemble_scores = np.average(all_scores, axis=0, weights=valid_weights)
        
        return ensemble_scores


class StackingEnsembleAnomalyDetector(EnsembleAnomalyDetector):
    """
    Stacking ensemble anomaly detector.
    
    Uses a meta-learner to combine predictions from base models.
    """
    
    def __init__(self, base_models: List[BaseAnomalyDetector] = None, config: Dict[str, Any] = None):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base anomaly detection models
            config: Configuration dictionary
        """
        super().__init__(base_models, config)
        
        # Meta-learner configuration
        ensemble_config = self.config.get('models', {}).get('ensemble', {})
        meta_config = ensemble_config.get('meta_learner', {})
        
        self.meta_learner_type = meta_config.get('type', 'xgboost')
        self.cv_folds = ensemble_config.get('strategy', {}).get('cross_validation_folds', 5)
        
        # Initialize meta-learner
        self.meta_model = self._create_meta_learner(meta_config)
        self.scaler = StandardScaler()
        
        logger.info(f"Stacking Ensemble initialized with {self.meta_learner_type} meta-learner")
    
    def _create_meta_learner(self, meta_config: Dict[str, Any]):
        """Create meta-learner model based on configuration."""
        if self.meta_learner_type == 'xgboost':
            xgb_config = meta_config.get('xgboost', {})
            return xgb.XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                learning_rate=xgb_config.get('learning_rate', 0.1),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                random_state=xgb_config.get('random_state', 42),
                eval_metric='logloss'
            )
        
        elif self.meta_learner_type == 'random_forest':
            rf_config = meta_config.get('random_forest', {})
            return RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', 10),
                min_samples_split=rf_config.get('min_samples_split', 2),
                min_samples_leaf=rf_config.get('min_samples_leaf', 1),
                random_state=rf_config.get('random_state', 42)
            )
        
        elif self.meta_learner_type == 'logistic_regression':
            lr_config = meta_config.get('logistic_regression', {})
            return LogisticRegression(
                C=lr_config.get('C', 1.0),
                penalty=lr_config.get('penalty', 'l2'),
                solver=lr_config.get('solver', 'liblinear'),
                random_state=lr_config.get('random_state', 42),
                max_iter=1000
            )
        
        else:
            raise ValueError(f"Unknown meta-learner type: {self.meta_learner_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsembleAnomalyDetector':
        """
        Fit stacking ensemble with cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        if y is None:
            raise ValueError("Stacking ensemble requires labels for training")
        
        logger.info("Fitting stacking ensemble with cross-validation")
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Fit meta-learner
        meta_features_scaled = self.scaler.fit_transform(meta_features)
        self.meta_model.fit(meta_features_scaled, y)
        
        # Fit all base models on full data
        for i, model in enumerate(self.base_models):
            logger.info(f"Fitting base model {i + 1}/{len(self.base_models)}: {model.model_name}")
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Failed to fit model {model.model_name}: {e}")
        
        self.is_fitted = True
        logger.info("Stacking ensemble training completed")
        
        return self
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Generate meta-features using cross-validation.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Meta-features array
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]
            
            for i, model in enumerate(self.base_models):
                try:
                    # Create a copy of the model for this fold
                    model_copy = type(model)(model.config)
                    
                    # Fit on training fold
                    model_copy.fit(X_train, y_train)
                    
                    # Predict on validation fold
                    val_scores = model_copy.predict_proba(X_val)
                    
                    # Handle different output formats
                    if val_scores.ndim == 2 and val_scores.shape[1] == 2:
                        val_scores = val_scores[:, 1]
                    elif val_scores.ndim == 2 and val_scores.shape[1] == 1:
                        val_scores = val_scores.flatten()
                    
                    meta_features[val_idx, i] = val_scores
                    
                except Exception as e:
                    logger.warning(f"Failed to generate meta-features for {model.model_name} in fold {fold}: {e}")
                    # Use default scores for failed models
                    meta_features[val_idx, i] = 0.5
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using stacking ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using stacking ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Get base model predictions
        base_predictions = self.get_base_predictions(X)
        
        # Scale meta-features
        base_predictions_scaled = self.scaler.transform(base_predictions)
        
        # Meta-learner prediction
        probabilities = self.meta_model.predict_proba(base_predictions_scaled)
        
        return probabilities


class VotingEnsembleAnomalyDetector(EnsembleAnomalyDetector):
    """
    Voting ensemble anomaly detector.
    
    Combines predictions using majority voting (hard voting) or 
    average probabilities (soft voting).
    """
    
    def __init__(self, base_models: List[BaseAnomalyDetector] = None, 
                 config: Dict[str, Any] = None, voting: str = 'soft'):
        """
        Initialize voting ensemble.
        
        Args:
            base_models: List of base anomaly detection models
            config: Configuration dictionary
            voting: Voting strategy ('hard' or 'soft')
        """
        super().__init__(base_models, config)
        
        self.voting = voting
        
        logger.info(f"Voting Ensemble initialized with {voting} voting")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'VotingEnsembleAnomalyDetector':
        """
        Fit all base models.
        
        Args:
            X: Training features
            y: Training labels (optional)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting voting ensemble models")
        
        # Fit each base model
        for i, model in enumerate(self.base_models):
            logger.info(f"Fitting base model {i + 1}/{len(self.base_models)}: {model.model_name}")
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Failed to fit model {model.model_name}: {e}")
        
        self.is_fitted = True
        logger.info("Voting ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using voting ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self.validate_input(X)
        
        if self.voting == 'hard':
            # Hard voting: majority vote
            all_predictions = []
            
            for model in self.base_models:
                try:
                    predictions = model.predict(X)
                    all_predictions.append(predictions)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model.model_name}: {e}")
                    continue
            
            if not all_predictions:
                raise ValueError("No valid predictions from base models")
            
            all_predictions = np.array(all_predictions)
            # Majority vote
            ensemble_predictions = (all_predictions.mean(axis=0) > 0.5).astype(int)
            
            return ensemble_predictions
        
        else:  # soft voting
            scores = self.predict_proba(X)
            return (scores > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores using soft voting.
        
        Args:
            X: Input features
            
        Returns:
            Average anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self.validate_input(X)
        
        all_scores = []
        
        for model in self.base_models:
            try:
                scores = model.predict_proba(X)
                
                # Handle different output formats
                if scores.ndim == 2 and scores.shape[1] == 2:
                    scores = scores[:, 1]
                elif scores.ndim == 2 and scores.shape[1] == 1:
                    scores = scores.flatten()
                
                all_scores.append(scores)
                
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model.model_name}: {e}")
                continue
        
        if not all_scores:
            raise ValueError("No valid predictions from base models")
        
        # Average scores
        all_scores = np.array(all_scores)
        ensemble_scores = all_scores.mean(axis=0)
        
        return ensemble_scores


class AdaptiveEnsembleAnomalyDetector(EnsembleAnomalyDetector):
    """
    Adaptive ensemble that dynamically adjusts model weights based on performance.
    """
    
    def __init__(self, base_models: List[BaseAnomalyDetector] = None, config: Dict[str, Any] = None):
        """
        Initialize adaptive ensemble.
        
        Args:
            base_models: List of base anomaly detection models
            config: Configuration dictionary
        """
        super().__init__(base_models, config)
        
        self.performance_history = []
        self.adaptive_weights = None
        
        logger.info("Adaptive Ensemble initialized")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveEnsembleAnomalyDetector':
        """
        Fit adaptive ensemble with performance-based weighting.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        if y is None:
            raise ValueError("Adaptive ensemble requires labels for training")
        
        logger.info("Fitting adaptive ensemble")
        
        # Evaluate each model using cross-validation
        model_scores = []
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Evaluating model {i + 1}/{len(self.base_models)}: {model.model_name}")
            
            try:
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X, y, cv=5, scoring='f1', 
                    error_score='raise'
                )
                avg_score = cv_scores.mean()
                model_scores.append(avg_score)
                
                logger.info(f"{model.model_name} CV F1 Score: {avg_score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model.model_name}: {e}")
                model_scores.append(0.0)
        
        # Calculate adaptive weights based on performance
        model_scores = np.array(model_scores)
        
        # Softmax weighting
        exp_scores = np.exp(model_scores - model_scores.max())
        self.adaptive_weights = exp_scores / exp_scores.sum()
        
        logger.info(f"Adaptive weights: {dict(zip([m.model_name for m in self.base_models], self.adaptive_weights))}")
        
        # Fit all models on full data
        for model in self.base_models:
            try:
                model.fit(X, y)
            except Exception as e:
                logger.error(f"Failed to fit {model.model_name}: {e}")
        
        self.is_fitted = True
        logger.info("Adaptive ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using adaptive weights.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        scores = self.predict_proba(X)
        return (scores > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using adaptive weighted ensemble.
        
        Args:
            X: Input features
            
        Returns:
            Weighted ensemble scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        X = self.validate_input(X)
        
        all_scores = []
        valid_weights = []
        
        for i, model in enumerate(self.base_models):
            try:
                scores = model.predict_proba(X)
                
                # Handle different output formats
                if scores.ndim == 2 and scores.shape[1] == 2:
                    scores = scores[:, 1]
                elif scores.ndim == 2 and scores.shape[1] == 1:
                    scores = scores.flatten()
                
                all_scores.append(scores)
                valid_weights.append(self.adaptive_weights[i])
                
            except Exception as e:
                logger.warning(f"Failed to get predictions from {model.model_name}: {e}")
                continue
        
        if not all_scores:
            raise ValueError("No valid predictions from base models")
        
        # Weighted average
        all_scores = np.array(all_scores)
        valid_weights = np.array(valid_weights)
        
        # Normalize weights
        valid_weights = valid_weights / valid_weights.sum()
        
        ensemble_scores = np.average(all_scores, axis=0, weights=valid_weights)
        
        return ensemble_scores


def create_ensemble_model(config: Dict[str, Any], ensemble_type: str = 'weighted') -> EnsembleAnomalyDetector:
    """
    Create ensemble anomaly detection model.
    
    Args:
        config: Configuration dictionary
        ensemble_type: Type of ensemble ('weighted', 'stacking', 'voting', 'adaptive')
        
    Returns:
        Ensemble anomaly detector
    """
    # Create base models
    base_models = []
    
    # Add TrAISformer if enabled
    if config.get('models', {}).get('traisformer', {}).get('enable', True):
        base_models.append(TrAISformerAnomalyDetector(config))
    
    # Add clustering model if enabled
    if config.get('models', {}).get('clustering', {}).get('enable', True):
        base_models.append(CombinedClusteringAnomalyDetector(config))
    
    # Add statistical model if enabled
    if config.get('models', {}).get('statistical', {}).get('enable', True):
        base_models.append(StatisticalAnomalyDetector(config))
    
    if not base_models:
        raise ValueError("No base models enabled in configuration")
    
    # Create ensemble
    if ensemble_type == 'weighted':
        return WeightedEnsembleAnomalyDetector(base_models, config)
    elif ensemble_type == 'stacking':
        return StackingEnsembleAnomalyDetector(base_models, config)
    elif ensemble_type == 'voting':
        return VotingEnsembleAnomalyDetector(base_models, config)
    elif ensemble_type == 'adaptive':
        return AdaptiveEnsembleAnomalyDetector(base_models, config)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}") 