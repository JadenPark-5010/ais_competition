"""
Model trainer for maritime anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import yaml
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from ..models.base_model import BaseAnomalyDetector
from ..models.ensemble_model import SuspiciousVesselEnsemble
from ..models.traisformer import create_traisformer_model
from ..models.clustering_model import create_clustering_model
from ..utils.metrics import AnomalyDetectionMetrics
from ..utils.logging_utils import ExperimentLogger

logger = logging.getLogger(__name__)


class AnomalyDetectionTrainer:
    """
    Trainer class for maritime anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.training_config = config.get('training', {})
        self.validation_config = config.get('validation', {})
        
        # Training parameters
        self.epochs = self.training_config.get('epochs', 100)
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.patience = self.training_config.get('patience', 10)
        self.min_delta = self.training_config.get('min_delta', 0.001)
        
        # Validation parameters
        self.cv_folds = self.validation_config.get('cross_validation', 5)
        self.validation_split = config.get('data', {}).get('validation_split', 0.2)
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.metrics_calculator = AnomalyDetectionMetrics()
        self.logger = ExperimentLogger(config)
        
        # Training history
        self.training_history = []
        self.best_model_state = None
        self.best_score = -np.inf
        
        logger.info("Anomaly Detection Trainer initialized")
    
    def create_model(self, model_type: str = 'ensemble') -> BaseAnomalyDetector:
        """
        Create model based on type.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Created model instance
        """
        logger.info(f"Creating {model_type} model")
        
        if model_type == 'ensemble':
            model = SuspiciousVesselEnsemble(self.config)
        elif model_type == 'traisformer':
            model = create_traisformer_model(self.config)
        elif model_type == 'clustering':
            model = create_clustering_model(self.config, model_type='combined')
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = model
        return model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Validation split size
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        if test_size is None:
            test_size = self.validation_split
        
        logger.info(f"Splitting data with validation size: {test_size}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size,
            random_state=42,
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        logger.info(f"Training set: {X_train_scaled.shape}, Validation set: {X_val_scaled.shape}")
        logger.info(f"Training labels distribution: {np.bincount(y_train)}")
        logger.info(f"Validation labels distribution: {np.bincount(y_val)}")
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def train(self, X: np.ndarray, y: np.ndarray, 
             model_type: str = 'ensemble',
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> BaseAnomalyDetector:
        """
        Train the anomaly detection model.
        
        Args:
            X: Training features
            y: Training labels
            model_type: Type of model to train
            validation_data: Optional validation data tuple (X_val, y_val)
            
        Returns:
            Trained model
        """
        self.logger.log_start()
        
        # Create model
        model = self.create_model(model_type)
        
        # Prepare data if validation data not provided
        if validation_data is None:
            X_train, X_val, y_train, y_val = self.prepare_data(X, y)
        else:
            X_train = self.scaler.fit_transform(X)
            y_train = y
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
        
        # Log model architecture
        self.logger.log_model(model)
        
        # Train model
        logger.info("Starting model training")
        
        if hasattr(model, 'fit'):
            # For models that support supervised training
            if model_type in ['ensemble', 'traisformer']:
                model.fit(X_train, y_train)
            else:
                # For unsupervised models
                model.fit(X_train)
        
        # Evaluate on validation set
        val_metrics = self.evaluate(model, X_val, y_val)
        
        # Log metrics
        self.logger.log_metrics(val_metrics, epoch=1, prefix="Validation")
        
        # Store best model
        self.best_model_state = model
        self.best_score = val_metrics.get('f1_score', 0)
        
        # Store scaler in model for later use
        model.scaler = self.scaler
        
        logger.info("Training completed")
        
        return model
    
    def train_with_cross_validation(self, X: np.ndarray, y: np.ndarray,
                                   model_type: str = 'ensemble') -> Dict[str, float]:
        """
        Train model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to train
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Training with {self.cv_folds}-fold cross-validation")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create model
        model = self.create_model(model_type)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {}
        metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
            logger.info(f"Training fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold = X_scaled[train_idx]
            X_val_fold = X_scaled[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            
            # Create fresh model for this fold
            fold_model = self.create_model(model_type)
            
            # Train
            if model_type in ['ensemble', 'traisformer']:
                fold_model.fit(X_train_fold, y_train_fold)
            else:
                fold_model.fit(X_train_fold)
            
            # Evaluate
            fold_metrics = self.evaluate(fold_model, X_val_fold, y_val_fold)
            metrics_list.append(fold_metrics)
            
            # Log fold results
            self.logger.log_metrics(fold_metrics, epoch=fold+1, prefix=f"Fold_{fold+1}")
        
        # Calculate average metrics
        for metric_name in metrics_list[0].keys():
            values = [m[metric_name] for m in metrics_list]
            cv_scores[f'{metric_name}_mean'] = np.mean(values)
            cv_scores[f'{metric_name}_std'] = np.std(values)
        
        # Train final model on full data
        logger.info("Training final model on full dataset")
        final_model = self.create_model(model_type)
        
        if model_type in ['ensemble', 'traisformer']:
            final_model.fit(X_scaled, y)
        else:
            final_model.fit(X_scaled)
        
        final_model.scaler = self.scaler
        self.model = final_model
        
        # Log cross-validation results
        logger.info("Cross-validation results:")
        for metric_name, value in cv_scores.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.4f}")
        
        return cv_scores
    
    def evaluate(self, model: BaseAnomalyDetector, 
                X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Evaluation metrics
        """
        # Get predictions
        y_pred = model.predict(X_val)
        y_scores = model.predict_proba(X_val)
        
        # Handle different output formats
        if y_scores.ndim == 2 and y_scores.shape[1] == 2:
            y_scores = y_scores[:, 1]
        elif y_scores.ndim == 2 and y_scores.shape[1] == 1:
            y_scores = y_scores.flatten()
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(y_val, y_pred, y_scores)
        
        return metrics
    
    def save_model(self, model: BaseAnomalyDetector, 
                  output_dir: str, model_name: str = "final_model") -> None:
        """
        Save trained model and metadata.
        
        Args:
            model: Trained model
            output_dir: Output directory
            model_name: Name for saved model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_path / f"{model_name}.pkl"
        model.save_model(model_path)
        
        # Save scaler
        scaler_path = output_path / f"{model_name}_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': model.model_name,
            'config': self.config,
            'training_history': self.training_history,
            'best_score': self.best_score,
            'feature_names': getattr(model, 'feature_names', None)
        }
        
        metadata_path = output_path / f"{model_name}_metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str) -> BaseAnomalyDetector:
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        model_path = Path(model_path)
        
        # Load model
        model = BaseAnomalyDetector()
        model.load_model(model_path)
        
        # Load scaler if available
        scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            model.scaler = self.scaler
        
        # Load metadata if available
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            self.training_history = metadata.get('training_history', [])
            self.best_score = metadata.get('best_score', 0)
        
        self.model = model
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
                            param_grid: Dict[str, List],
                            model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for search
            model_type: Type of model
            
        Returns:
            Best parameters and score
        """
        logger.info("Starting hyperparameter search")
        
        best_score = -np.inf
        best_params = None
        results = []
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))
            
            logger.info(f"Testing parameters: {params}")
            
            # Update config with current parameters
            temp_config = self.config.copy()
            for param_name, param_value in params.items():
                # Navigate nested config structure
                keys = param_name.split('.')
                config_section = temp_config
                for key in keys[:-1]:
                    config_section = config_section.setdefault(key, {})
                config_section[keys[-1]] = param_value
            
            # Create trainer with updated config
            temp_trainer = AnomalyDetectionTrainer(temp_config)
            
            # Perform cross-validation
            cv_results = temp_trainer.train_with_cross_validation(X, y, model_type)
            
            # Get score (use F1 score as default metric)
            score = cv_results.get('f1_score_mean', 0)
            
            results.append({
                'params': params,
                'score': score,
                'cv_results': cv_results
            })
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"Score: {score:.4f}")
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
    
    def get_feature_importance(self, model: Optional[BaseAnomalyDetector] = None) -> Optional[np.ndarray]:
        """
        Get feature importance from trained model.
        
        Args:
            model: Model to get importance from (uses self.model if None)
            
        Returns:
            Feature importance array or None
        """
        if model is None:
            model = self.model
        
        if model is None:
            logger.warning("No model available for feature importance")
            return None
        
        return model.get_feature_importance()
    
    def generate_training_report(self, X_val: np.ndarray, y_val: np.ndarray) -> str:
        """
        Generate comprehensive training report.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Training report string
        """
        if self.model is None:
            return "No model trained yet"
        
        # Get predictions
        y_pred = self.model.predict(X_val)
        
        # Generate classification report
        report = classification_report(y_val, y_pred, target_names=['Normal', 'Anomaly'])
        
        # Add model information
        model_info = f"""
Maritime Anomaly Detection Training Report
==========================================

Model Type: {self.model.model_name}
Training Configuration:
  - Epochs: {self.epochs}
  - Learning Rate: {self.learning_rate}
  - Cross-validation Folds: {self.cv_folds}
  - Validation Split: {self.validation_split}

Best Score: {self.best_score:.4f}

Classification Report:
{report}

Training History:
{len(self.training_history)} training iterations completed
"""
        
        return model_info


def create_trainer(config: Dict[str, Any]) -> AnomalyDetectionTrainer:
    """
    Create trainer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured trainer
    """
    return AnomalyDetectionTrainer(config) 