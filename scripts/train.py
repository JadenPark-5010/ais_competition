#!/usr/bin/env python3
"""
Training script for maritime anomaly detection system.

This script orchestrates the entire training pipeline including:
- Data loading and preprocessing
- Feature engineering
- Model training
- Evaluation and validation
- Model saving
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.logging_utils import ExperimentLogger
from utils.metrics import AnomalyDetectionMetrics, plot_confusion_matrix, plot_roc_curve
from features.feature_engineering import AISFeatureExtractor
from models.ensemble_model import create_ensemble_model
from models.traisformer import create_traisformer_model
from models.clustering_model import create_clustering_model


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str, logger: ExperimentLogger) -> pd.DataFrame:
    """
    Load AIS data from CSV file.
    
    Args:
        data_path: Path to data file
        logger: Experiment logger
        
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    logger.info(f"Loaded data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Basic data validation
    required_columns = ['Timestamp', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'COG']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for empty data
    if len(df) == 0:
        raise ValueError("Data file is empty")
    
    logger.info("Data loaded successfully")
    return df


def create_synthetic_labels(df: pd.DataFrame, anomaly_ratio: float = 0.1) -> np.ndarray:
    """
    Create synthetic anomaly labels for demonstration.
    
    In a real scenario, you would have actual labels or use unsupervised methods.
    
    Args:
        df: Input DataFrame
        anomaly_ratio: Ratio of anomalies to inject
        
    Returns:
        Binary labels (0: normal, 1: anomaly)
    """
    n_samples = len(df)
    n_anomalies = int(n_samples * anomaly_ratio)
    
    # Create labels (mostly normal)
    labels = np.zeros(n_samples, dtype=int)
    
    # Randomly assign some samples as anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_indices] = 1
    
    return labels


def prepare_features(df: pd.DataFrame, config: dict, logger: ExperimentLogger) -> pd.DataFrame:
    """
    Extract features from AIS data.
    
    Args:
        df: Input AIS DataFrame
        config: Configuration dictionary
        logger: Experiment logger
        
    Returns:
        Feature DataFrame
    """
    logger.info("Starting feature extraction")
    
    # Initialize feature extractor
    feature_extractor = AISFeatureExtractor(config)
    
    # Extract features
    features_df = feature_extractor.extract_features(df)
    
    logger.info(f"Feature extraction completed. Shape: {features_df.shape}")
    logger.info(f"Features: {list(features_df.columns)}")
    
    # Handle missing values
    if features_df.isnull().any().any():
        logger.warning("Found missing values in features, filling with median")
        features_df = features_df.fillna(features_df.median())
    
    # Remove non-numeric columns except vessel_id
    numeric_columns = features_df.select_dtypes(include=[np.number]).columns
    if 'vessel_id' in features_df.columns:
        features_df = features_df[['vessel_id'] + list(numeric_columns)]
    else:
        features_df = features_df[numeric_columns]
    
    return features_df


def split_data(features_df: pd.DataFrame, labels: np.ndarray, config: dict, logger: ExperimentLogger):
    """
    Split data into training and validation sets.
    
    Args:
        features_df: Feature DataFrame
        labels: Target labels
        config: Configuration dictionary
        logger: Experiment logger
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    logger.info("Splitting data into train/validation sets")
    
    # Remove vessel_id if present
    if 'vessel_id' in features_df.columns:
        X = features_df.drop('vessel_id', axis=1).values
    else:
        X = features_df.values
    
    # Split configuration
    validation_split = config.get('data', {}).get('validation_split', 0.2)
    random_state = 42
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, 
        test_size=validation_split,
        random_state=random_state,
        stratify=labels
    )
    
    logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    logger.info(f"Training labels distribution: {np.bincount(y_train)}")
    logger.info(f"Validation labels distribution: {np.bincount(y_val)}")
    
    return X_train, X_val, y_train, y_val


def train_model(X_train: np.ndarray, y_train: np.ndarray, config: dict, 
                model_type: str, logger: ExperimentLogger):
    """
    Train the specified model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary
        model_type: Type of model to train
        logger: Experiment logger
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_type} model")
    
    # Create model based on type
    if model_type == 'ensemble':
        model = create_ensemble_model(config, ensemble_type='weighted')
    elif model_type == 'traisformer':
        model = create_traisformer_model(config)
    elif model_type == 'clustering':
        model = create_clustering_model(config, model_type='combined')
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Log model information
    logger.log_model(model)
    
    # Train model
    model.fit(X_train, y_train)
    
    logger.info(f"{model_type} model training completed")
    
    return model


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, 
                  logger: ExperimentLogger, save_plots: bool = True):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        logger: Experiment logger
        save_plots: Whether to save evaluation plots
        
    Returns:
        Evaluation metrics dictionary
    """
    logger.info("Evaluating model performance")
    
    # Get predictions
    y_pred = model.predict(X_val)
    y_scores = model.predict_proba(X_val)
    
    # Handle different output formats
    if y_scores.ndim == 2 and y_scores.shape[1] == 2:
        y_scores = y_scores[:, 1]
    elif y_scores.ndim == 2 and y_scores.shape[1] == 1:
        y_scores = y_scores.flatten()
    
    # Calculate metrics
    metrics_calculator = AnomalyDetectionMetrics()
    metrics = metrics_calculator.calculate_metrics(y_val, y_pred, y_scores)
    
    # Log metrics
    logger.info("Validation Results:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")
    
    # Generate and save plots
    if save_plots:
        try:
            # Create results directory
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Confusion matrix
            cm_fig = plot_confusion_matrix(y_val, y_pred)
            cm_fig.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
            logger.info("Confusion matrix saved")
            
            # ROC curve
            roc_fig = plot_roc_curve(y_val, y_scores)
            roc_fig.savefig(results_dir / "roc_curve.png", dpi=300, bbox_inches='tight')
            logger.info("ROC curve saved")
            
        except Exception as e:
            logger.warning(f"Failed to save plots: {e}")
    
    return metrics


def save_model(model, config: dict, metrics: dict, logger: ExperimentLogger):
    """
    Save trained model and metadata.
    
    Args:
        model: Trained model
        config: Configuration dictionary
        metrics: Evaluation metrics
        logger: Experiment logger
    """
    logger.info("Saving model")
    
    # Create models directory
    models_dir = Path(config.get('output', {}).get('model_dir', 'models'))
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "final_model.pkl"
    model.save_model(model_path)
    
    # Save scaler if used
    if hasattr(model, 'scaler'):
        scaler_path = models_dir / "scaler.pkl"
        joblib.dump(model.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    
    # Save metadata
    metadata = {
        'model_type': model.model_name,
        'config': config,
        'metrics': metrics,
        'feature_names': getattr(model, 'feature_names', None)
    }
    
    metadata_path = models_dir / "model_metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Metadata saved to {metadata_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train maritime anomaly detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data CSV file")
    parser.add_argument("--model", type=str, default="ensemble",
                       choices=["ensemble", "traisformer", "clustering"],
                       help="Type of model to train")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Update output directory
        if args.output_dir:
            config.setdefault('output', {})['results_dir'] = args.output_dir
        
        # Initialize logger
        logger = ExperimentLogger(config)
        logger.log_start()
        
        # Log arguments
        logger.info(f"Arguments: {vars(args)}")
        
        # Load data
        df = load_data(args.data, logger)
        
        # Create synthetic labels (replace with real labels in production)
        logger.warning("Using synthetic labels for demonstration. Replace with real labels in production.")
        labels = create_synthetic_labels(df, anomaly_ratio=0.1)
        
        # Extract features
        features_df = prepare_features(df, config, logger)
        
        # Split data
        X_train, X_val, y_train, y_val = split_data(features_df, labels, config, logger)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        logger.info("Feature scaling completed")
        
        # Train model
        model = train_model(X_train_scaled, y_train, config, args.model, logger)
        
        # Store scaler in model for later use
        model.scaler = scaler
        
        # Evaluate model
        metrics = evaluate_model(model, X_val_scaled, y_val, logger)
        
        # Save model
        save_model(model, config, metrics, logger)
        
        # Log completion
        logger.log_end(metrics)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 