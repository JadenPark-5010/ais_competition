#!/usr/bin/env python3
"""
Demo script for maritime anomaly detection system.

This script demonstrates the complete pipeline with sample data.
"""

import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.logging_utils import setup_logging
from data.data_loader import AISDataLoader
from data.preprocessing import AISPreprocessor
from features.feature_engineering import AISFeatureExtractor
from training.trainer import AnomalyDetectionTrainer
from training.validator import ModelValidator

import logging
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load default configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration if file doesn't exist
        config = {
            'data': {
                'validation_split': 0.2,
                'random_state': 42
            },
            'preprocessing': {
                'normalize': True,
                'handle_missing': 'interpolate',
                'outlier_detection': True,
                'outlier_threshold': 3.0
            },
            'features': {
                'kinematic_features': True,
                'geographic_features': True,
                'temporal_features': True,
                'behavioral_features': True,
                'traisformer_features': True
            },
            'training': {
                'epochs': 50,
                'learning_rate': 0.001,
                'patience': 10
            },
            'validation': {
                'cross_validation': 3,
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
            }
        }
    
    return config


def generate_sample_data(n_vessels: int = 50, n_points_per_vessel: int = 30) -> pd.DataFrame:
    """
    Generate sample AIS data for demonstration.
    
    Args:
        n_vessels: Number of vessels to generate
        n_points_per_vessel: Number of data points per vessel
        
    Returns:
        Sample AIS DataFrame
    """
    logger.info(f"Generating sample data: {n_vessels} vessels, {n_points_per_vessel} points each")
    
    config = load_config()
    data_loader = AISDataLoader(config)
    
    # Generate sample data
    df = data_loader.load_sample_data(n_vessels, n_points_per_vessel)
    
    logger.info(f"Generated sample data with shape: {df.shape}")
    
    return df


def create_synthetic_anomalies(df: pd.DataFrame, anomaly_rate: float = 0.1) -> tuple:
    """
    Create synthetic anomalies in the data for demonstration.
    
    Args:
        df: Input DataFrame
        anomaly_rate: Fraction of vessels to make anomalous
        
    Returns:
        Tuple of (modified_df, labels)
    """
    logger.info(f"Creating synthetic anomalies with rate: {anomaly_rate}")
    
    df = df.copy()
    vessels = df['MMSI'].unique()
    n_anomalous = int(len(vessels) * anomaly_rate)
    
    # Randomly select vessels to make anomalous
    np.random.seed(42)
    anomalous_vessels = np.random.choice(vessels, n_anomalous, replace=False)
    
    # Create labels
    labels = []
    
    for _, row in df.iterrows():
        if row['MMSI'] in anomalous_vessels:
            labels.append(1)  # Anomaly
        else:
            labels.append(0)  # Normal
    
    # Modify anomalous vessels to have unusual patterns
    for vessel_mmsi in anomalous_vessels:
        vessel_mask = df['MMSI'] == vessel_mmsi
        
        # Add random anomalous behaviors
        anomaly_type = np.random.choice(['speed', 'course', 'position'])
        
        if anomaly_type == 'speed':
            # Unusual speed patterns
            df.loc[vessel_mask, 'SOG'] *= np.random.uniform(2.0, 5.0)
        elif anomaly_type == 'course':
            # Erratic course changes
            df.loc[vessel_mask, 'COG'] += np.random.uniform(-180, 180)
            df.loc[vessel_mask, 'COG'] = df.loc[vessel_mask, 'COG'] % 360
        elif anomaly_type == 'position':
            # Position jumps
            df.loc[vessel_mask, 'Latitude'] += np.random.uniform(-0.1, 0.1)
            df.loc[vessel_mask, 'Longitude'] += np.random.uniform(-0.1, 0.1)
    
    labels = np.array(labels)
    
    logger.info(f"Created {n_anomalous} anomalous vessels ({labels.sum()} anomalous points)")
    
    return df, labels


def run_complete_pipeline(df: pd.DataFrame, labels: np.ndarray, config: dict) -> dict:
    """
    Run the complete anomaly detection pipeline.
    
    Args:
        df: Input AIS DataFrame
        labels: True labels for evaluation
        config: Configuration dictionary
        
    Returns:
        Results dictionary
    """
    logger.info("Running complete anomaly detection pipeline")
    
    # Step 1: Preprocessing
    logger.info("Step 1: Data preprocessing")
    preprocessor = AISPreprocessor(config)
    df_processed = preprocessor.preprocess(df)
    
    # Step 2: Feature extraction
    logger.info("Step 2: Feature extraction")
    feature_extractor = AISFeatureExtractor(config)
    features_df = feature_extractor.extract_features(df_processed)
    
    # Handle missing values
    if features_df.isnull().any().any():
        logger.warning("Found missing values in features, filling with median")
        features_df = features_df.fillna(features_df.median())
    
    # Prepare features and labels
    if 'vessel_id' in features_df.columns:
        X = features_df.drop('vessel_id', axis=1).values
    else:
        X = features_df.values
    
    # Aggregate labels by vessel (take majority vote)
    vessel_labels = {}
    for i, vessel_id in enumerate(df_processed['MMSI'].unique()):
        vessel_mask = df_processed['MMSI'] == vessel_id
        vessel_label_votes = labels[df['MMSI'] == vessel_id]
        vessel_labels[vessel_id] = int(vessel_label_votes.mean() > 0.5)
    
    y = np.array([vessel_labels.get(vid, 0) for vid in features_df.get('vessel_id', range(len(features_df)))])
    
    logger.info(f"Prepared features: {X.shape}, labels: {y.shape}")
    logger.info(f"Label distribution: {np.bincount(y)}")
    
    # Step 3: Model training
    logger.info("Step 3: Model training")
    trainer = AnomalyDetectionTrainer(config)
    
    # Train different model types
    results = {}
    
    for model_type in ['clustering', 'ensemble']:
        try:
            logger.info(f"Training {model_type} model")
            
            # Train model
            model = trainer.train(X, y, model_type=model_type)
            
            # Evaluate model
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale test data
            if hasattr(model, 'scaler') and model.scaler is not None:
                X_test_scaled = model.scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            
            # Get predictions
            y_pred = model.predict(X_test_scaled)
            y_scores = model.predict_proba(X_test_scaled)
            
            # Handle different output formats
            if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                y_scores = y_scores[:, 1]
            elif y_scores.ndim == 2 and y_scores.shape[1] == 1:
                y_scores = y_scores.flatten()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_scores) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            results[model_type] = {
                'model': model,
                'metrics': metrics,
                'predictions': y_pred,
                'scores': y_scores,
                'test_labels': y_test
            }
            
            logger.info(f"{model_type} model results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
                
        except Exception as e:
            logger.error(f"Failed to train {model_type} model: {e}")
            continue
    
    return results


def print_demo_summary(results: dict) -> None:
    """
    Print a summary of the demo results.
    
    Args:
        results: Results dictionary from pipeline
    """
    print("\n" + "="*60)
    print("MARITIME ANOMALY DETECTION DEMO SUMMARY")
    print("="*60)
    
    if not results:
        print("No models were successfully trained.")
        return
    
    print(f"\nSuccessfully trained {len(results)} models:")
    
    # Create comparison table
    print(f"\n{'Model':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 70)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<15} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} "
              f"{metrics['f1_score']:<10.4f} "
              f"{metrics['roc_auc']:<10.4f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    best_f1 = results[best_model]['metrics']['f1_score']
    
    print(f"\nBest performing model: {best_model} (F1-Score: {best_f1:.4f})")
    
    print("\nDemo completed successfully!")
    print("The system is ready for training on real AIS data.")
    print("\nNext steps:")
    print("1. Prepare your AIS dataset in the required format")
    print("2. Run: python scripts/train.py --data your_data.csv")
    print("3. Make predictions: python scripts/predict.py --model models/trained_model.pkl --data test_data.csv")
    print("4. Generate submission: python scripts/submit.py --model models/trained_model.pkl --test-data test_data.csv")


def main():
    """Main demo function."""
    print("Maritime Anomaly Detection System Demo")
    print("=====================================")
    
    try:
        # Setup logging
        setup_logging(
            log_level="INFO",
            log_dir="logs",
            experiment_name="maritime_demo"
        )
        
        logger.info("Starting maritime anomaly detection demo")
        
        # Load configuration
        config = load_config()
        logger.info("Loaded configuration")
        
        # Generate sample data
        df = generate_sample_data(n_vessels=30, n_points_per_vessel=20)
        
        # Create synthetic anomalies
        df_with_anomalies, labels = create_synthetic_anomalies(df, anomaly_rate=0.15)
        
        # Run complete pipeline
        results = run_complete_pipeline(df_with_anomalies, labels, config)
        
        # Print summary
        print_demo_summary(results)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nDemo failed with error: {e}")
        print("Please check the logs for more details.")


if __name__ == "__main__":
    main() 