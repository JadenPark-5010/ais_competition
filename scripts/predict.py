#!/usr/bin/env python3
"""
Prediction script for maritime anomaly detection system.

This script loads a trained model and makes predictions on new AIS data.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.logging_utils import setup_logging
from data.data_loader import AISDataLoader
from data.preprocessing import AISPreprocessor
from features.feature_engineering import AISFeatureExtractor
from models.base_model import BaseAnomalyDetector

import logging
logger = logging.getLogger(__name__)


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


def load_trained_model(model_path: str) -> BaseAnomalyDetector:
    """
    Load trained model from file.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = BaseAnomalyDetector()
    model.load_model(model_path)
    
    # Load scaler if available
    scaler_path = model_path.parent / f"{model_path.stem}_scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        model.scaler = scaler
        logger.info("Loaded associated scaler")
    else:
        logger.warning("No scaler found - predictions may be inaccurate")
    
    logger.info("Model loaded successfully")
    return model


def load_and_preprocess_data(data_path: str, config: dict) -> pd.DataFrame:
    """
    Load and preprocess AIS data.
    
    Args:
        data_path: Path to data file or directory
        config: Configuration dictionary
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}")
    
    # Initialize data loader
    data_loader = AISDataLoader(config)
    
    # Load data
    data_path = Path(data_path)
    if data_path.is_file():
        df = data_loader.load_csv(data_path)
    elif data_path.is_dir():
        df = data_loader.load_directory(data_path)
    else:
        raise ValueError(f"Invalid data path: {data_path}")
    
    logger.info(f"Loaded {len(df)} records")
    
    # Preprocess data
    preprocessor = AISPreprocessor(config)
    df_processed = preprocessor.preprocess(df)
    
    logger.info(f"Preprocessed data shape: {df_processed.shape}")
    
    return df_processed


def extract_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Extract features from preprocessed AIS data.
    
    Args:
        df: Preprocessed AIS DataFrame
        config: Configuration dictionary
        
    Returns:
        Feature DataFrame
    """
    logger.info("Extracting features")
    
    # Initialize feature extractor
    feature_extractor = AISFeatureExtractor(config)
    
    # Extract features
    features_df = feature_extractor.extract_features(df)
    
    logger.info(f"Extracted features shape: {features_df.shape}")
    
    # Handle missing values
    if features_df.isnull().any().any():
        logger.warning("Found missing values in features, filling with median")
        features_df = features_df.fillna(features_df.median())
    
    return features_df


def make_predictions(model: BaseAnomalyDetector, features_df: pd.DataFrame) -> tuple:
    """
    Make predictions using trained model.
    
    Args:
        model: Trained model
        features_df: Feature DataFrame
        
    Returns:
        Tuple of (predictions, scores, vessel_ids)
    """
    logger.info("Making predictions")
    
    # Extract vessel IDs if present
    vessel_ids = None
    if 'vessel_id' in features_df.columns:
        vessel_ids = features_df['vessel_id'].values
        X = features_df.drop('vessel_id', axis=1).values
    else:
        X = features_df.values
    
    # Scale features if scaler is available
    if hasattr(model, 'scaler') and model.scaler is not None:
        X = model.scaler.transform(X)
        logger.info("Applied feature scaling")
    
    # Make predictions
    predictions = model.predict(X)
    scores = model.predict_proba(X)
    
    # Handle different output formats
    if scores.ndim == 2 and scores.shape[1] == 2:
        scores = scores[:, 1]  # Use anomaly class probability
    elif scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores.flatten()
    
    logger.info(f"Made predictions for {len(predictions)} samples")
    logger.info(f"Predicted anomalies: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
    
    return predictions, scores, vessel_ids


def save_predictions(predictions: np.ndarray, scores: np.ndarray, 
                    vessel_ids: np.ndarray, output_path: str) -> None:
    """
    Save predictions to file.
    
    Args:
        predictions: Binary predictions
        scores: Anomaly scores
        vessel_ids: Vessel identifiers
        output_path: Output file path
    """
    logger.info(f"Saving predictions to {output_path}")
    
    # Create results DataFrame
    results_data = {
        'anomaly_prediction': predictions,
        'anomaly_score': scores
    }
    
    if vessel_ids is not None:
        results_data['vessel_id'] = vessel_ids
    else:
        results_data['sample_id'] = range(len(predictions))
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"Predictions saved successfully")
    
    # Log summary statistics
    logger.info("Prediction Summary:")
    logger.info(f"  Total samples: {len(predictions)}")
    logger.info(f"  Predicted anomalies: {predictions.sum()}")
    logger.info(f"  Anomaly rate: {predictions.mean()*100:.2f}%")
    logger.info(f"  Score statistics:")
    logger.info(f"    Mean: {scores.mean():.4f}")
    logger.info(f"    Std: {scores.std():.4f}")
    logger.info(f"    Min: {scores.min():.4f}")
    logger.info(f"    Max: {scores.max():.4f}")


def generate_prediction_report(predictions: np.ndarray, scores: np.ndarray,
                             vessel_ids: np.ndarray, output_dir: str) -> None:
    """
    Generate detailed prediction report.
    
    Args:
        predictions: Binary predictions
        scores: Anomaly scores
        vessel_ids: Vessel identifiers
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create detailed report
    report = f"""
Maritime Anomaly Detection Prediction Report
===========================================

Prediction Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
------------------
Total Samples: {len(predictions)}
Predicted Anomalies: {predictions.sum()}
Anomaly Rate: {predictions.mean()*100:.2f}%

Score Distribution:
------------------
Mean Score: {scores.mean():.4f}
Standard Deviation: {scores.std():.4f}
Minimum Score: {scores.min():.4f}
Maximum Score: {scores.max():.4f}
Median Score: {np.median(scores):.4f}

Score Percentiles:
-----------------
10th Percentile: {np.percentile(scores, 10):.4f}
25th Percentile: {np.percentile(scores, 25):.4f}
75th Percentile: {np.percentile(scores, 75):.4f}
90th Percentile: {np.percentile(scores, 90):.4f}
95th Percentile: {np.percentile(scores, 95):.4f}
99th Percentile: {np.percentile(scores, 99):.4f}

High-Risk Vessels (Top 10 Anomaly Scores):
==========================================
"""
    
    # Add top anomalous vessels
    if vessel_ids is not None:
        # Get top 10 anomalous vessels
        top_indices = np.argsort(scores)[-10:][::-1]
        
        for i, idx in enumerate(top_indices):
            vessel_id = vessel_ids[idx]
            score = scores[idx]
            prediction = "ANOMALY" if predictions[idx] == 1 else "NORMAL"
            report += f"{i+1:2d}. Vessel {vessel_id}: Score {score:.4f} ({prediction})\n"
    else:
        # Get top 10 anomalous samples
        top_indices = np.argsort(scores)[-10:][::-1]
        
        for i, idx in enumerate(top_indices):
            score = scores[idx]
            prediction = "ANOMALY" if predictions[idx] == 1 else "NORMAL"
            report += f"{i+1:2d}. Sample {idx}: Score {score:.4f} ({prediction})\n"
    
    # Save report
    report_path = output_dir / "prediction_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Prediction report saved to {report_path}")


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Make predictions with maritime anomaly detection model")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to data file or directory")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Output file for predictions")
    parser.add_argument("--output-dir", type=str, default="predictions",
                       help="Output directory for detailed results")
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate detailed prediction report")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(
            log_level="INFO",
            log_dir="logs",
            experiment_name="maritime_prediction"
        )
        
        logger.info("Starting maritime anomaly detection prediction")
        logger.info(f"Arguments: {vars(args)}")
        
        # Load trained model
        model = load_trained_model(args.model)
        
        # Load and preprocess data
        df = load_and_preprocess_data(args.data, config)
        
        # Extract features
        features_df = extract_features(df, config)
        
        # Make predictions
        predictions, scores, vessel_ids = make_predictions(model, features_df)
        
        # Save predictions
        save_predictions(predictions, scores, vessel_ids, args.output)
        
        # Generate detailed report if requested
        if args.generate_report:
            generate_prediction_report(predictions, scores, vessel_ids, args.output_dir)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 