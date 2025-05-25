#!/usr/bin/env python3
"""
Submission script for maritime anomaly detection competition.

This script generates submission files in the required format for competition.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

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
    
    logger.info("Model loaded successfully")
    return model


def process_test_data(test_data_path: str, config: dict) -> pd.DataFrame:
    """
    Process test data for submission.
    
    Args:
        test_data_path: Path to test data
        config: Configuration dictionary
        
    Returns:
        Processed feature DataFrame with vessel IDs
    """
    logger.info(f"Processing test data from {test_data_path}")
    
    # Initialize components
    data_loader = AISDataLoader(config)
    preprocessor = AISPreprocessor(config)
    feature_extractor = AISFeatureExtractor(config)
    
    # Load test data
    test_path = Path(test_data_path)
    if test_path.is_file():
        df = data_loader.load_csv(test_path)
    elif test_path.is_dir():
        df = data_loader.load_directory(test_path)
    else:
        raise ValueError(f"Invalid test data path: {test_path}")
    
    logger.info(f"Loaded {len(df)} test records")
    
    # Store original vessel IDs before preprocessing
    original_vessel_ids = df['MMSI'].unique()
    logger.info(f"Found {len(original_vessel_ids)} unique vessels in test data")
    
    # Preprocess data
    df_processed = preprocessor.preprocess(df)
    logger.info(f"Preprocessed test data shape: {df_processed.shape}")
    
    # Extract features
    features_df = feature_extractor.extract_features(df_processed)
    logger.info(f"Extracted features shape: {features_df.shape}")
    
    # Handle missing values
    if features_df.isnull().any().any():
        logger.warning("Found missing values in features, filling with median")
        features_df = features_df.fillna(features_df.median())
    
    # Ensure all original vessels are represented
    processed_vessel_ids = set(features_df['vessel_id'].values) if 'vessel_id' in features_df.columns else set()
    missing_vessels = set(original_vessel_ids) - processed_vessel_ids
    
    if missing_vessels:
        logger.warning(f"Missing {len(missing_vessels)} vessels after processing: {list(missing_vessels)[:10]}...")
        
        # Create dummy entries for missing vessels with median feature values
        if len(features_df) > 0:
            median_features = features_df.select_dtypes(include=[np.number]).median()
            
            for vessel_id in missing_vessels:
                dummy_row = median_features.copy()
                dummy_row['vessel_id'] = vessel_id
                features_df = pd.concat([features_df, dummy_row.to_frame().T], ignore_index=True)
            
            logger.info(f"Added {len(missing_vessels)} dummy entries for missing vessels")
    
    return features_df


def generate_submission(model: BaseAnomalyDetector, features_df: pd.DataFrame,
                       submission_format: str = "competition") -> pd.DataFrame:
    """
    Generate submission DataFrame.
    
    Args:
        model: Trained model
        features_df: Feature DataFrame
        submission_format: Format for submission ("competition", "detailed")
        
    Returns:
        Submission DataFrame
    """
    logger.info("Generating submission predictions")
    
    # Extract vessel IDs
    if 'vessel_id' in features_df.columns:
        vessel_ids = features_df['vessel_id'].values
        X = features_df.drop('vessel_id', axis=1).values
    else:
        vessel_ids = np.arange(len(features_df))
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
    
    logger.info(f"Generated predictions for {len(predictions)} vessels")
    logger.info(f"Predicted anomalies: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
    
    # Create submission DataFrame based on format
    if submission_format == "competition":
        # Standard competition format: vessel_id, prediction
        submission_df = pd.DataFrame({
            'vessel_id': vessel_ids,
            'anomaly': predictions
        })
    elif submission_format == "detailed":
        # Detailed format with scores and additional info
        submission_df = pd.DataFrame({
            'vessel_id': vessel_ids,
            'anomaly_prediction': predictions,
            'anomaly_score': scores,
            'confidence': np.abs(scores - 0.5) * 2  # Convert to confidence [0,1]
        })
    else:
        raise ValueError(f"Unknown submission format: {submission_format}")
    
    # Sort by vessel_id for consistency
    submission_df = submission_df.sort_values('vessel_id').reset_index(drop=True)
    
    return submission_df


def validate_submission(submission_df: pd.DataFrame, required_vessels: set = None) -> bool:
    """
    Validate submission format and completeness.
    
    Args:
        submission_df: Submission DataFrame
        required_vessels: Set of required vessel IDs
        
    Returns:
        True if valid, raises exception if invalid
    """
    logger.info("Validating submission format")
    
    # Check required columns
    required_columns = ['vessel_id']
    if 'anomaly' in submission_df.columns:
        required_columns.append('anomaly')
    elif 'anomaly_prediction' in submission_df.columns:
        required_columns.append('anomaly_prediction')
    else:
        raise ValueError("Submission must contain 'anomaly' or 'anomaly_prediction' column")
    
    missing_columns = [col for col in required_columns if col not in submission_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if submission_df[required_columns].isnull().any().any():
        raise ValueError("Submission contains missing values in required columns")
    
    # Check prediction values
    pred_col = 'anomaly' if 'anomaly' in submission_df.columns else 'anomaly_prediction'
    unique_predictions = submission_df[pred_col].unique()
    
    if not all(pred in [0, 1] for pred in unique_predictions):
        raise ValueError(f"Predictions must be 0 or 1, found: {unique_predictions}")
    
    # Check vessel ID uniqueness
    if submission_df['vessel_id'].duplicated().any():
        duplicates = submission_df[submission_df['vessel_id'].duplicated()]['vessel_id'].unique()
        raise ValueError(f"Duplicate vessel IDs found: {duplicates[:10]}")
    
    # Check required vessels if provided
    if required_vessels is not None:
        submission_vessels = set(submission_df['vessel_id'].values)
        missing_vessels = required_vessels - submission_vessels
        extra_vessels = submission_vessels - required_vessels
        
        if missing_vessels:
            logger.warning(f"Missing {len(missing_vessels)} required vessels")
        
        if extra_vessels:
            logger.warning(f"Found {len(extra_vessels)} extra vessels not in requirements")
    
    logger.info("Submission validation passed")
    return True


def save_submission(submission_df: pd.DataFrame, output_path: str,
                   include_metadata: bool = True) -> None:
    """
    Save submission to file with metadata.
    
    Args:
        submission_df: Submission DataFrame
        output_path: Output file path
        include_metadata: Whether to include metadata header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving submission to {output_path}")
    
    if include_metadata:
        # Create metadata header
        metadata_lines = [
            f"# Maritime Anomaly Detection Submission",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Total vessels: {len(submission_df)}",
            f"# Predicted anomalies: {submission_df.get('anomaly', submission_df.get('anomaly_prediction', [])).sum()}",
            f"# Anomaly rate: {submission_df.get('anomaly', submission_df.get('anomaly_prediction', [])).mean()*100:.2f}%",
            ""
        ]
        
        # Write metadata and data
        with open(output_path, 'w') as f:
            for line in metadata_lines:
                f.write(line + '\n')
            
            # Write CSV data
            submission_df.to_csv(f, index=False)
    else:
        # Save without metadata
        submission_df.to_csv(output_path, index=False)
    
    logger.info("Submission saved successfully")
    
    # Log summary
    pred_col = 'anomaly' if 'anomaly' in submission_df.columns else 'anomaly_prediction'
    logger.info("Submission Summary:")
    logger.info(f"  Total vessels: {len(submission_df)}")
    logger.info(f"  Predicted anomalies: {submission_df[pred_col].sum()}")
    logger.info(f"  Anomaly rate: {submission_df[pred_col].mean()*100:.2f}%")


def create_submission_package(submission_df: pd.DataFrame, model_path: str,
                            output_dir: str, team_name: str = "Team") -> None:
    """
    Create complete submission package with documentation.
    
    Args:
        submission_df: Submission DataFrame
        model_path: Path to trained model
        output_dir: Output directory
        team_name: Team name for submission
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating submission package in {output_dir}")
    
    # Save main submission file
    submission_path = output_dir / "submission.csv"
    save_submission(submission_df, submission_path, include_metadata=False)
    
    # Create detailed submission with scores if available
    if 'anomaly_score' in submission_df.columns:
        detailed_path = output_dir / "submission_detailed.csv"
        submission_df.to_csv(detailed_path, index=False)
        logger.info(f"Saved detailed submission to {detailed_path}")
    
    # Create submission documentation
    doc_content = f"""
# {team_name} - Maritime Anomaly Detection Submission

## Submission Details
- **Team**: {team_name}
- **Submission Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Vessels**: {len(submission_df)}
- **Predicted Anomalies**: {submission_df.get('anomaly', submission_df.get('anomaly_prediction', [])).sum()}
- **Anomaly Rate**: {submission_df.get('anomaly', submission_df.get('anomaly_prediction', [])).mean()*100:.2f}%

## Model Information
- **Model Path**: {model_path}
- **Model Type**: Maritime Anomaly Detection Ensemble

## Files Included
- `submission.csv`: Main submission file (required format)
- `submission_detailed.csv`: Detailed submission with scores (if available)
- `submission_info.md`: This documentation file

## Methodology
This submission uses an ensemble approach combining:
1. TrAISformer (Transformer-based model)
2. Clustering-based anomaly detection
3. Statistical anomaly detection methods

The model analyzes AIS trajectory data to identify vessels with anomalous behavior patterns.

## Feature Engineering
Features extracted include:
- Kinematic features (speed, acceleration, course changes)
- Geographic features (trajectory complexity, distance metrics)
- Temporal features (time patterns, periodicity)
- Behavioral features (navigation patterns, maneuvers)
- TrAISformer features (four-hot encoding, entropy)

## Contact
For questions about this submission, please contact the team.
"""
    
    doc_path = output_dir / "submission_info.md"
    with open(doc_path, 'w') as f:
        f.write(doc_content)
    
    logger.info(f"Created submission package with {len(list(output_dir.glob('*')))} files")


def main():
    """Main submission function."""
    parser = argparse.ArgumentParser(description="Generate submission for maritime anomaly detection competition")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Path to test data file or directory")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output", type=str, default="submission.csv",
                       help="Output submission file")
    parser.add_argument("--output-dir", type=str, default="submissions",
                       help="Output directory for submission package")
    parser.add_argument("--format", type=str, default="competition",
                       choices=["competition", "detailed"],
                       help="Submission format")
    parser.add_argument("--team-name", type=str, default="Maritime AI Team",
                       help="Team name for submission")
    parser.add_argument("--create-package", action="store_true",
                       help="Create complete submission package")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(
            log_level="INFO",
            log_dir="logs",
            experiment_name="maritime_submission"
        )
        
        logger.info("Starting maritime anomaly detection submission generation")
        logger.info(f"Arguments: {vars(args)}")
        
        # Load trained model
        model = load_trained_model(args.model)
        
        # Process test data
        features_df = process_test_data(args.test_data, config)
        
        # Generate submission
        submission_df = generate_submission(model, features_df, args.format)
        
        # Validate submission
        validate_submission(submission_df)
        
        # Save submission
        if args.create_package:
            create_submission_package(submission_df, args.model, args.output_dir, args.team_name)
        else:
            save_submission(submission_df, args.output)
        
        logger.info("Submission generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 