"""
Advanced Ensemble Training Script for Maritime Anomaly Detection

This script trains and evaluates an ensemble model combining TrAISformer
with traditional ML models for robust anomaly detection.
"""

import numpy as np
import pandas as pd
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.advanced_ensemble import AdvancedEnsembleDetector
from data.data_loader import AISDataLoader
from features.feature_engineering import FeatureEngineer
from utils.visualization import plot_ensemble_results, plot_model_comparison
from utils.metrics import calculate_ensemble_metrics

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleTrainer:
    """
    Comprehensive trainer for the advanced ensemble model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ensemble trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ensemble_model = None
        self.feature_engineer = None
        self.results = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize feature engineer and ensemble model."""
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config.get('features', {}))
        
        # Initialize ensemble model
        self.ensemble_model = AdvancedEnsembleDetector(self.config)
        
        logger.info("Ensemble trainer initialized")
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for training.
        
        Args:
            data_path: Path to the data directory
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Loading and preparing data...")
        
        # Load data
        data_loader = AISDataLoader(self.config['data'])
        df = data_loader.load_data(data_path)
        
        # Engineer features
        logger.info("Engineering features...")
        df_features = self.feature_engineer.engineer_features(df)
        
        # Split data
        train_df, temp_df = train_test_split(
            df_features, 
            test_size=0.3, 
            random_state=42,
            stratify=df_features['is_suspicious'] if 'is_suspicious' in df_features.columns else None
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df['is_suspicious'] if 'is_suspicious' in temp_df.columns else None
        )
        
        logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def train_ensemble(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
        """
        Train the ensemble model.
        
        Args:
            train_df: Training data
            val_df: Validation data
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting ensemble training...")
        
        # Prepare training data
        X_train = train_df.drop(['is_suspicious'], axis=1, errors='ignore')
        y_train = train_df['is_suspicious'] if 'is_suspicious' in train_df.columns else np.zeros(len(train_df))
        
        # Train ensemble
        self.ensemble_model.fit(X_train.values, y_train.values)
        
        # Validate
        X_val = val_df.drop(['is_suspicious'], axis=1, errors='ignore')
        y_val = val_df['is_suspicious'] if 'is_suspicious' in val_df.columns else np.zeros(len(val_df))
        
        val_metrics = self.ensemble_model.evaluate(X_val.values, y_val.values)
        
        logger.info("Validation Results:")
        for metric, value in val_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return val_metrics
    
    def evaluate_ensemble(self, test_df: pd.DataFrame) -> Dict:
        """
        Evaluate the ensemble model on test data.
        
        Args:
            test_df: Test data
            
        Returns:
            Test results dictionary
        """
        logger.info("Evaluating ensemble model...")
        
        X_test = test_df.drop(['is_suspicious'], axis=1, errors='ignore')
        y_test = test_df['is_suspicious'] if 'is_suspicious' in test_df.columns else np.zeros(len(test_df))
        
        # Get predictions
        predictions = self.ensemble_model.predict(X_test.values)
        probabilities = self.ensemble_model.predict_proba(X_test.values)[:, 1]
        
        # Calculate comprehensive metrics
        test_metrics = calculate_ensemble_metrics(y_test.values, predictions, probabilities)
        
        # Store results
        self.results['test_metrics'] = test_metrics
        self.results['predictions'] = predictions
        self.results['probabilities'] = probabilities
        self.results['true_labels'] = y_test.values
        
        logger.info("Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return test_metrics
    
    def analyze_model_performance(self) -> Dict:
        """
        Analyze individual model performance within the ensemble.
        
        Returns:
            Model analysis results
        """
        logger.info("Analyzing individual model performance...")
        
        # Get feature importance from all models
        feature_importance = self.ensemble_model.get_feature_importance()
        
        # Analyze model contributions
        model_analysis = {
            'feature_importance': feature_importance,
            'model_weights': getattr(self.ensemble_model, 'model_weights', None),
            'meta_learner_importance': feature_importance.get('meta_learner', {})
        }
        
        self.results['model_analysis'] = model_analysis
        
        return model_analysis
    
    def cross_validate_ensemble(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on the ensemble model.
        
        Args:
            df: Full dataset
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        X = df.drop(['is_suspicious'], axis=1, errors='ignore').values
        y = df['is_suspicious'].values if 'is_suspicious' in df.columns else np.zeros(len(df))
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc': [],
            'average_precision': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create new ensemble for this fold
            fold_ensemble = AdvancedEnsembleDetector(self.config)
            fold_ensemble.fit(X_train, y_train)
            
            # Evaluate
            fold_metrics = fold_ensemble.evaluate(X_val, y_val)
            
            for metric, value in fold_metrics.items():
                if metric in cv_scores:
                    cv_scores[metric].append(value)
        
        # Calculate mean and std
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        logger.info("Cross-validation Results:")
        for metric, value in cv_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        self.results['cv_results'] = cv_results
        
        return cv_results
    
    def generate_visualizations(self, output_dir: Path):
        """
        Generate comprehensive visualizations.
        
        Args:
            output_dir: Output directory for visualizations
        """
        logger.info("Generating visualizations...")
        
        # Create visualization directory
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        if 'test_metrics' not in self.results:
            logger.warning("No test results available for visualization")
            return
        
        y_true = self.results['true_labels']
        y_pred = self.results['predictions']
        y_prob = self.results['probabilities']
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Ensemble Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = self.results['test_metrics']['roc_auc']
        plt.plot(fpr, tpr, label=f'Ensemble (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(viz_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_score = self.results['test_metrics']['average_precision']
        plt.plot(recall, precision, label=f'Ensemble (AP = {ap_score:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(viz_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature Importance (if available)
        if 'model_analysis' in self.results:
            feature_importance = self.results['model_analysis']['feature_importance']
            
            if 'meta_learner' in feature_importance:
                plt.figure(figsize=(10, 6))
                meta_importance = feature_importance['meta_learner']
                models = list(meta_importance.keys())
                importances = list(meta_importance.values())
                
                plt.bar(models, importances)
                plt.title('Meta-Learner Model Importance')
                plt.xlabel('Models')
                plt.ylabel('Importance')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(viz_dir / 'meta_learner_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # 5. Probability Distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(y_prob[y_true == 0], bins=50, alpha=0.7, label='Normal', density=True)
        plt.hist(y_prob[y_true == 1], bins=50, alpha=0.7, label='Anomaly', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution by Class')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot([y_prob[y_true == 0], y_prob[y_true == 1]], labels=['Normal', 'Anomaly'])
        plt.ylabel('Predicted Probability')
        plt.title('Probability Distribution Boxplot')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def save_results(self, output_dir: Path):
        """
        Save all results and models.
        
        Args:
            output_dir: Output directory
        """
        logger.info("Saving results...")
        
        # Save ensemble model
        model_path = output_dir / 'ensemble_model.joblib'
        self.ensemble_model.save_model(str(model_path))
        
        # Save feature engineer
        feature_engineer_path = output_dir / 'feature_engineer.joblib'
        joblib.dump(self.feature_engineer, feature_engineer_path)
        
        # Save results
        results_path = output_dir / 'results.yaml'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for YAML serialization
            yaml_results = {}
            for key, value in self.results.items():
                if isinstance(value, np.ndarray):
                    yaml_results[key] = value.tolist()
                elif isinstance(value, dict):
                    yaml_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                       for k, v in value.items()}
                else:
                    yaml_results[key] = value
            
            yaml.dump(yaml_results, f, default_flow_style=False)
        
        # Save configuration
        config_path = output_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_submission(self, test_df: pd.DataFrame, output_dir: Path):
        """
        Generate submission file for competition.
        
        Args:
            test_df: Test data
            output_dir: Output directory
        """
        logger.info("Generating submission file...")
        
        # Prepare test data
        X_test = test_df.drop(['is_suspicious'], axis=1, errors='ignore')
        
        # Get predictions
        probabilities = self.ensemble_model.predict_proba(X_test.values)[:, 1]
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': range(len(test_df)),
            'probability': probabilities
        })
        
        # Save submission
        submission_path = output_dir / 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        logger.info(f"Submission file saved to {submission_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train advanced ensemble for maritime anomaly detection')
    parser.add_argument('--config', type=str, default='config/ensemble_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='outputs/ensemble/',
                       help='Output directory')
    parser.add_argument('--cv', action='store_true',
                       help='Perform cross-validation')
    parser.add_argument('--submission', action='store_true',
                       help='Generate submission file')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = EnsembleTrainer(config)
    
    # Load and prepare data
    train_df, val_df, test_df = trainer.load_and_prepare_data(args.data)
    
    # Train ensemble
    val_metrics = trainer.train_ensemble(train_df, val_df)
    
    # Evaluate ensemble
    test_metrics = trainer.evaluate_ensemble(test_df)
    
    # Analyze model performance
    model_analysis = trainer.analyze_model_performance()
    
    # Cross-validation (optional)
    if args.cv:
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        cv_results = trainer.cross_validate_ensemble(full_df)
    
    # Generate visualizations
    trainer.generate_visualizations(output_dir)
    
    # Save results
    trainer.save_results(output_dir)
    
    # Generate submission (optional)
    if args.submission:
        trainer.generate_submission(test_df, output_dir)
    
    # Print final summary
    logger.info("=" * 50)
    logger.info("ENSEMBLE TRAINING COMPLETED")
    logger.info("=" * 50)
    logger.info("Final Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    if args.cv and 'cv_results' in trainer.results:
        logger.info("\nCross-Validation Results:")
        cv_results = trainer.results['cv_results']
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in cv_results:
                logger.info(f"  {metric}: {cv_results[mean_key]:.4f} ± {cv_results[std_key]:.4f}")
    
    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main() 