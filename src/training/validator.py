"""
Model validator for maritime anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

from ..models.base_model import BaseAnomalyDetector
from ..utils.metrics import (
    AnomalyDetectionMetrics, plot_confusion_matrix, plot_roc_curve,
    plot_precision_recall_curve, plot_threshold_metrics
)

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Model validator for comprehensive evaluation of anomaly detection models.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.validation_config = self.config.get('validation', {})
        
        # Validation parameters
        self.cv_folds = self.validation_config.get('cross_validation', 5)
        self.metrics = self.validation_config.get('metrics', 
                                                 ['accuracy', 'precision', 'recall', 'f1', 'auc'])
        
        # Initialize metrics calculator
        self.metrics_calculator = AnomalyDetectionMetrics()
        
        # Results storage
        self.validation_results = {}
        self.cross_validation_results = {}
        
        logger.info("Model Validator initialized")
    
    def validate_single_model(self, model: BaseAnomalyDetector,
                             X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str = "model") -> Dict[str, Any]:
        """
        Validate a single model on test data.
        
        Args:
            model: Trained model to validate
            X_test: Test features
            y_test: Test labels
            model_name: Name for the model
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating model: {model_name}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)
        
        # Handle different output formats
        if y_scores.ndim == 2 and y_scores.shape[1] == 2:
            y_scores = y_scores[:, 1]
        elif y_scores.ndim == 2 and y_scores.shape[1] == 1:
            y_scores = y_scores.flatten()
        
        # Calculate comprehensive metrics
        metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred, y_scores)
        
        # Find optimal threshold
        optimal_threshold, optimal_f1 = self.metrics_calculator.find_optimal_threshold(
            y_test, y_scores, metric='f1_score'
        )
        
        # Calculate detection rate at low FPR
        detection_rate_1pct = self.metrics_calculator.calculate_detection_rate_at_fpr(
            y_test, y_scores, target_fpr=0.01
        )
        
        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'optimal_threshold': optimal_threshold,
            'optimal_f1_score': optimal_f1,
            'detection_rate_at_1pct_fpr': detection_rate_1pct,
            'predictions': y_pred,
            'scores': y_scores,
            'true_labels': y_test
        }
        
        self.validation_results[model_name] = results
        
        # Log key metrics
        logger.info(f"Validation results for {model_name}:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric_name in metrics:
                logger.info(f"  {metric_name}: {metrics[metric_name]:.4f}")
        
        return results
    
    def cross_validate_model(self, model: BaseAnomalyDetector,
                            X: np.ndarray, y: np.ndarray,
                            model_name: str = "model") -> Dict[str, Any]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Feature matrix
            y: Target labels
            model_name: Name for the model
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Cross-validating model: {model_name}")
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Collect results for each fold
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            logger.info(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh model instance for this fold
            fold_model = type(model)(model.config)
            
            # Train on fold
            try:
                if hasattr(fold_model, 'fit'):
                    if hasattr(model, 'classes_'):  # Supervised model
                        fold_model.fit(X_train, y_train)
                    else:  # Unsupervised model
                        fold_model.fit(X_train)
                
                # Validate on fold
                fold_metrics = self.validate_single_model(
                    fold_model, X_val, y_val, f"{model_name}_fold_{fold+1}"
                )
                
                fold_results.append(fold_metrics['metrics'])
                
            except Exception as e:
                logger.error(f"Error in fold {fold + 1}: {e}")
                continue
        
        if not fold_results:
            raise ValueError("All folds failed during cross-validation")
        
        # Calculate statistics across folds
        cv_stats = {}
        for metric_name in fold_results[0].keys():
            values = [result[metric_name] for result in fold_results]
            cv_stats[f'{metric_name}_mean'] = np.mean(values)
            cv_stats[f'{metric_name}_std'] = np.std(values)
            cv_stats[f'{metric_name}_min'] = np.min(values)
            cv_stats[f'{metric_name}_max'] = np.max(values)
        
        results = {
            'model_name': model_name,
            'cv_folds': self.cv_folds,
            'fold_results': fold_results,
            'cv_statistics': cv_stats
        }
        
        self.cross_validation_results[model_name] = results
        
        # Log cross-validation summary
        logger.info(f"Cross-validation results for {model_name}:")
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            mean_key = f'{metric_name}_mean'
            std_key = f'{metric_name}_std'
            if mean_key in cv_stats:
                logger.info(f"  {metric_name}: {cv_stats[mean_key]:.4f} ± {cv_stats[std_key]:.4f}")
        
        return results
    
    def compare_models(self, models: Dict[str, BaseAnomalyDetector],
                      X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models on the same test set.
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Comparison results DataFrame
        """
        logger.info(f"Comparing {len(models)} models")
        
        comparison_results = []
        
        for model_name, model in models.items():
            try:
                results = self.validate_single_model(model, X_test, y_test, model_name)
                
                # Extract key metrics for comparison
                metrics = results['metrics']
                comparison_row = {
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0),
                    'Optimal_Threshold': results.get('optimal_threshold', 0.5),
                    'Detection_Rate_1%_FPR': results.get('detection_rate_at_1pct_fpr', 0)
                }
                
                comparison_results.append(comparison_row)
                
            except Exception as e:
                logger.error(f"Error validating model {model_name}: {e}")
                continue
        
        if not comparison_results:
            raise ValueError("No models could be validated successfully")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        logger.info("Model comparison completed")
        logger.info(f"Best model by F1-Score: {comparison_df.iloc[0]['Model']}")
        
        return comparison_df
    
    def generate_validation_plots(self, model_name: str, 
                                 output_dir: str = "validation_plots") -> None:
        """
        Generate validation plots for a model.
        
        Args:
            model_name: Name of the model to plot
            output_dir: Directory to save plots
        """
        if model_name not in self.validation_results:
            raise ValueError(f"No validation results found for model: {model_name}")
        
        results = self.validation_results[model_name]
        y_true = results['true_labels']
        y_pred = results['predictions']
        y_scores = results['scores']
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating validation plots for {model_name}")
        
        # 1. Confusion Matrix
        try:
            cm_fig = plot_confusion_matrix(y_true, y_pred, title=f'Confusion Matrix - {model_name}')
            cm_fig.savefig(output_path / f'{model_name}_confusion_matrix.png', 
                          dpi=300, bbox_inches='tight')
            plt.close(cm_fig)
            logger.info(f"Saved confusion matrix plot")
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix: {e}")
        
        # 2. ROC Curve
        try:
            roc_fig = plot_roc_curve(y_true, y_scores, title=f'ROC Curve - {model_name}')
            roc_fig.savefig(output_path / f'{model_name}_roc_curve.png', 
                           dpi=300, bbox_inches='tight')
            plt.close(roc_fig)
            logger.info(f"Saved ROC curve plot")
        except Exception as e:
            logger.warning(f"Failed to generate ROC curve: {e}")
        
        # 3. Precision-Recall Curve
        try:
            pr_fig = plot_precision_recall_curve(y_true, y_scores, 
                                                title=f'Precision-Recall Curve - {model_name}')
            pr_fig.savefig(output_path / f'{model_name}_pr_curve.png', 
                          dpi=300, bbox_inches='tight')
            plt.close(pr_fig)
            logger.info(f"Saved precision-recall curve plot")
        except Exception as e:
            logger.warning(f"Failed to generate PR curve: {e}")
        
        # 4. Threshold Analysis
        try:
            threshold_fig = plot_threshold_metrics(y_true, y_scores,
                                                  title=f'Threshold Analysis - {model_name}')
            threshold_fig.savefig(output_path / f'{model_name}_threshold_analysis.png', 
                                 dpi=300, bbox_inches='tight')
            plt.close(threshold_fig)
            logger.info(f"Saved threshold analysis plot")
        except Exception as e:
            logger.warning(f"Failed to generate threshold analysis: {e}")
        
        # 5. Score Distribution
        try:
            self._plot_score_distribution(y_true, y_scores, model_name, output_path)
            logger.info(f"Saved score distribution plot")
        except Exception as e:
            logger.warning(f"Failed to generate score distribution: {e}")
    
    def _plot_score_distribution(self, y_true: np.ndarray, y_scores: np.ndarray,
                                model_name: str, output_path: Path) -> None:
        """
        Plot distribution of anomaly scores by class.
        
        Args:
            y_true: True labels
            y_scores: Predicted scores
            model_name: Model name
            output_path: Output directory path
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Separate scores by class
        normal_scores = y_scores[y_true == 0]
        anomaly_scores = y_scores[y_true == 1]
        
        # Plot histograms
        ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        
        # Add optimal threshold line
        if model_name in self.validation_results:
            optimal_threshold = self.validation_results[model_name]['optimal_threshold']
            ax.axvline(optimal_threshold, color='green', linestyle='--', 
                      label=f'Optimal Threshold: {optimal_threshold:.3f}')
        
        ax.set_xlabel('Anomaly Score')
        ax.set_ylabel('Density')
        ax.set_title(f'Score Distribution - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.savefig(output_path / f'{model_name}_score_distribution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def generate_validation_report(self, model_name: str) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Validation report string
        """
        if model_name not in self.validation_results:
            return f"No validation results found for model: {model_name}"
        
        results = self.validation_results[model_name]
        metrics = results['metrics']
        
        # Generate classification report
        y_true = results['true_labels']
        y_pred = results['predictions']
        class_report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'])
        
        # Create comprehensive report
        report = f"""
Maritime Anomaly Detection Validation Report
===========================================

Model: {model_name}
Validation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
-------------------
Accuracy:           {metrics.get('accuracy', 0):.4f}
Precision:          {metrics.get('precision', 0):.4f}
Recall:             {metrics.get('recall', 0):.4f}
F1-Score:           {metrics.get('f1_score', 0):.4f}
ROC-AUC:            {metrics.get('roc_auc', 0):.4f}
Average Precision:  {metrics.get('average_precision', 0):.4f}

Confusion Matrix Components:
---------------------------
True Positives:     {metrics.get('true_positives', 0)}
True Negatives:     {metrics.get('true_negatives', 0)}
False Positives:    {metrics.get('false_positives', 0)}
False Negatives:    {metrics.get('false_negatives', 0)}

Additional Metrics:
------------------
Specificity:        {metrics.get('specificity', 0):.4f}
Sensitivity:        {metrics.get('sensitivity', 0):.4f}
Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}
MCC:                {metrics.get('mcc', 0):.4f}

Threshold Analysis:
------------------
Optimal Threshold:  {results.get('optimal_threshold', 0.5):.4f}
Optimal F1-Score:   {results.get('optimal_f1_score', 0):.4f}
Detection Rate @ 1% FPR: {results.get('detection_rate_at_1pct_fpr', 0):.4f}

Classification Report:
{class_report}
"""
        
        # Add cross-validation results if available
        if model_name in self.cross_validation_results:
            cv_results = self.cross_validation_results[model_name]
            cv_stats = cv_results['cv_statistics']
            
            report += f"""

Cross-Validation Results ({cv_results['cv_folds']} folds):
========================================================
Accuracy:    {cv_stats.get('accuracy_mean', 0):.4f} ± {cv_stats.get('accuracy_std', 0):.4f}
Precision:   {cv_stats.get('precision_mean', 0):.4f} ± {cv_stats.get('precision_std', 0):.4f}
Recall:      {cv_stats.get('recall_mean', 0):.4f} ± {cv_stats.get('recall_std', 0):.4f}
F1-Score:    {cv_stats.get('f1_score_mean', 0):.4f} ± {cv_stats.get('f1_score_std', 0):.4f}
ROC-AUC:     {cv_stats.get('roc_auc_mean', 0):.4f} ± {cv_stats.get('roc_auc_std', 0):.4f}
"""
        
        return report
    
    def save_validation_results(self, output_dir: str) -> None:
        """
        Save all validation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save validation results
        for model_name in self.validation_results.keys():
            # Save detailed report
            report = self.generate_validation_report(model_name)
            report_path = output_path / f'{model_name}_validation_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Generate plots
            self.generate_validation_plots(model_name, str(output_path))
        
        # Save comparison results if multiple models
        if len(self.validation_results) > 1:
            models = {}
            for model_name, results in self.validation_results.items():
                # Create dummy model for comparison (just need the results)
                models[model_name] = None
            
            # Create comparison table
            comparison_data = []
            for model_name, results in self.validation_results.items():
                metrics = results['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1-Score': metrics.get('f1_score', 0),
                    'ROC-AUC': metrics.get('roc_auc', 0)
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(output_path / 'model_comparison.csv', index=False)
        
        logger.info(f"Validation results saved to {output_path}")
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """
        Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Name of the best model
        """
        if not self.validation_results:
            raise ValueError("No validation results available")
        
        best_score = -np.inf
        best_model = None
        
        for model_name, results in self.validation_results.items():
            score = results['metrics'].get(metric, 0)
            if score > best_score:
                best_score = score
                best_model = model_name
        
        logger.info(f"Best model by {metric}: {best_model} (score: {best_score:.4f})")
        
        return best_model


def create_validator(config: Dict[str, Any]) -> ModelValidator:
    """
    Create model validator from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured validator
    """
    return ModelValidator(config) 