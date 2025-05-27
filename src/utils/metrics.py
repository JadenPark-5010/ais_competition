"""
Evaluation metrics for maritime anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class AnomalyDetectionMetrics:
    """
    Comprehensive metrics calculator for anomaly detection tasks.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Decision threshold for binary classification
        """
        self.threshold = threshold
        self.metrics_history = []
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for anomaly detection.
        
        Args:
            y_true: True binary labels (0: normal, 1: anomaly)
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Matthews Correlation Coefficient
        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if mcc_denominator != 0:
            metrics['mcc'] = (tp * tn - fp * fn) / mcc_denominator
        else:
            metrics['mcc'] = 0
        
        # Score-based metrics (if scores are provided)
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                metrics['average_precision'] = average_precision_score(y_true, y_scores)
            except ValueError:
                # Handle cases where only one class is present
                metrics['roc_auc'] = 0.5
                metrics['average_precision'] = np.mean(y_true)
        
        return metrics
    
    def calculate_threshold_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate metrics across different thresholds.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            thresholds: Array of thresholds to evaluate
            
        Returns:
            Dictionary containing metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 101)
        
        metrics_per_threshold = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1_score': [],
            'accuracy': []
        }
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            metrics_per_threshold['precision'].append(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics_per_threshold['recall'].append(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics_per_threshold['f1_score'].append(
                f1_score(y_true, y_pred, zero_division=0)
            )
            metrics_per_threshold['accuracy'].append(
                accuracy_score(y_true, y_pred)
            )
        
        # Convert to numpy arrays
        for key in ['precision', 'recall', 'f1_score', 'accuracy']:
            metrics_per_threshold[key] = np.array(metrics_per_threshold[key])
        
        return metrics_per_threshold
    
    def find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        metric: str = 'f1_score'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold based on specified metric.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            metric: Metric to optimize ('f1_score', 'precision', 'recall', 'accuracy')
            
        Returns:
            Tuple of (optimal_threshold, optimal_metric_value)
        """
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_scores)
        
        if metric not in threshold_metrics:
            raise ValueError(f"Metric '{metric}' not available")
        
        optimal_idx = np.argmax(threshold_metrics[metric])
        optimal_threshold = threshold_metrics['thresholds'][optimal_idx]
        optimal_value = threshold_metrics[metric][optimal_idx]
        
        return optimal_threshold, optimal_value
    
    def calculate_detection_rate_at_fpr(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        target_fpr: float = 0.01
    ) -> float:
        """
        Calculate detection rate (TPR) at specific false positive rate.
        
        Args:
            y_true: True binary labels
            y_scores: Prediction scores
            target_fpr: Target false positive rate
            
        Returns:
            Detection rate at target FPR
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        
        # Find closest FPR to target
        idx = np.argmin(np.abs(fpr - target_fpr))
        return tpr[idx]
    
    def update_history(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Update metrics history.
        
        Args:
            metrics: Metrics dictionary
            epoch: Current epoch number
        """
        metrics_with_epoch = metrics.copy()
        metrics_with_epoch['epoch'] = epoch
        self.metrics_history.append(metrics_with_epoch)
    
    def get_best_metrics(self, metric: str = 'f1_score') -> Dict[str, float]:
        """
        Get best metrics from history based on specified metric.
        
        Args:
            metric: Metric to use for finding best performance
            
        Returns:
            Best metrics dictionary
        """
        if not self.metrics_history:
            return {}
        
        best_idx = np.argmax([m.get(metric, 0) for m in self.metrics_history])
        return self.metrics_history[best_idx]
    
    def print_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate and return classification report.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            target_names: Names for classes
            
        Returns:
            Classification report string
        """
        if target_names is None:
            target_names = ['Normal', 'Anomaly']
        
        return classification_report(y_true, y_pred, target_names=target_names)


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    class_names: List[str] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    title: str = 'ROC Curve',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    title: str = 'Precision-Recall Curve',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')
    
    # Baseline (random classifier)
    baseline = np.mean(y_true)
    ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
               label=f'Random classifier (AP = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_threshold_metrics(
    y_true: np.ndarray, 
    y_scores: np.ndarray,
    metrics: List[str] = None,
    title: str = 'Metrics vs Threshold',
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot metrics vs threshold.
    
    Args:
        y_true: True binary labels
        y_scores: Prediction scores
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    
    calculator = AnomalyDetectionMetrics()
    threshold_metrics = calculator.calculate_threshold_metrics(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric in metrics:
        if metric in threshold_metrics:
            ax.plot(threshold_metrics['thresholds'], threshold_metrics[metric], 
                   label=metric.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def calculate_classification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray, 
                                   y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'average_precision': average_precision_score(y_true, y_prob)
    }
    
    # Additional metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics.update({
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0
    })
    
    return metrics


def calculate_ensemble_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray, 
                             y_prob: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics specifically for ensemble models.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary of ensemble-specific metrics
    """
    base_metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    
    # Additional ensemble-specific metrics
    ensemble_metrics = base_metrics.copy()
    
    # Calibration metrics
    from sklearn.calibration import calibration_curve
    
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=10
        )
        
        # Brier score (lower is better)
        brier_score = np.mean((y_prob - y_true) ** 2)
        ensemble_metrics['brier_score'] = brier_score
        
        # Reliability (calibration error)
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        ensemble_metrics['calibration_error'] = calibration_error
        
    except Exception:
        ensemble_metrics['brier_score'] = np.nan
        ensemble_metrics['calibration_error'] = np.nan
    
    # Confidence intervals for key metrics
    n_bootstrap = 1000
    bootstrap_metrics = bootstrap_confidence_intervals(
        y_true, y_pred, y_prob, n_bootstrap=n_bootstrap
    )
    
    for metric, (lower, upper) in bootstrap_metrics.items():
        ensemble_metrics[f'{metric}_ci_lower'] = lower
        ensemble_metrics[f'{metric}_ci_upper'] = upper
    
    return ensemble_metrics


def bootstrap_confidence_intervals(y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_prob: np.ndarray,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
    """
    Calculate bootstrap confidence intervals for key metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        
    Returns:
        Dictionary of confidence intervals
    """
    np.random.seed(42)
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    bootstrap_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]
        
        # Calculate metrics for bootstrap sample
        try:
            bootstrap_scores['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_scores['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_scores['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_scores['f1_score'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_scores['roc_auc'].append(roc_auc_score(y_true_boot, y_prob_boot))
        except Exception:
            # Skip this bootstrap sample if there's an error
            continue
    
    # Calculate confidence intervals
    confidence_intervals = {}
    for metric, scores in bootstrap_scores.items():
        if scores:  # Only if we have valid scores
            lower = np.percentile(scores, lower_percentile)
            upper = np.percentile(scores, upper_percentile)
            confidence_intervals[metric] = (lower, upper)
        else:
            confidence_intervals[metric] = (np.nan, np.nan)
    
    return confidence_intervals


def calculate_threshold_metrics(y_true: np.ndarray, 
                              y_prob: np.ndarray,
                              thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Calculate metrics across different probability thresholds.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        thresholds: Array of thresholds to evaluate
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        metrics['threshold'] = threshold
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def find_optimal_threshold(y_true: np.ndarray, 
                         y_prob: np.ndarray,
                         metric: str = 'f1_score') -> Tuple[float, float]:
    """
    Find optimal threshold based on specified metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1_score', 'precision', 'recall', etc.)
        
    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    threshold_df = calculate_threshold_metrics(y_true, y_prob)
    
    if metric not in threshold_df.columns:
        raise ValueError(f"Metric '{metric}' not found in calculated metrics")
    
    optimal_idx = threshold_df[metric].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
    optimal_value = threshold_df.loc[optimal_idx, metric]
    
    return optimal_threshold, optimal_value


def plot_threshold_analysis(y_true: np.ndarray, 
                          y_prob: np.ndarray,
                          save_path: Optional[str] = None) -> None:
    """
    Plot threshold analysis showing how metrics change with threshold.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
    """
    threshold_df = calculate_threshold_metrics(y_true, y_prob)
    
    plt.figure(figsize=(12, 8))
    
    # Plot key metrics
    metrics_to_plot = ['precision', 'recall', 'f1_score', 'accuracy']
    
    for metric in metrics_to_plot:
        plt.plot(threshold_df['threshold'], threshold_df[metric], 
                label=metric.replace('_', ' ').title(), linewidth=2)
    
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_model_agreement(predictions_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calculate agreement between different models.
    
    Args:
        predictions_dict: Dictionary mapping model names to prediction arrays
        
    Returns:
        Dictionary of pairwise agreement scores
    """
    model_names = list(predictions_dict.keys())
    agreement_scores = {}
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            pred1 = predictions_dict[model1]
            pred2 = predictions_dict[model2]
            
            # Calculate agreement (percentage of same predictions)
            agreement = np.mean(pred1 == pred2)
            agreement_scores[f'{model1}_vs_{model2}'] = agreement
    
    return agreement_scores


def calculate_diversity_metrics(predictions_dict: Dict[str, np.ndarray],
                              y_true: np.ndarray) -> Dict[str, float]:
    """
    Calculate diversity metrics for ensemble models.
    
    Args:
        predictions_dict: Dictionary mapping model names to prediction arrays
        y_true: True labels
        
    Returns:
        Dictionary of diversity metrics
    """
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        return {}
    
    # Convert predictions to matrix
    pred_matrix = np.column_stack([predictions_dict[name] for name in model_names])
    
    # Q-statistic (Yule's Q)
    q_statistics = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            pred_i = pred_matrix[:, i]
            pred_j = pred_matrix[:, j]
            
            # 2x2 contingency table
            n11 = np.sum((pred_i == 1) & (pred_j == 1))
            n10 = np.sum((pred_i == 1) & (pred_j == 0))
            n01 = np.sum((pred_i == 0) & (pred_j == 1))
            n00 = np.sum((pred_i == 0) & (pred_j == 0))
            
            # Q-statistic
            if (n11 * n00 + n01 * n10) != 0:
                q = (n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10)
                q_statistics.append(q)
    
    # Disagreement measure
    disagreement_scores = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            pred_i = pred_matrix[:, i]
            pred_j = pred_matrix[:, j]
            
            disagreement = np.mean(pred_i != pred_j)
            disagreement_scores.append(disagreement)
    
    # Double fault measure
    double_fault_scores = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            pred_i = pred_matrix[:, i]
            pred_j = pred_matrix[:, j]
            
            both_wrong = np.sum((pred_i != y_true) & (pred_j != y_true))
            double_fault = both_wrong / len(y_true)
            double_fault_scores.append(double_fault)
    
    diversity_metrics = {
        'mean_q_statistic': np.mean(q_statistics) if q_statistics else 0.0,
        'mean_disagreement': np.mean(disagreement_scores) if disagreement_scores else 0.0,
        'mean_double_fault': np.mean(double_fault_scores) if double_fault_scores else 0.0,
        'pairwise_diversity': len([d for d in disagreement_scores if d > 0.1]) / len(disagreement_scores) if disagreement_scores else 0.0
    }
    
    return diversity_metrics


def generate_classification_report_dict(y_true: np.ndarray, 
                                       y_pred: np.ndarray) -> Dict:
    """
    Generate a detailed classification report as a dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Classification report as dictionary
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    report['confusion_matrix'] = cm.tolist()
    
    return report 