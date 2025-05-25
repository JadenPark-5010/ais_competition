"""
Evaluation metrics for maritime anomaly detection.
"""

import numpy as np
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