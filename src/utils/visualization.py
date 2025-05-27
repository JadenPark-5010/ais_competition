"""
Visualization utilities for maritime anomaly detection models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_attention_heatmap(attention_weights: np.ndarray, 
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot attention heatmap for TrAISformer model.
    
    Args:
        attention_weights: Attention weights array [seq_len, seq_len]
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        cmap='Blues',
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False
    )
    
    plt.title('TrAISformer Attention Heatmap')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(training_history: Dict[str, List[float]], 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot training curves for model training.
    
    Args:
        training_history: Dictionary containing training metrics
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss curves
    if 'train_loss' in training_history and 'val_loss' in training_history:
        axes[0, 0].plot(training_history['train_loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(training_history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curves
    if 'train_acc' in training_history and 'val_acc' in training_history:
        axes[0, 1].plot(training_history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(training_history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # AUC curves
    if 'val_auc' in training_history:
        axes[1, 0].plot(training_history['val_auc'], label='Validation AUC', linewidth=2, color='green')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score curves
    if 'val_f1' in training_history:
        axes[1, 1].plot(training_history['val_f1'], label='Validation F1', linewidth=2, color='red')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ensemble_results(results_dict: Dict[str, Dict], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot ensemble model results comparison.
    
    Args:
        results_dict: Dictionary containing results for different models
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Extract metrics
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create data for plotting
    data = []
    for model in models:
        model_results = results_dict[model]
        if 'test_metrics' in model_results:
            metrics_data = model_results['test_metrics']
        elif 'metrics' in model_results:
            metrics_data = model_results['metrics']
        else:
            continue
            
        for metric in metrics:
            if metric in metrics_data:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metrics_data[metric]
                })
    
    if not data:
        print("No data available for plotting")
        return
    
    df = pd.DataFrame(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        metric_title = metric.replace('_', ' ').title()
        metric_data = df[df['Metric'] == metric_title]
        
        if not metric_data.empty:
            axes[i].bar(metric_data['Model'], metric_data['Value'])
            axes[i].set_title(f'{metric_title} Comparison')
            axes[i].set_ylabel(metric_title)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
    
    # Overall comparison
    pivot_df = df.pivot(index='Model', columns='Metric', values='Value')
    if not pivot_df.empty:
        axes[5].bar(range(len(models)), pivot_df.mean(axis=1))
        axes[5].set_title('Average Performance')
        axes[5].set_ylabel('Average Score')
        axes[5].set_xticks(range(len(models)))
        axes[5].set_xticklabels(models, rotation=45)
        axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_model_comparison(results_dict: Dict[str, Dict], 
                         save_path: Optional[str] = None) -> None:
    """
    Create an interactive model comparison plot using Plotly.
    
    Args:
        results_dict: Dictionary containing results for different models
        save_path: Path to save the plot (HTML format)
    """
    # Extract data
    models = []
    metrics_data = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'roc_auc': []
    }
    
    for model_name, results in results_dict.items():
        if 'test_metrics' in results:
            model_metrics = results['test_metrics']
        elif 'metrics' in results:
            model_metrics = results['metrics']
        else:
            continue
        
        models.append(model_name)
        for metric in metrics_data.keys():
            metrics_data[metric].append(model_metrics.get(metric, 0.0))
    
    # Create radar chart
    fig = go.Figure()
    
    for i, model in enumerate(models):
        values = [metrics_data[metric][i] for metric in metrics_data.keys()]
        values.append(values[0])  # Close the radar chart
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metrics_data.keys()) + [list(metrics_data.keys())[0]],
            fill='toself',
            name=model,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Model Performance Comparison (Radar Chart)"
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def plot_feature_importance(importance_dict: Dict[str, Dict[str, float]], 
                           save_path: Optional[str] = None,
                           top_n: int = 20,
                           figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot feature importance for different models.
    
    Args:
        importance_dict: Dictionary containing feature importance for each model
        save_path: Path to save the plot
        top_n: Number of top features to show
        figsize: Figure size
    """
    n_models = len(importance_dict)
    if n_models == 0:
        print("No feature importance data available")
        return
    
    # Calculate subplot layout
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, (model_name, importance) in enumerate(importance_dict.items()):
        if i >= len(axes):
            break
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        axes[i].barh(y_pos, importances)
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(features)
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'{model_name} Feature Importance')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory_attention(trajectory: np.ndarray, 
                             attention_weights: np.ndarray,
                             save_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot trajectory with attention weights overlay.
    
    Args:
        trajectory: Trajectory data [seq_len, 4] (lat, lon, speed, course)
        attention_weights: Attention weights [seq_len, seq_len]
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract trajectory components
    lats = trajectory[:, 0]
    lons = trajectory[:, 1]
    speeds = trajectory[:, 2]
    courses = trajectory[:, 3]
    
    # Average attention for each position
    avg_attention = np.mean(attention_weights, axis=0)
    
    # Plot trajectory with attention
    scatter = axes[0, 0].scatter(lons, lats, c=avg_attention, cmap='Reds', s=50)
    axes[0, 0].plot(lons, lats, 'b-', alpha=0.5, linewidth=1)
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('Trajectory with Attention Weights')
    plt.colorbar(scatter, ax=axes[0, 0], label='Attention Weight')
    
    # Speed profile with attention
    axes[0, 1].plot(speeds, 'g-', linewidth=2, label='Speed')
    ax_twin = axes[0, 1].twinx()
    ax_twin.plot(avg_attention, 'r--', linewidth=2, label='Attention')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Speed (knots)', color='g')
    ax_twin.set_ylabel('Attention Weight', color='r')
    axes[0, 1].set_title('Speed Profile with Attention')
    
    # Course profile with attention
    axes[1, 0].plot(courses, 'b-', linewidth=2, label='Course')
    ax_twin2 = axes[1, 0].twinx()
    ax_twin2.plot(avg_attention, 'r--', linewidth=2, label='Attention')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Course (degrees)', color='b')
    ax_twin2.set_ylabel('Attention Weight', color='r')
    axes[1, 0].set_title('Course Profile with Attention')
    
    # Attention heatmap
    im = axes[1, 1].imshow(attention_weights, cmap='Blues', aspect='auto')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    axes[1, 1].set_title('Attention Heatmap')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_confidence(y_true: np.ndarray, 
                              y_prob: np.ndarray,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plot prediction confidence distribution.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Confidence distribution by class
    normal_probs = y_prob[y_true == 0]
    anomaly_probs = y_prob[y_true == 1]
    
    axes[0, 0].hist(normal_probs, bins=50, alpha=0.7, label='Normal', density=True)
    axes[0, 0].hist(anomaly_probs, bins=50, alpha=0.7, label='Anomaly', density=True)
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Probability Distribution by Class')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot
    axes[0, 1].boxplot([normal_probs, anomaly_probs], labels=['Normal', 'Anomaly'])
    axes[0, 1].set_ylabel('Predicted Probability')
    axes[0, 1].set_title('Probability Distribution Boxplot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calibration plot
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    axes[1, 0].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    axes[1, 0].plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    axes[1, 0].set_xlabel('Mean Predicted Probability')
    axes[1, 0].set_ylabel('Fraction of Positives')
    axes[1, 0].set_title('Calibration Plot')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confidence vs accuracy
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(y_true[mask] == (y_prob[mask] > 0.5))
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    axes[1, 1].bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7)
    axes[1, 1].set_xlabel('Confidence Bin')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Confidence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_interactive_dashboard(results_dict: Dict[str, Dict], 
                               save_path: Optional[str] = None) -> None:
    """
    Create an interactive dashboard for model results.
    
    Args:
        results_dict: Dictionary containing results for different models
        save_path: Path to save the dashboard (HTML format)
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Comparison', 'ROC Curves', 'Precision-Recall', 'Feature Importance'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Model comparison
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    for metric in metrics:
        values = []
        for model in models:
            if 'test_metrics' in results_dict[model]:
                values.append(results_dict[model]['test_metrics'].get(metric, 0))
            else:
                values.append(0)
        
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric),
            row=1, col=1
        )
    
    # Add more interactive elements as needed
    fig.update_layout(
        title="Maritime Anomaly Detection Dashboard",
        showlegend=True,
        height=800
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show() 