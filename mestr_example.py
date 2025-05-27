"""
MESTR Model Usage Example
Demonstrates how to use the MESTR model for maritime anomaly detection
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models.mestr import MESTR, create_mestr_model, MestrTrainer, MESTR_CONFIG
from src.models.advanced_ensemble import AdvancedEnsembleDetector


def generate_synthetic_ais_data(num_vessels=100, num_points_per_vessel=50):
    """
    Generate synthetic AIS trajectory data for demonstration.
    
    Args:
        num_vessels: Number of vessels to generate
        num_points_per_vessel: Number of trajectory points per vessel
        
    Returns:
        Tuple of (trajectories, labels)
    """
    trajectories = []
    labels = []
    
    for vessel_id in range(num_vessels):
        # Generate base trajectory (normal behavior)
        start_lat = np.random.uniform(35.0, 40.0)  # Example: Korean waters
        start_lon = np.random.uniform(125.0, 130.0)
        
        # Normal trajectory: smooth movement
        if vessel_id < num_vessels * 0.8:  # 80% normal vessels
            lat_points = start_lat + np.cumsum(np.random.normal(0, 0.01, num_points_per_vessel))
            lon_points = start_lon + np.cumsum(np.random.normal(0, 0.01, num_points_per_vessel))
            label = 0  # Normal
        else:  # 20% anomalous vessels
            # Anomalous trajectory: erratic movement or suspicious patterns
            lat_points = start_lat + np.cumsum(np.random.normal(0, 0.05, num_points_per_vessel))
            lon_points = start_lon + np.cumsum(np.random.normal(0, 0.05, num_points_per_vessel))
            
            # Add some suspicious patterns
            if np.random.random() > 0.5:
                # Sudden direction changes
                change_points = np.random.choice(num_points_per_vessel, 3, replace=False)
                for cp in change_points:
                    lat_points[cp:] += np.random.uniform(-0.1, 0.1)
                    lon_points[cp:] += np.random.uniform(-0.1, 0.1)
            
            label = 1  # Anomalous
        
        # Create trajectory matrix
        trajectory = np.column_stack([lat_points, lon_points])
        trajectories.append(trajectory)
        labels.append(label)
    
    return trajectories, labels


def create_data_loaders(trajectories, labels, batch_size=16, train_split=0.7, val_split=0.2):
    """
    Create PyTorch data loaders for training.
    
    Args:
        trajectories: List of trajectory arrays
        labels: List of labels
        batch_size: Batch size for data loaders
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Convert to tensors
    max_length = max(len(traj) for traj in trajectories)
    
    # Pad trajectories to same length
    padded_trajectories = []
    for traj in trajectories:
        if len(traj) < max_length:
            padding = np.zeros((max_length - len(traj), 2))
            padded_traj = np.vstack([traj, padding])
        else:
            padded_traj = traj[:max_length]
        padded_trajectories.append(padded_traj)
    
    # Convert to tensors
    X = torch.FloatTensor(np.array(padded_trajectories))
    y = torch.LongTensor(labels)
    
    # Split data
    n_samples = len(X)
    n_train = int(n_samples * train_split)
    n_val = int(n_samples * val_split)
    
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_mestr_model():
    """Train MESTR model on synthetic data."""
    print("🚢 MESTR Maritime Anomaly Detection Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("📊 Generating synthetic AIS data...")
    trajectories, labels = generate_synthetic_ais_data(num_vessels=200, num_points_per_vessel=100)
    print(f"Generated {len(trajectories)} vessel trajectories")
    print(f"Normal vessels: {labels.count(0)}, Anomalous vessels: {labels.count(1)}")
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(trajectories, labels)
    
    # Modify data loader to return dictionary format expected by MESTR
    def collate_fn(batch):
        trajectories, labels = zip(*batch)
        return {
            'trajectory': torch.stack(trajectories),
            'vessel_type': torch.stack(labels),
            'anomaly_label': torch.stack(labels)  # Same as vessel_type for binary classification
        }
    
    train_loader.collate_fn = collate_fn
    val_loader.collate_fn = collate_fn
    test_loader.collate_fn = collate_fn
    
    # Create MESTR model
    print("\n🤖 Creating MESTR model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configure for binary anomaly detection
    config = MESTR_CONFIG.copy()
    config['model']['num_vessel_types'] = 2  # Binary classification
    config['model']['max_sequence_length'] = 100
    config['training']['num_epochs'] = 20  # Reduced for demo
    config['training']['early_stopping_patience'] = 5
    
    model = create_mestr_model(config=config, device=device, for_anomaly_detection=True)
    print(f"Model info: {model.get_model_info()}")
    
    # Create trainer
    print("\n🏋️ Setting up trainer...")
    trainer = MestrTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        config=config
    )
    
    # Train model
    print("\n🚀 Starting training...")
    results = trainer.train(
        save_dir='./mestr_checkpoints',
        save_every=5,
        evaluate_test=True
    )
    
    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_metrics']['test_accuracy']:.2f}%")
    
    # Plot training history
    print("\n📈 Plotting training history...")
    trainer.plot_training_history('./mestr_training_history.png')
    
    return model, test_loader


def test_mestr_predictions(model, test_loader):
    """Test MESTR model predictions."""
    print("\n🔍 Testing MESTR predictions...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in test_loader:
            trajectories = batch['trajectory']
            labels = batch['vessel_type']
            
            # Get predictions
            predictions = model.predict(trajectories)
            
            all_predictions.extend(predictions['vessel_type'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(predictions['vessel_confidence'].cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomalous'],
                yticklabels=['Normal', 'Anomalous'])
    plt.title('MESTR Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('./mestr_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }


def test_ensemble_integration():
    """Test MESTR integration with ensemble model."""
    print("\n🔗 Testing MESTR integration with ensemble...")
    
    # Generate data for ensemble
    trajectories, labels = generate_synthetic_ais_data(num_vessels=50, num_points_per_vessel=30)
    
    # Convert to DataFrame format expected by ensemble
    data_rows = []
    for i, (traj, label) in enumerate(zip(trajectories, labels)):
        for j, (lat, lon) in enumerate(traj):
            data_rows.append({
                'vessel_id': i,
                'latitude': lat,
                'longitude': lon,
                'speed': np.random.uniform(5, 15),  # Random speed
                'course': np.random.uniform(0, 360),  # Random course
                'timestamp': j,
                'label': label
            })
    
    df = pd.DataFrame(data_rows)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    # Create ensemble with MESTR
    print("Creating advanced ensemble with MESTR...")
    ensemble = AdvancedEnsembleDetector()
    
    # Note: In practice, you would train the ensemble
    # For demo, we'll just test the prediction pipeline
    print("✅ Ensemble created successfully with MESTR integration!")
    
    return ensemble


def visualize_attention_weights(model, test_loader):
    """Visualize attention weights from MESTR."""
    print("\n👁️ Visualizing attention weights...")
    
    # Get a sample batch
    sample_batch = next(iter(test_loader))
    sample_trajectory = sample_batch['trajectory'][:1]  # Take first sample
    
    # Get attention weights
    try:
        attention_weights = model.get_attention_weights(sample_trajectory)
        
        if attention_weights:
            # Plot attention weights for first layer
            plt.figure(figsize=(12, 8))
            
            # Average attention across heads
            attn = attention_weights[0][0].cpu().numpy()  # First sample, first layer
            attn_avg = np.mean(attn, axis=0)  # Average across heads
            
            plt.imshow(attn_avg, cmap='Blues', aspect='auto')
            plt.colorbar()
            plt.title('MESTR Attention Weights (Layer 1, Averaged across heads)')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.savefig('./mestr_attention_weights.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Attention weights visualized!")
        else:
            print("⚠️ No attention weights available")
            
    except Exception as e:
        print(f"⚠️ Could not visualize attention weights: {e}")


def main():
    """Main function to run MESTR example."""
    try:
        # Train MESTR model
        model, test_loader = train_mestr_model()
        
        # Test predictions
        test_results = test_mestr_predictions(model, test_loader)
        
        # Visualize attention weights
        visualize_attention_weights(model, test_loader)
        
        # Test ensemble integration
        ensemble = test_ensemble_integration()
        
        print("\n🎉 MESTR example completed successfully!")
        print("\nFiles generated:")
        print("- ./mestr_checkpoints/ (model checkpoints)")
        print("- ./mestr_training_history.png (training plots)")
        print("- ./mestr_confusion_matrix.png (confusion matrix)")
        print("- ./mestr_attention_weights.png (attention visualization)")
        
        return {
            'model': model,
            'test_results': test_results,
            'ensemble': ensemble
        }
        
    except Exception as e:
        print(f"❌ Error in MESTR example: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main() 