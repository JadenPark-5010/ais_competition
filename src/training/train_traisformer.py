"""
TrAISformer Training Script for Maritime Anomaly Detection

This script provides comprehensive training pipeline for TrAISformer model
with advanced features like attention visualization, early stopping, and
model interpretation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import yaml
import argparse
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.traisformer import TrAISformerAnomalyDetector, TrajectoryDataset, FourHotEncoder
from data.data_loader import AISDataLoader
from utils.visualization import plot_attention_heatmap, plot_training_curves
from utils.metrics import calculate_classification_metrics

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrAISformerTrainer:
    """
    Comprehensive trainer for TrAISformer model with advanced features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Data components
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.four_hot_encoder = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'val_auc': [],
            'val_f1': []
        }
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'traisformer-anomaly'),
                config=config,
                name=config.get('experiment_name', 'traisformer-experiment')
            )
    
    def prepare_data(self, data_path: str) -> None:
        """
        Prepare and load data for training.
        
        Args:
            data_path: Path to the data directory
        """
        logger.info("Preparing data...")
        
        # Load data
        data_loader = AISDataLoader(self.config['data'])
        df = data_loader.load_data(data_path)
        
        # Prepare trajectories
        trajectories = self._prepare_trajectories(df)
        labels = self._prepare_labels(df)
        
        # Initialize and fit four-hot encoder
        self.four_hot_encoder = FourHotEncoder(
            lat_bins=self.config['traisformer']['lat_bins'],
            lon_bins=self.config['traisformer']['lon_bins'],
            speed_bins=self.config['traisformer']['speed_bins'],
            course_bins=self.config['traisformer']['course_bins']
        )
        self.four_hot_encoder.fit(trajectories)
        
        # Split data
        train_trajectories, temp_trajectories, train_labels, temp_labels = train_test_split(
            trajectories, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        val_trajectories, test_trajectories, val_labels, test_labels = train_test_split(
            temp_trajectories, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        # Create datasets
        train_dataset = TrajectoryDataset(
            train_trajectories, train_labels, self.four_hot_encoder,
            max_length=self.config['traisformer']['max_seq_length']
        )
        
        val_dataset = TrajectoryDataset(
            val_trajectories, val_labels, self.four_hot_encoder,
            max_length=self.config['traisformer']['max_seq_length']
        )
        
        test_dataset = TrajectoryDataset(
            test_trajectories, test_labels, self.four_hot_encoder,
            max_length=self.config['traisformer']['max_seq_length']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers']
        )
        
        logger.info(f"Data prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    def _prepare_trajectories(self, df: pd.DataFrame) -> List[np.ndarray]:
        """Prepare trajectory data from DataFrame."""
        trajectories = []
        
        # Group by vessel_id or mmsi
        group_col = 'vessel_id' if 'vessel_id' in df.columns else 'mmsi'
        
        for vessel_id in df[group_col].unique():
            vessel_data = df[df[group_col] == vessel_id].sort_values('timestamp')
            
            # Extract trajectory features
            trajectory = vessel_data[['latitude', 'longitude', 'speed', 'course']].values
            
            if len(trajectory) >= 10:  # Minimum trajectory length
                trajectories.append(trajectory)
        
        return trajectories
    
    def _prepare_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare labels from DataFrame."""
        # Group by vessel and take the label (assuming one label per vessel)
        group_col = 'vessel_id' if 'vessel_id' in df.columns else 'mmsi'
        label_col = 'is_suspicious' if 'is_suspicious' in df.columns else 'label'
        
        labels = []
        for vessel_id in df[group_col].unique():
            vessel_data = df[df[group_col] == vessel_id]
            label = vessel_data[label_col].iloc[0]  # Take first label
            labels.append(label)
        
        return np.array(labels)
    
    def initialize_model(self) -> None:
        """Initialize the TrAISformer model."""
        logger.info("Initializing TrAISformer model...")
        
        # Create model
        self.model = TrAISformerAnomalyDetector(self.config['traisformer'])
        self.model.to(self.device)
        
        # Initialize optimizer
        if self.config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['training']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Initialize scheduler
        if self.config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        elif self.config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # Initialize loss function
        if self.config['training']['loss_function'] == 'cross_entropy':
            class_weights = torch.tensor(self.config['training']['class_weights']).float().to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif self.config['training']['loss_function'] == 'focal':
            self.criterion = self._focal_loss
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _focal_loss(self, outputs: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        ce_loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store for metrics calculation
                probabilities = torch.softmax(outputs, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        # Calculate additional metrics
        auc_score = roc_auc_score(all_labels, all_probabilities)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, auc_score, f1
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc, val_f1 = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_auc'].append(val_auc)
            self.training_history['val_f1'].append(val_f1)
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'val_f1': val_f1,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Print progress
            logger.info(
                f'Epoch {epoch + 1}/{self.config["training"]["epochs"]} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                f'Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}'
            )
            
            # Early stopping
            if val_auc > self.best_val_score:
                self.best_val_score = val_auc
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config['training']['patience']:
                    logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                    break
        
        logger.info("Training completed!")
    
    def evaluate(self) -> Dict:
        """Evaluate the model on test set."""
        logger.info("Evaluating model...")
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_classification_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Print results
        logger.info("Test Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def visualize_attention(self, sample_idx: int = 0) -> None:
        """Visualize attention weights for a sample."""
        logger.info("Generating attention visualization...")
        
        # Get a sample from test set
        sample_batch = next(iter(self.test_loader))
        input_ids = sample_batch['input_ids'][sample_idx:sample_idx+1].to(self.device)
        attention_mask = sample_batch['attention_mask'][sample_idx:sample_idx+1].to(self.device)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs, attention_weights = self.model(input_ids, attention_mask, return_attention=True)
        
        # Plot attention heatmap
        plot_attention_heatmap(
            attention_weights[0].cpu().numpy(),
            save_path='attention_heatmap.png'
        )
        
        logger.info("Attention visualization saved to attention_heatmap.png")
    
    def plot_training_curves(self) -> None:
        """Plot training curves."""
        plot_training_curves(
            self.training_history,
            save_path='training_curves.png'
        )
        logger.info("Training curves saved to training_curves.png")
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
            'config': self.config
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved to {filename}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {filename}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TrAISformer for maritime anomaly detection')
    parser.add_argument('--config', type=str, default='config/traisformer_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='outputs/',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = TrAISformerTrainer(config)
    
    # Prepare data
    trainer.prepare_data(args.data)
    
    # Initialize model
    trainer.initialize_model()
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Generate visualizations
    trainer.plot_training_curves()
    trainer.visualize_attention()
    
    # Save final results
    results = {
        'config': config,
        'metrics': metrics,
        'training_history': trainer.training_history
    }
    
    with open(output_dir / 'results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    logger.info(f"Training completed! Results saved to {output_dir}")


if __name__ == '__main__':
    main() 