"""
TrAISformer model for maritime anomaly detection.

This module implements a Transformer-based model specifically designed for
AIS trajectory data, using four-hot encoding and attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.preprocessing import StandardScaler
import math

from .base_model import SupervisedAnomalyDetector

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class FourHotEncoder:
    """
    Four-hot encoder for AIS trajectory data.
    
    Encodes latitude, longitude, speed, and course into discrete bins
    for transformer processing.
    """
    
    def __init__(self, 
                 lat_bins: int = 100,
                 lon_bins: int = 100, 
                 speed_bins: int = 50,
                 course_bins: int = 36):
        """
        Initialize four-hot encoder.
        
        Args:
            lat_bins: Number of latitude bins
            lon_bins: Number of longitude bins
            speed_bins: Number of speed bins
            course_bins: Number of course bins
        """
        self.lat_bins = lat_bins
        self.lon_bins = lon_bins
        self.speed_bins = speed_bins
        self.course_bins = course_bins
        
        self.lat_edges = None
        self.lon_edges = None
        self.speed_edges = None
        self.course_edges = None
        
        self.vocab_size = lat_bins + lon_bins + speed_bins + course_bins
        
    def fit(self, trajectories: List[np.ndarray]) -> 'FourHotEncoder':
        """
        Fit the encoder to trajectory data.
        
        Args:
            trajectories: List of trajectory arrays with columns [lat, lon, speed, course]
            
        Returns:
            Self for method chaining
        """
        # Collect all values
        all_lats = []
        all_lons = []
        all_speeds = []
        all_courses = []
        
        for traj in trajectories:
            if traj.shape[1] >= 4:
                all_lats.extend(traj[:, 0])
                all_lons.extend(traj[:, 1])
                all_speeds.extend(traj[:, 2])
                all_courses.extend(traj[:, 3])
        
        # Create bin edges
        self.lat_edges = np.linspace(np.min(all_lats), np.max(all_lats), self.lat_bins + 1)
        self.lon_edges = np.linspace(np.min(all_lons), np.max(all_lons), self.lon_bins + 1)
        self.speed_edges = np.linspace(0, np.max(all_speeds), self.speed_bins + 1)
        self.course_edges = np.linspace(0, 360, self.course_bins + 1)
        
        return self
    
    def transform(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Transform trajectory to four-hot encoded sequence.
        
        Args:
            trajectory: Trajectory array with columns [lat, lon, speed, course]
            
        Returns:
            Four-hot encoded sequence of shape (seq_len, vocab_size)
        """
        if self.lat_edges is None:
            raise ValueError("Encoder must be fitted before transform")
        
        seq_len = len(trajectory)
        encoded = np.zeros((seq_len, self.vocab_size))
        
        for i, point in enumerate(trajectory):
            lat, lon, speed, course = point[:4]
            
            # Digitize values
            lat_bin = np.digitize(lat, self.lat_edges) - 1
            lon_bin = np.digitize(lon, self.lon_edges) - 1
            speed_bin = np.digitize(speed, self.speed_edges) - 1
            course_bin = np.digitize(course, self.course_edges) - 1
            
            # Clip to valid range
            lat_bin = np.clip(lat_bin, 0, self.lat_bins - 1)
            lon_bin = np.clip(lon_bin, 0, self.lon_bins - 1)
            speed_bin = np.clip(speed_bin, 0, self.speed_bins - 1)
            course_bin = np.clip(course_bin, 0, self.course_bins - 1)
            
            # Set four-hot encoding
            encoded[i, lat_bin] = 1
            encoded[i, self.lat_bins + lon_bin] = 1
            encoded[i, self.lat_bins + self.lon_bins + speed_bin] = 1
            encoded[i, self.lat_bins + self.lon_bins + self.speed_bins + course_bin] = 1
        
        return encoded


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory data.
    """
    
    def __init__(self, 
                 trajectories: List[np.ndarray],
                 labels: Optional[np.ndarray] = None,
                 encoder: Optional[FourHotEncoder] = None,
                 max_length: int = 512):
        """
        Initialize trajectory dataset.
        
        Args:
            trajectories: List of trajectory arrays
            labels: Optional labels for supervised learning
            encoder: Four-hot encoder
            max_length: Maximum sequence length
        """
        self.trajectories = trajectories
        self.labels = labels
        self.encoder = encoder
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx]
        
        # Truncate or pad trajectory
        if len(trajectory) > self.max_length:
            trajectory = trajectory[:self.max_length]
        
        # Encode trajectory
        if self.encoder:
            encoded = self.encoder.transform(trajectory)
        else:
            encoded = trajectory
        
        # Pad sequence
        seq_len = len(encoded)
        if seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, encoded.shape[1]))
            encoded = np.vstack([encoded, padding])
        
        # Create attention mask
        attention_mask = np.zeros(self.max_length)
        attention_mask[:seq_len] = 1
        
        result = {
            'input_ids': torch.FloatTensor(encoded),
            'attention_mask': torch.FloatTensor(attention_mask),
            'seq_length': torch.LongTensor([seq_len])
        }
        
        if self.labels is not None:
            result['labels'] = torch.LongTensor([self.labels[idx]])
        
        return result


class TrAISformerModel(nn.Module):
    """
    TrAISformer neural network model.
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 512,
                 num_classes: int = 2):
        """
        Initialize TrAISformer model.
        
        Args:
            vocab_size: Size of vocabulary (four-hot encoding dimension)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            num_classes: Number of output classes
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Input projection
        self.input_projection = nn.Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input sequences of shape (batch_size, seq_len, vocab_size)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Project input to model dimension
        x = self.input_projection(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for positions to mask)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling with attention mask
        if attention_mask is not None:
            # Mask out padded positions
            mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * mask
            # Average over valid positions
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / seq_lengths
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class TrAISformerAnomalyDetector(SupervisedAnomalyDetector):
    """
    TrAISformer-based anomaly detector for maritime trajectories.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize TrAISformer anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Model configuration
        model_config = self.config.get('models', {}).get('traisformer', {})
        
        self.d_model = model_config.get('d_model', 256)
        self.nhead = model_config.get('nhead', 8)
        self.num_layers = model_config.get('num_layers', 6)
        self.dropout = model_config.get('dropout', 0.1)
        self.max_seq_length = model_config.get('max_seq_length', 512)
        
        # Encoding configuration
        encoding_config = model_config.get('encoding', {})
        self.lat_bins = encoding_config.get('lat_bins', 100)
        self.lon_bins = encoding_config.get('lon_bins', 100)
        self.speed_bins = encoding_config.get('speed_bins', 50)
        self.course_bins = encoding_config.get('course_bins', 36)
        
        # Training configuration
        training_config = model_config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 0.0001)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.batch_size = self.config.get('data', {}).get('batch_size', 32)
        
        # Initialize components
        self.encoder = FourHotEncoder(
            self.lat_bins, self.lon_bins, 
            self.speed_bins, self.course_bins
        )
        self.scaler = StandardScaler()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"TrAISformer initialized with device: {self.device}")
    
    def _prepare_trajectories(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Prepare trajectory data from feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            List of trajectory arrays
        """
        # This is a simplified version - in practice, you would need to
        # reconstruct trajectories from the feature matrix or work with
        # raw trajectory data directly
        
        # For now, assume X contains trajectory features that can be
        # converted back to [lat, lon, speed, course] format
        trajectories = []
        
        # Extract basic trajectory features
        for i in range(len(X)):
            # This is a placeholder - you would implement proper trajectory
            # reconstruction based on your feature engineering
            traj = np.random.rand(50, 4)  # Placeholder trajectory
            trajectories.append(traj)
        
        return trajectories
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TrAISformerAnomalyDetector':
        """
        Fit the TrAISformer model.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        y = self.validate_labels(y)
        
        logger.info("Fitting TrAISformer model")
        
        # Prepare trajectory data
        trajectories = self._prepare_trajectories(X)
        
        # Fit encoder
        self.encoder.fit(trajectories)
        
        # Create model
        self.model = TrAISformerModel(
            vocab_size=self.encoder.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            max_seq_length=self.max_seq_length,
            num_classes=len(np.unique(y))
        ).to(self.device)
        
        # Create dataset and dataloader
        dataset = TrajectoryDataset(
            trajectories=trajectories,
            labels=y,
            encoder=self.encoder,
            max_length=self.max_seq_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        # Training setup
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = self.config.get('training', {}).get('epochs', 10)
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].squeeze().to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        logger.info("TrAISformer training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Prepare trajectory data
        trajectories = self._prepare_trajectories(X)
        
        # Create dataset
        dataset = TrajectoryDataset(
            trajectories=trajectories,
            encoder=self.encoder,
            max_length=self.max_seq_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Prediction
        self.model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                probabilities = F.softmax(logits, dim=1)
                
                all_probabilities.append(probabilities.cpu().numpy())
        
        return np.vstack(all_probabilities)


def create_traisformer_model(config: Dict[str, Any]) -> TrAISformerAnomalyDetector:
    """
    Create TrAISformer model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TrAISformer anomaly detector
    """
    return TrAISformerAnomalyDetector(config) 