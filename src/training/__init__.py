"""
Training module for maritime anomaly detection.
"""

from .trainer import AnomalyDetectionTrainer
from .validator import ModelValidator

__all__ = [
    'AnomalyDetectionTrainer',
    'ModelValidator'
] 