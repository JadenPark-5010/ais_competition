"""
Clustering-based anomaly detection models for maritime data.

This module implements various clustering approaches including DBSCAN,
Isolation Forest, and HDBSCAN for unsupervised anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import hdbscan
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF

from .base_model import UnsupervisedAnomalyDetector

logger = logging.getLogger(__name__)


class DBSCANAnomalyDetector(UnsupervisedAnomalyDetector):
    """
    DBSCAN-based anomaly detector.
    
    Uses DBSCAN clustering to identify outliers as anomalies.
    Points that don't belong to any cluster (noise points) are considered anomalies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DBSCAN anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # DBSCAN parameters
        dbscan_config = self.config.get('models', {}).get('clustering', {}).get('dbscan', {})
        
        self.eps = dbscan_config.get('eps', 0.5)
        self.min_samples = dbscan_config.get('min_samples', 5)
        self.metric = dbscan_config.get('metric', 'euclidean')
        self.algorithm = dbscan_config.get('algorithm', 'auto')
        self.leaf_size = dbscan_config.get('leaf_size', 30)
        
        # Preprocessing
        preprocessing_config = dbscan_config.get('preprocessing', {})
        self.scaler_type = preprocessing_config.get('scaler', 'standard')
        self.use_pca = preprocessing_config.get('pca_components', None) is not None
        self.pca_components = preprocessing_config.get('pca_components', 50)
        
        # Initialize components
        self.scaler = self._create_scaler()
        self.pca = PCA(n_components=self.pca_components) if self.use_pca else None
        
        logger.info("DBSCAN Anomaly Detector initialized")
    
    def _create_scaler(self):
        """Create scaler based on configuration."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        elif self.scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        else:
            return StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DBSCANAnomalyDetector':
        """
        Fit DBSCAN model.
        
        Args:
            X: Training features
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting DBSCAN model")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        if self.use_pca:
            X_scaled = self.pca.fit_transform(X_scaled)
            logger.info(f"Applied PCA: {X.shape[1]} -> {X_scaled.shape[1]} features")
        
        # Fit DBSCAN
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size
        )
        
        cluster_labels = self.model.fit_predict(X_scaled)
        
        # Store cluster information
        self.cluster_labels_ = cluster_labels
        self.n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        self.n_noise_ = list(cluster_labels).count(-1)
        
        logger.info(f"DBSCAN found {self.n_clusters_} clusters and {self.n_noise_} noise points")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using DBSCAN.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Preprocess
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        # For new data, we need to determine cluster membership
        # Use nearest neighbor approach to existing clusters
        from sklearn.neighbors import NearestNeighbors
        
        # Get training data cluster centers
        training_data = self.scaler.transform(self.validate_input(X))  # This is a simplification
        if self.use_pca:
            training_data = self.pca.transform(training_data)
        
        # Find cluster centers
        cluster_centers = []
        for cluster_id in range(self.n_clusters_):
            cluster_mask = self.cluster_labels_ == cluster_id
            if cluster_mask.sum() > 0:
                center = training_data[cluster_mask].mean(axis=0)
                cluster_centers.append(center)
        
        if not cluster_centers:
            # No clusters found, all points are anomalies
            return np.ones(len(X), dtype=int)
        
        cluster_centers = np.array(cluster_centers)
        
        # Find nearest cluster for each point
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(cluster_centers)
        
        distances, _ = nn.kneighbors(X_scaled)
        
        # Points far from any cluster center are anomalies
        # Use eps as threshold
        predictions = (distances.flatten() > self.eps).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Preprocess
        X_scaled = self.scaler.transform(X)
        if self.use_pca:
            X_scaled = self.pca.transform(X_scaled)
        
        # Calculate distances to cluster centers
        from sklearn.neighbors import NearestNeighbors
        
        # This is a simplified approach - in practice you'd store training data
        training_data = X_scaled  # Placeholder
        
        cluster_centers = []
        for cluster_id in range(self.n_clusters_):
            cluster_mask = self.cluster_labels_ == cluster_id
            if cluster_mask.sum() > 0:
                center = training_data[cluster_mask].mean(axis=0)
                cluster_centers.append(center)
        
        if not cluster_centers:
            return np.ones(len(X))
        
        cluster_centers = np.array(cluster_centers)
        
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(cluster_centers)
        
        distances, _ = nn.kneighbors(X_scaled)
        
        # Normalize distances to [0, 1] range
        scores = distances.flatten() / (self.eps + 1e-8)
        scores = np.clip(scores, 0, 1)
        
        return scores


class IsolationForestAnomalyDetector(UnsupervisedAnomalyDetector):
    """
    Isolation Forest anomaly detector.
    
    Uses Isolation Forest algorithm to detect anomalies based on
    the principle that anomalies are easier to isolate.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Isolation Forest anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Isolation Forest parameters
        if_config = self.config.get('models', {}).get('clustering', {}).get('isolation_forest', {})
        
        self.n_estimators = if_config.get('n_estimators', 100)
        self.max_samples = if_config.get('max_samples', 'auto')
        self.contamination = if_config.get('contamination', 0.1)
        self.max_features = if_config.get('max_features', 1.0)
        self.bootstrap = if_config.get('bootstrap', False)
        self.random_state = if_config.get('random_state', 42)
        
        # Preprocessing
        preprocessing_config = if_config.get('preprocessing', {})
        self.scaler_type = preprocessing_config.get('scaler', 'standard')
        
        self.scaler = self._create_scaler()
        
        logger.info("Isolation Forest Anomaly Detector initialized")
    
    def _create_scaler(self):
        """Create scaler based on configuration."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return StandardScaler()
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'IsolationForestAnomalyDetector':
        """
        Fit Isolation Forest model.
        
        Args:
            X: Training features
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting Isolation Forest model")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state
        )
        
        self.model.fit(X_scaled)
        
        self.is_fitted = True
        logger.info("Isolation Forest training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X_scaled)
        
        # Convert to 0/1 format (0 for normal, 1 for anomaly)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores (more negative = more anomalous)
        scores = self.model.decision_function(X_scaled)
        
        # Convert to [0, 1] range where higher = more anomalous
        scores = -scores  # Flip sign
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores


class HDBSCANAnomalyDetector(UnsupervisedAnomalyDetector):
    """
    HDBSCAN-based anomaly detector.
    
    Uses HDBSCAN clustering with outlier scores for anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize HDBSCAN anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # HDBSCAN parameters
        hdbscan_config = self.config.get('models', {}).get('clustering', {}).get('hdbscan', {})
        
        self.min_cluster_size = hdbscan_config.get('min_cluster_size', 5)
        self.min_samples = hdbscan_config.get('min_samples', 3)
        self.cluster_selection_epsilon = hdbscan_config.get('cluster_selection_epsilon', 0.0)
        self.alpha = hdbscan_config.get('alpha', 1.0)
        
        self.scaler = StandardScaler()
        
        logger.info("HDBSCAN Anomaly Detector initialized")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'HDBSCANAnomalyDetector':
        """
        Fit HDBSCAN model.
        
        Args:
            X: Training features
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting HDBSCAN model")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit HDBSCAN
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            alpha=self.alpha
        )
        
        cluster_labels = self.model.fit_predict(X_scaled)
        
        # Store results
        self.cluster_labels_ = cluster_labels
        self.outlier_scores_ = self.model.outlier_scores_
        self.n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        logger.info(f"HDBSCAN found {self.n_clusters_} clusters")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using outlier scores.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        scores = self.predict_proba(X)
        
        if self.threshold_ is None:
            # Use default threshold based on training data
            threshold = np.percentile(self.outlier_scores_, 90)
        else:
            threshold = self.threshold_
        
        return (scores > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outlier scores.
        
        Args:
            X: Input features
            
        Returns:
            Outlier scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        # For new data, approximate outlier scores
        # This is a simplified approach
        from sklearn.neighbors import NearestNeighbors
        
        # Use training data to estimate scores
        nn = NearestNeighbors(n_neighbors=self.min_samples)
        nn.fit(self.scaler.transform(X))  # Simplified - should use training data
        
        distances, _ = nn.kneighbors(X_scaled)
        scores = distances.mean(axis=1)
        
        # Normalize scores
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores


class CombinedClusteringAnomalyDetector(UnsupervisedAnomalyDetector):
    """
    Combined clustering anomaly detector.
    
    Combines multiple clustering-based approaches for robust anomaly detection.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize combined clustering anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Initialize individual detectors
        self.isolation_forest = IsolationForestAnomalyDetector(config)
        self.dbscan = DBSCANAnomalyDetector(config)
        
        # Weights for combining scores
        self.weights = [0.6, 0.4]  # [isolation_forest, dbscan]
        
        logger.info("Combined Clustering Anomaly Detector initialized")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'CombinedClusteringAnomalyDetector':
        """
        Fit all component models.
        
        Args:
            X: Training features
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting combined clustering models")
        
        # Fit individual models
        self.isolation_forest.fit(X)
        self.dbscan.fit(X)
        
        self.is_fitted = True
        logger.info("Combined clustering training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using combined approach.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        scores = self.predict_proba(X)
        
        if self.threshold_ is None:
            threshold = 0.5
        else:
            threshold = self.threshold_
        
        return (scores > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict combined anomaly scores.
        
        Args:
            X: Input features
            
        Returns:
            Combined anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        
        # Get scores from individual models
        if_scores = self.isolation_forest.predict_proba(X)
        dbscan_scores = self.dbscan.predict_proba(X)
        
        # Combine scores using weighted average
        combined_scores = (
            self.weights[0] * if_scores +
            self.weights[1] * dbscan_scores
        )
        
        return combined_scores


def create_clustering_model(config: Dict[str, Any], model_type: str = 'combined') -> UnsupervisedAnomalyDetector:
    """
    Create clustering-based anomaly detection model.
    
    Args:
        config: Configuration dictionary
        model_type: Type of clustering model ('isolation_forest', 'dbscan', 'hdbscan', 'combined')
        
    Returns:
        Clustering anomaly detector
    """
    if model_type == 'isolation_forest':
        return IsolationForestAnomalyDetector(config)
    elif model_type == 'dbscan':
        return DBSCANAnomalyDetector(config)
    elif model_type == 'hdbscan':
        return HDBSCANAnomalyDetector(config)
    elif model_type == 'combined':
        return CombinedClusteringAnomalyDetector(config)
    else:
        raise ValueError(f"Unknown clustering model type: {model_type}")


class StatisticalAnomalyDetector(UnsupervisedAnomalyDetector):
    """
    Statistical anomaly detector combining multiple statistical methods.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize statistical anomaly detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Statistical model configurations
        stat_config = self.config.get('models', {}).get('statistical', {})
        
        # Initialize models
        self.models = {}
        
        # One-Class SVM
        svm_config = stat_config.get('one_class_svm', {})
        self.models['svm'] = OneClassSVM(
            kernel=svm_config.get('kernel', 'rbf'),
            gamma=svm_config.get('gamma', 'scale'),
            nu=svm_config.get('nu', 0.1)
        )
        
        # Local Outlier Factor
        lof_config = stat_config.get('lof', {})
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=lof_config.get('n_neighbors', 20),
            algorithm=lof_config.get('algorithm', 'auto'),
            leaf_size=lof_config.get('leaf_size', 30),
            contamination=lof_config.get('contamination', 0.1),
            novelty=True  # For prediction on new data
        )
        
        # Elliptic Envelope
        ee_config = stat_config.get('elliptic_envelope', {})
        self.models['elliptic'] = EllipticEnvelope(
            contamination=ee_config.get('contamination', 0.1),
            support_fraction=ee_config.get('support_fraction', None),
            random_state=ee_config.get('random_state', 42)
        )
        
        self.scaler = StandardScaler()
        self.weights = [0.4, 0.3, 0.3]  # [svm, lof, elliptic]
        
        logger.info("Statistical Anomaly Detector initialized")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StatisticalAnomalyDetector':
        """
        Fit all statistical models.
        
        Args:
            X: Training features
            y: Ignored (unsupervised)
            
        Returns:
            Self for method chaining
        """
        X = self.validate_input(X)
        
        logger.info("Fitting statistical models")
        
        # Preprocessing
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit all models
        for name, model in self.models.items():
            try:
                model.fit(X_scaled)
                logger.info(f"Fitted {name} model")
            except Exception as e:
                logger.warning(f"Failed to fit {name} model: {e}")
        
        self.is_fitted = True
        logger.info("Statistical models training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using combined statistical approach.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions
        """
        scores = self.predict_proba(X)
        
        if self.threshold_ is None:
            threshold = 0.5
        else:
            threshold = self.threshold_
        
        return (scores > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict combined anomaly scores.
        
        Args:
            X: Input features
            
        Returns:
            Combined anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self.validate_input(X)
        X_scaled = self.scaler.transform(X)
        
        scores = []
        
        for name, model in self.models.items():
            try:
                if name == 'svm':
                    # SVM decision function (more negative = more anomalous)
                    model_scores = -model.decision_function(X_scaled)
                elif name == 'lof':
                    # LOF negative outlier factor (more negative = more anomalous)
                    model_scores = -model.decision_function(X_scaled)
                elif name == 'elliptic':
                    # Elliptic envelope decision function
                    model_scores = -model.decision_function(X_scaled)
                
                # Normalize to [0, 1]
                model_scores = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min() + 1e-8)
                scores.append(model_scores)
                
            except Exception as e:
                logger.warning(f"Failed to predict with {name} model: {e}")
                scores.append(np.zeros(len(X)))
        
        # Combine scores
        if scores:
            combined_scores = np.average(scores, axis=0, weights=self.weights[:len(scores)])
        else:
            combined_scores = np.zeros(len(X))
        
        return combined_scores 