"""
Feature engineering for AIS maritime data.

This module extracts comprehensive features from AIS data including:
- Kinematic features (speed, acceleration, direction changes)
- Geographic features (distance, trajectory complexity)
- Temporal features (duration, periodicity)
- Behavioral features (maneuvers, stops)
- TrAISformer features (four-hot encoding, entropy)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from geopy.distance import geodesic
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class AISFeatureExtractor:
    """
    Comprehensive feature extractor for AIS maritime data.
    
    Extracts kinematic, geographic, temporal, behavioral, and TrAISformer features
    from vessel trajectory data.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration dictionary with feature extraction parameters
        """
        self.config = config or {}
        self.feature_config = self.config.get('features', {})
        
        # Default parameters
        self.speed_threshold = self.feature_config.get('kinematic', {}).get('speed_threshold', 0.1)
        self.acceleration_threshold = self.feature_config.get('kinematic', {}).get('acceleration_threshold', 0.5)
        self.window_size = self.feature_config.get('kinematic', {}).get('window_size', 10)
        self.distance_threshold = self.feature_config.get('geographic', {}).get('distance_threshold', 1000)
        self.coastal_distance = self.feature_config.get('geographic', {}).get('coastal_distance', 5000)
        self.maneuver_threshold = self.feature_config.get('behavioral', {}).get('maneuver_threshold', 30)
        self.stop_threshold = self.feature_config.get('behavioral', {}).get('stop_threshold', 0.5)
        
        # TrAISformer encoding parameters
        trais_config = self.feature_config.get('traisformer', {})
        self.lat_bins = trais_config.get('lat_bins', 100)
        self.lon_bins = trais_config.get('lon_bins', 100)
        self.speed_bins = trais_config.get('speed_bins', 50)
        self.course_bins = trais_config.get('course_bins', 36)
        
        logger.info("AIS Feature Extractor initialized")
    
    def extract_features(self, df: pd.DataFrame, vessel_id_col: str = 'MMSI') -> pd.DataFrame:
        """
        Extract all features from AIS data.
        
        Args:
            df: AIS DataFrame with required columns
            vessel_id_col: Column name for vessel identifier
            
        Returns:
            DataFrame with extracted features per vessel
        """
        logger.info(f"Extracting features for {df[vessel_id_col].nunique()} vessels")
        
        # Validate required columns
        self._validate_columns(df)
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        features_list = []
        
        for vessel_id in df[vessel_id_col].unique():
            vessel_data = df[df[vessel_id_col] == vessel_id].copy()
            vessel_data = vessel_data.sort_values('Timestamp')
            
            if len(vessel_data) < 2:
                logger.warning(f"Vessel {vessel_id} has insufficient data points")
                continue
            
            try:
                vessel_features = self._extract_vessel_features(vessel_data, vessel_id)
                features_list.append(vessel_features)
            except Exception as e:
                logger.error(f"Error extracting features for vessel {vessel_id}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No features extracted from any vessel")
        
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} vessels")
        
        return features_df
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist in the DataFrame."""
        required_cols = [
            'Timestamp', 'MMSI', 'Latitude', 'Longitude', 'SOG', 'COG',
            'Navigational status', 'ROT', 'Heading'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess AIS data."""
        df = df.copy()
        
        # Convert timestamp
        if df['Timestamp'].dtype == 'object':
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M:%S')
        
        # Handle missing values
        numeric_cols = ['Latitude', 'Longitude', 'SOG', 'COG', 'ROT', 'Heading']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid coordinates
        df = df[(df['Latitude'].between(-90, 90)) & (df['Longitude'].between(-180, 180))]
        
        # Remove duplicate timestamps per vessel
        df = df.drop_duplicates(subset=['MMSI', 'Timestamp'])
        
        return df
    
    def _extract_vessel_features(self, vessel_data: pd.DataFrame, vessel_id: str) -> Dict:
        """Extract all features for a single vessel."""
        features = {'vessel_id': vessel_id}
        
        # Extract different types of features
        if self.feature_config.get('kinematic', {}).get('enable', True):
            features.update(self._extract_kinematic_features(vessel_data))
        
        if self.feature_config.get('geographic', {}).get('enable', True):
            features.update(self._extract_geographic_features(vessel_data))
        
        if self.feature_config.get('temporal', {}).get('enable', True):
            features.update(self._extract_temporal_features(vessel_data))
        
        if self.feature_config.get('behavioral', {}).get('enable', True):
            features.update(self._extract_behavioral_features(vessel_data))
        
        if self.feature_config.get('traisformer', {}).get('enable', True):
            features.update(self._extract_traisformer_features(vessel_data))
        
        # Extract vessel characteristics
        features.update(self._extract_vessel_characteristics(vessel_data))
        
        return features
    
    def _extract_kinematic_features(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract kinematic features (speed, acceleration, direction changes)."""
        features = {}
        
        # Speed statistics
        speeds = vessel_data['SOG'].dropna()
        if len(speeds) > 0:
            features.update({
                'speed_mean': speeds.mean(),
                'speed_std': speeds.std(),
                'speed_max': speeds.max(),
                'speed_min': speeds.min(),
                'speed_median': speeds.median(),
                'speed_q75': speeds.quantile(0.75),
                'speed_q25': speeds.quantile(0.25),
                'speed_range': speeds.max() - speeds.min(),
                'speed_cv': speeds.std() / speeds.mean() if speeds.mean() > 0 else 0
            })
        
        # Calculate acceleration
        if len(vessel_data) > 1:
            time_diffs = vessel_data['Timestamp'].diff().dt.total_seconds()
            speed_diffs = vessel_data['SOG'].diff()
            
            # Avoid division by zero
            valid_mask = (time_diffs > 0) & (~speed_diffs.isna())
            if valid_mask.sum() > 0:
                accelerations = speed_diffs[valid_mask] / time_diffs[valid_mask]
                
                features.update({
                    'acceleration_mean': accelerations.mean(),
                    'acceleration_std': accelerations.std(),
                    'acceleration_max': accelerations.max(),
                    'acceleration_min': accelerations.min(),
                    'max_acceleration': accelerations.abs().max(),
                    'acceleration_changes': (accelerations.abs() > self.acceleration_threshold).sum()
                })
        
        # Course changes
        courses = vessel_data['COG'].dropna()
        if len(courses) > 1:
            course_diffs = np.diff(courses)
            # Handle circular nature of course (0-360 degrees)
            course_diffs = np.where(course_diffs > 180, course_diffs - 360, course_diffs)
            course_diffs = np.where(course_diffs < -180, course_diffs + 360, course_diffs)
            
            features.update({
                'course_change_mean': np.mean(np.abs(course_diffs)),
                'course_change_std': np.std(course_diffs),
                'course_change_max': np.max(np.abs(course_diffs)),
                'total_course_change': np.sum(np.abs(course_diffs)),
                'sharp_turns': (np.abs(course_diffs) > self.maneuver_threshold).sum()
            })
        
        # Rate of Turn statistics
        rot_values = vessel_data['ROT'].dropna()
        if len(rot_values) > 0:
            features.update({
                'rot_mean': rot_values.mean(),
                'rot_std': rot_values.std(),
                'rot_max': rot_values.max(),
                'rot_min': rot_values.min(),
                'rot_abs_mean': rot_values.abs().mean()
            })
        
        return features
    
    def _extract_geographic_features(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract geographic features (distances, trajectory complexity)."""
        features = {}
        
        # Calculate distances between consecutive points
        distances = []
        for i in range(1, len(vessel_data)):
            try:
                dist = geodesic(
                    (vessel_data.iloc[i-1]['Latitude'], vessel_data.iloc[i-1]['Longitude']),
                    (vessel_data.iloc[i]['Latitude'], vessel_data.iloc[i]['Longitude'])
                ).meters
                distances.append(dist)
            except:
                continue
        
        if distances:
            distances = np.array(distances)
            
            # Distance statistics
            features.update({
                'total_distance': distances.sum(),
                'mean_distance_between_points': distances.mean(),
                'max_distance_between_points': distances.max(),
                'distance_std': distances.std()
            })
            
            # Straight line distance
            start_point = (vessel_data.iloc[0]['Latitude'], vessel_data.iloc[0]['Longitude'])
            end_point = (vessel_data.iloc[-1]['Latitude'], vessel_data.iloc[-1]['Longitude'])
            straight_distance = geodesic(start_point, end_point).meters
            
            features['straight_line_distance'] = straight_distance
            features['straightness_index'] = straight_distance / distances.sum() if distances.sum() > 0 else 0
        
        # Bounding box features
        lats = vessel_data['Latitude']
        lons = vessel_data['Longitude']
        
        features.update({
            'lat_range': lats.max() - lats.min(),
            'lon_range': lons.max() - lons.min(),
            'bounding_box_area': (lats.max() - lats.min()) * (lons.max() - lons.min()),
            'lat_center': lats.mean(),
            'lon_center': lons.mean()
        })
        
        # Trajectory complexity (sinuosity)
        if len(distances) > 2:
            # Calculate sinuosity as ratio of actual path to straight line
            features['sinuosity'] = distances.sum() / straight_distance if straight_distance > 0 else 1
            
            # Fractal dimension approximation
            features['fractal_dimension'] = self._calculate_fractal_dimension(
                vessel_data[['Latitude', 'Longitude']].values
            )
        
        return features
    
    def _extract_temporal_features(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract temporal features (duration, periodicity, time patterns)."""
        features = {}
        
        # Journey duration
        start_time = vessel_data['Timestamp'].min()
        end_time = vessel_data['Timestamp'].max()
        duration = (end_time - start_time).total_seconds()
        
        features.update({
            'journey_duration_hours': duration / 3600,
            'journey_duration_days': duration / (3600 * 24),
            'num_data_points': len(vessel_data),
            'data_point_frequency': len(vessel_data) / (duration / 3600) if duration > 0 else 0
        })
        
        # Time intervals between data points
        time_diffs = vessel_data['Timestamp'].diff().dt.total_seconds().dropna()
        if len(time_diffs) > 0:
            features.update({
                'mean_time_interval': time_diffs.mean(),
                'std_time_interval': time_diffs.std(),
                'max_time_gap': time_diffs.max(),
                'min_time_interval': time_diffs.min()
            })
        
        # Time of day analysis
        hours = vessel_data['Timestamp'].dt.hour
        features.update({
            'night_hours_ratio': ((hours >= 22) | (hours <= 6)).mean(),
            'day_hours_ratio': ((hours >= 6) & (hours <= 18)).mean(),
            'evening_hours_ratio': ((hours >= 18) & (hours <= 22)).mean()
        })
        
        # Day of week analysis
        weekdays = vessel_data['Timestamp'].dt.weekday
        features.update({
            'weekday_ratio': (weekdays < 5).mean(),
            'weekend_ratio': (weekdays >= 5).mean()
        })
        
        # Periodicity analysis using FFT
        if len(vessel_data) > 10:
            try:
                # Analyze speed periodicity
                speeds = vessel_data['SOG'].interpolate().values
                if len(speeds) > 0 and not np.all(speeds == speeds[0]):
                    fft_speeds = np.abs(fft(speeds))
                    dominant_freq_idx = np.argmax(fft_speeds[1:len(fft_speeds)//2]) + 1
                    features['speed_dominant_frequency'] = dominant_freq_idx
                    features['speed_periodicity_strength'] = fft_speeds[dominant_freq_idx] / np.sum(fft_speeds)
            except:
                features['speed_dominant_frequency'] = 0
                features['speed_periodicity_strength'] = 0
        
        return features
    
    def _extract_behavioral_features(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract behavioral features (navigation patterns, stops, maneuvers)."""
        features = {}
        
        # Navigation status analysis
        if 'Navigational status' in vessel_data.columns:
            nav_status = vessel_data['Navigational status'].value_counts(normalize=True)
            
            # Common navigation statuses
            status_mapping = {
                'Under way using engine': 'underway_engine',
                'At anchor': 'anchored',
                'Not under command': 'not_under_command',
                'Restricted manoeuvrability': 'restricted_maneuver',
                'Moored': 'moored'
            }
            
            for status, feature_name in status_mapping.items():
                features[f'{feature_name}_ratio'] = nav_status.get(status, 0)
        
        # Stop detection (low speed periods)
        low_speed_mask = vessel_data['SOG'] <= self.stop_threshold
        if len(low_speed_mask) > 0:
            features['stop_ratio'] = low_speed_mask.mean()
            
            # Count stop events
            stop_events = 0
            in_stop = False
            for is_stopped in low_speed_mask:
                if is_stopped and not in_stop:
                    stop_events += 1
                    in_stop = True
                elif not is_stopped:
                    in_stop = False
            
            features['num_stop_events'] = stop_events
        
        # Maneuver detection
        if len(vessel_data) > 1:
            course_changes = np.abs(np.diff(vessel_data['COG'].fillna(method='ffill')))
            course_changes = np.minimum(course_changes, 360 - course_changes)  # Handle circular nature
            
            sharp_maneuvers = course_changes > self.maneuver_threshold
            features.update({
                'sharp_maneuver_count': sharp_maneuvers.sum(),
                'sharp_maneuver_ratio': sharp_maneuvers.mean(),
                'max_course_change': course_changes.max() if len(course_changes) > 0 else 0
            })
            
            # U-turn detection (course change > 150 degrees)
            u_turns = course_changes > 150
            features['u_turn_count'] = u_turns.sum()
        
        # Speed variability patterns
        speeds = vessel_data['SOG'].dropna()
        if len(speeds) > 1:
            speed_changes = np.abs(np.diff(speeds))
            features.update({
                'speed_variability': speed_changes.std(),
                'sudden_speed_changes': (speed_changes > speeds.mean()).sum(),
                'speed_consistency': 1 - (speed_changes.std() / speeds.mean()) if speeds.mean() > 0 else 0
            })
        
        # Irregular behavior score (composite metric)
        irregular_score = 0
        if 'sharp_maneuver_ratio' in features:
            irregular_score += features['sharp_maneuver_ratio']
        if 'speed_variability' in features and speeds.mean() > 0:
            irregular_score += features['speed_variability'] / speeds.mean()
        if 'u_turn_count' in features:
            irregular_score += features['u_turn_count'] / len(vessel_data)
        
        features['irregular_behavior_score'] = irregular_score
        
        return features
    
    def _extract_traisformer_features(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract TrAISformer-style features (four-hot encoding, entropy)."""
        features = {}
        
        # Four-hot encoding bins
        lats = vessel_data['Latitude'].dropna()
        lons = vessel_data['Longitude'].dropna()
        speeds = vessel_data['SOG'].dropna()
        courses = vessel_data['COG'].dropna()
        
        if len(lats) > 0:
            # Latitude binning
            lat_min, lat_max = lats.min(), lats.max()
            if lat_max > lat_min:
                lat_bins = np.linspace(lat_min, lat_max, self.lat_bins)
                lat_digitized = np.digitize(lats, lat_bins)
                features['lat_entropy'] = self._calculate_entropy(lat_digitized)
                features['lat_unique_bins'] = len(np.unique(lat_digitized))
            
            # Longitude binning
            lon_min, lon_max = lons.min(), lons.max()
            if lon_max > lon_min:
                lon_bins = np.linspace(lon_min, lon_max, self.lon_bins)
                lon_digitized = np.digitize(lons, lon_bins)
                features['lon_entropy'] = self._calculate_entropy(lon_digitized)
                features['lon_unique_bins'] = len(np.unique(lon_digitized))
        
        if len(speeds) > 0:
            # Speed binning
            speed_bins = np.linspace(0, speeds.max(), self.speed_bins)
            speed_digitized = np.digitize(speeds, speed_bins)
            features['speed_entropy'] = self._calculate_entropy(speed_digitized)
            features['speed_unique_bins'] = len(np.unique(speed_digitized))
        
        if len(courses) > 0:
            # Course binning (0-360 degrees)
            course_bins = np.linspace(0, 360, self.course_bins)
            course_digitized = np.digitize(courses, course_bins)
            features['course_entropy'] = self._calculate_entropy(course_digitized)
            features['course_unique_bins'] = len(np.unique(course_digitized))
        
        # Trajectory entropy (combined position entropy)
        if len(lats) > 0 and len(lons) > 0:
            # Create 2D position bins
            position_pairs = list(zip(lat_digitized[:len(lon_digitized)], 
                                    lon_digitized[:len(lat_digitized)]))
            features['trajectory_entropy'] = self._calculate_entropy(position_pairs)
        
        # Transition probabilities
        if len(vessel_data) > 1:
            features.update(self._calculate_transition_probabilities(vessel_data))
        
        return features
    
    def _extract_vessel_characteristics(self, vessel_data: pd.DataFrame) -> Dict:
        """Extract vessel characteristic features."""
        features = {}
        
        # Get vessel information from first row (assuming it's consistent)
        vessel_info = vessel_data.iloc[0]
        
        # Vessel dimensions
        if 'Length' in vessel_data.columns and pd.notna(vessel_info['Length']):
            features['vessel_length'] = vessel_info['Length']
        
        if 'Width' in vessel_data.columns and pd.notna(vessel_info['Width']):
            features['vessel_width'] = vessel_info['Width']
        
        if 'Length' in features and 'Width' in features:
            features['vessel_area'] = features['vessel_length'] * features['vessel_width']
            features['vessel_aspect_ratio'] = features['vessel_length'] / features['vessel_width']
        
        # Ship type encoding
        if 'Ship type' in vessel_data.columns:
            ship_type = vessel_info['Ship type']
            # Create binary features for common ship types
            ship_types = ['Cargo', 'Tanker', 'Passenger', 'Fishing', 'Tug', 'Military']
            for ship_type_name in ship_types:
                features[f'is_{ship_type_name.lower()}'] = int(ship_type_name.lower() in str(ship_type).lower())
        
        # Cargo type encoding
        if 'Cargo type' in vessel_data.columns:
            cargo_type = vessel_info['Cargo type']
            cargo_types = ['Container', 'Bulk', 'Oil', 'Gas', 'Chemical']
            for cargo_type_name in cargo_types:
                features[f'cargo_{cargo_type_name.lower()}'] = int(cargo_type_name.lower() in str(cargo_type).lower())
        
        return features
    
    def _calculate_fractal_dimension(self, trajectory: np.ndarray) -> float:
        """Calculate fractal dimension of trajectory using box-counting method."""
        try:
            if len(trajectory) < 3:
                return 1.0
            
            # Normalize trajectory
            trajectory = (trajectory - trajectory.min(axis=0)) / (trajectory.max(axis=0) - trajectory.min(axis=0))
            
            # Box-counting method
            scales = np.logspace(-2, 0, num=10)
            counts = []
            
            for scale in scales:
                # Create grid
                grid_size = int(1 / scale)
                if grid_size < 2:
                    continue
                
                # Count occupied boxes
                boxes = set()
                for point in trajectory:
                    box_x = int(point[0] * grid_size)
                    box_y = int(point[1] * grid_size)
                    boxes.add((box_x, box_y))
                
                counts.append(len(boxes))
            
            if len(counts) < 2:
                return 1.0
            
            # Calculate fractal dimension
            scales = scales[:len(counts)]
            log_scales = np.log(1 / scales)
            log_counts = np.log(counts)
            
            # Linear regression
            slope, _, _, _, _ = stats.linregress(log_scales, log_counts)
            return max(1.0, min(2.0, slope))  # Clamp between 1 and 2
            
        except:
            return 1.0
    
    def _calculate_entropy(self, sequence) -> float:
        """Calculate Shannon entropy of a sequence."""
        try:
            _, counts = np.unique(sequence, return_counts=True)
            probabilities = counts / len(sequence)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except:
            return 0.0
    
    def _calculate_transition_probabilities(self, vessel_data: pd.DataFrame) -> Dict:
        """Calculate state transition probabilities."""
        features = {}
        
        try:
            # Speed state transitions
            speeds = vessel_data['SOG'].dropna()
            if len(speeds) > 1:
                # Define speed states
                speed_states = pd.cut(speeds, bins=3, labels=['low', 'medium', 'high'])
                
                # Calculate transition matrix
                transitions = {}
                for i in range(len(speed_states) - 1):
                    current_state = speed_states.iloc[i]
                    next_state = speed_states.iloc[i + 1]
                    
                    if pd.notna(current_state) and pd.notna(next_state):
                        key = f"{current_state}_to_{next_state}"
                        transitions[key] = transitions.get(key, 0) + 1
                
                # Normalize to probabilities
                total_transitions = sum(transitions.values())
                if total_transitions > 0:
                    for key, count in transitions.items():
                        features[f'speed_transition_{key}'] = count / total_transitions
            
        except Exception as e:
            logger.warning(f"Error calculating transition probabilities: {e}")
        
        return features


def create_feature_pipeline(config: Dict) -> AISFeatureExtractor:
    """
    Create feature extraction pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured feature extractor
    """
    return AISFeatureExtractor(config)


def extract_features_batch(
    data_path: str,
    output_path: str,
    config: Dict,
    chunk_size: int = 10000
) -> None:
    """
    Extract features from large datasets in batches.
    
    Args:
        data_path: Path to input CSV file
        output_path: Path to save extracted features
        config: Configuration dictionary
        chunk_size: Number of vessels to process per batch
    """
    logger.info(f"Starting batch feature extraction from {data_path}")
    
    extractor = AISFeatureExtractor(config)
    
    # Read data in chunks
    chunk_iter = pd.read_csv(data_path, chunksize=chunk_size)
    
    all_features = []
    
    for i, chunk in enumerate(chunk_iter):
        logger.info(f"Processing chunk {i + 1}")
        
        try:
            features = extractor.extract_features(chunk)
            all_features.append(features)
        except Exception as e:
            logger.error(f"Error processing chunk {i + 1}: {e}")
            continue
    
    if all_features:
        # Combine all features
        final_features = pd.concat(all_features, ignore_index=True)
        
        # Save to file
        final_features.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        logger.info(f"Final feature shape: {final_features.shape}")
    else:
        logger.error("No features extracted from any chunk")
        raise ValueError("Feature extraction failed for all chunks") 