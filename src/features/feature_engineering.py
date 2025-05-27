"""
북한 의심 선박 탐지를 위한 고급 피처 엔지니어링
AIS 데이터로부터 의심 행동 패턴을 탐지하는 피처들을 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class MaritimeFeatureEngineer:
    """해양 선박 데이터 피처 엔지니어링"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 피처 생성"""
        print("피처 엔지니어링 시작...")
        
        # 기본 전처리
        df = self._preprocess_data(df)
        
        # 시계열 피처
        df = self._create_temporal_features(df)
        
        # 궤적 분석 피처
        df = self._create_trajectory_features(df)
        
        # 행동 패턴 피처
        df = self._create_behavioral_features(df)
        
        # 지리적 피처
        df = self._create_geographic_features(df)
        
        # 이상 탐지 피처
        df = self._create_anomaly_features(df)
        
        # 통계적 피처
        df = self._create_statistical_features(df)
        
        # 클러스터링 피처
        df = self._create_clustering_features(df)
        
        print(f"피처 엔지니어링 완료. 총 {len(df.columns)} 개 피처")
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리"""
        df = df.copy()
        
        # 시간 정렬
        df = df.sort_values(['vessel_id', 'datetime']).reset_index(drop=True)
        
        # 결측값 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # 이상값 처리 (속도)
        df.loc[df['sog'] > 50, 'sog'] = 50  # 최대 속도 제한
        df.loc[df['sog'] < 0, 'sog'] = 0    # 음수 속도 제거
        
        # 코스 각도 정규화 (0-360)
        df['cog'] = df['cog'] % 360
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 관련 피처 생성"""
        df = df.copy()
        
        # 기본 시간 피처
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day'] = df['datetime'].dt.day
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # 시간대 분류
        df['time_period'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['night', 'morning', 'afternoon', 'evening'],
                                  include_lowest=True)
        
        # 계절 분류
        df['season'] = pd.cut(df['month'], 
                             bins=[0, 3, 6, 9, 12], 
                             labels=['winter', 'spring', 'summer', 'autumn'],
                             include_lowest=True)
        
        # 주말/평일
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 야간 활동 (22시-6시)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # 시간 간격 계산
        vessel_groups = df.groupby('vessel_id')
        df['time_diff'] = vessel_groups['datetime'].diff().dt.total_seconds() / 60  # 분 단위
        df['time_diff'] = df['time_diff'].fillna(0)
        
        return df
    
    def _create_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """궤적 분석 피처 생성"""
        df = df.copy()
        vessel_groups = df.groupby('vessel_id')
        
        # 위치 변화
        df['lat_diff'] = vessel_groups['lat'].diff().fillna(0)
        df['lon_diff'] = vessel_groups['lon'].diff().fillna(0)
        
        # 거리 계산 (Haversine)
        df['distance_km'] = self._calculate_haversine_distance(df)
        
        # 속도 계산 (실제 이동 거리 기반)
        df['calculated_speed'] = np.where(df['time_diff'] > 0, 
                                        df['distance_km'] / (df['time_diff'] / 60), 0)
        
        # 속도 차이 (보고된 속도 vs 계산된 속도)
        df['speed_discrepancy'] = abs(df['sog'] - df['calculated_speed'])
        
        # 방향 변화
        df['bearing'] = self._calculate_bearing(df)
        df['bearing_change'] = vessel_groups['bearing'].diff().fillna(0)
        
        # 360도 경계 처리
        df.loc[df['bearing_change'] > 180, 'bearing_change'] -= 360
        df.loc[df['bearing_change'] < -180, 'bearing_change'] += 360
        
        # 궤적 복잡도 - 선박별로 처리
        windows = self.config['feature_engineering']['time_windows']
        for window in windows:
            df[f'trajectory_complexity_{window}'] = 0.0
        
        # 선박별로 궤적 복잡도 계산
        for vessel_id in df['vessel_id'].unique():
            vessel_mask = df['vessel_id'] == vessel_id
            vessel_data = df[vessel_mask].copy()
            
            if len(vessel_data) > 1:
                for window in windows:
                    if len(vessel_data) >= window:
                        complexity = vessel_data['bearing_change'].rolling(
                            window=window, min_periods=1).std().fillna(0)
                        df.loc[vessel_mask, f'trajectory_complexity_{window}'] = complexity.values
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """행동 패턴 피처 생성"""
        df = df.copy()
        vessel_groups = df.groupby('vessel_id')
        
        # 속도 패턴
        df['speed_change'] = vessel_groups['sog'].diff().fillna(0)
        df['speed_acceleration'] = vessel_groups['speed_change'].diff().fillna(0)
        
        # 코스 패턴
        df['course_change'] = vessel_groups['cog'].diff().fillna(0)
        # 360도 경계 처리
        df.loc[df['course_change'] > 180, 'course_change'] -= 360
        df.loc[df['course_change'] < -180, 'course_change'] += 360
        
        # 머무름 행동 (loitering)
        df['is_loitering'] = (df['sog'] < 1.0).astype(int)
        df['is_slow_moving'] = (df['sog'] < 3.0).astype(int)
        df['is_fast_moving'] = (df['sog'] > 15.0).astype(int)
        
        # 윈도우별 통계 - 선박별로 처리
        windows = self.config['feature_engineering']['time_windows']
        
        # 초기화
        for window in windows:
            df[f'speed_mean_{window}'] = 0.0
            df[f'speed_std_{window}'] = 0.0
            df[f'speed_max_{window}'] = 0.0
            df[f'speed_min_{window}'] = 0.0
            df[f'course_change_mean_{window}'] = 0.0
            df[f'course_change_std_{window}'] = 0.0
            df[f'loitering_ratio_{window}'] = 0.0
            df[f'night_activity_ratio_{window}'] = 0.0
        
        # 선박별로 윈도우 통계 계산
        for vessel_id in df['vessel_id'].unique():
            vessel_mask = df['vessel_id'] == vessel_id
            vessel_data = df[vessel_mask].copy()
            
            if len(vessel_data) > 1:
                for window in windows:
                    if len(vessel_data) >= window:
                        # 속도 통계
                        df.loc[vessel_mask, f'speed_mean_{window}'] = vessel_data['sog'].rolling(
                            window=window, min_periods=1).mean().values
                        df.loc[vessel_mask, f'speed_std_{window}'] = vessel_data['sog'].rolling(
                            window=window, min_periods=1).std().fillna(0).values
                        df.loc[vessel_mask, f'speed_max_{window}'] = vessel_data['sog'].rolling(
                            window=window, min_periods=1).max().values
                        df.loc[vessel_mask, f'speed_min_{window}'] = vessel_data['sog'].rolling(
                            window=window, min_periods=1).min().values
                        
                        # 코스 변화 통계
                        df.loc[vessel_mask, f'course_change_mean_{window}'] = vessel_data['course_change'].rolling(
                            window=window, min_periods=1).mean().values
                        df.loc[vessel_mask, f'course_change_std_{window}'] = vessel_data['course_change'].rolling(
                            window=window, min_periods=1).std().fillna(0).values
                        
                        # 머무름 비율
                        df.loc[vessel_mask, f'loitering_ratio_{window}'] = vessel_data['is_loitering'].rolling(
                            window=window, min_periods=1).mean().values
                        
                        # 야간 활동 비율
                        df.loc[vessel_mask, f'night_activity_ratio_{window}'] = vessel_data['is_night'].rolling(
                            window=window, min_periods=1).mean().values
        
        return df
    
    def _create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """지리적 피처 생성"""
        df = df.copy()
        
        # 북한 수역 근접도
        nk_waters = self.config['feature_engineering']['geographic_zones']['north_korea_waters']
        df['nk_proximity'] = self._calculate_zone_proximity(df, nk_waters)
        df['in_nk_waters'] = ((df['lat'] >= nk_waters['lat_min']) & 
                             (df['lat'] <= nk_waters['lat_max']) &
                             (df['lon'] >= nk_waters['lon_min']) & 
                             (df['lon'] <= nk_waters['lon_max'])).astype(int)
        
        # 어업 구역 근접도
        fishing_zones = self.config['feature_engineering']['geographic_zones']['fishing_zones']
        df['fishing_zone_proximity'] = 0
        for zone in fishing_zones:
            proximity = self._calculate_zone_proximity(df, zone)
            df['fishing_zone_proximity'] = np.maximum(df['fishing_zone_proximity'], proximity)
        
        # 해안선으로부터의 거리 (근사치)
        df['distance_to_coast'] = np.minimum(
            np.abs(df['lat'] - 35.0),  # 한국 남해안 근사
            np.abs(df['lon'] - 126.0)  # 서해안 근사
        )
        
        # 위치 클러스터링
        df['lat_cluster'] = pd.cut(df['lat'], bins=10, labels=False)
        df['lon_cluster'] = pd.cut(df['lon'], bins=10, labels=False)
        
        return df
    
    def _create_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """이상 행동 탐지 피처"""
        df = df.copy()
        vessel_groups = df.groupby('vessel_id')
        
        # 급격한 변화 탐지
        thresholds = self.config['feature_engineering']['anomaly_thresholds']
        
        df['sudden_speed_change'] = (abs(df['speed_change']) > thresholds['speed_change_rate']).astype(int)
        df['sudden_course_change'] = (abs(df['course_change']) > thresholds['course_change_rate']).astype(int)
        
        # 연속 머무름 시간 - 선박별로 처리
        df['loitering_duration'] = 0.0
        df['extended_loitering'] = 0
        df['speed_outlier'] = 0
        df['zigzag_pattern'] = 0.0
        
        for vessel_id in df['vessel_id'].unique():
            vessel_mask = df['vessel_id'] == vessel_id
            vessel_data = df[vessel_mask].copy()
            
            if len(vessel_data) > 1:
                # 연속 머무름 시간
                loitering_duration = vessel_data['is_loitering'].rolling(
                    window=60, min_periods=1).sum()  # 1시간 윈도우
                df.loc[vessel_mask, 'loitering_duration'] = loitering_duration.values
                df.loc[vessel_mask, 'extended_loitering'] = (loitering_duration > thresholds['loitering_time']).astype(int).values
                
                # 지그재그 패턴 (빈번한 코스 변화)
                zigzag = vessel_data['course_change'].rolling(
                    window=10, min_periods=1).apply(lambda x: (abs(x) > 30).sum())
                df.loc[vessel_mask, 'zigzag_pattern'] = zigzag.values
        
        # 비정상적인 속도 패턴
        df['speed_outlier'] = self._detect_outliers(df, 'sog', vessel_groups)
        
        # AIS 신호 간격 이상
        df['ais_gap_anomaly'] = (df['time_diff'] > 30).astype(int)  # 30분 이상 간격
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """통계적 피처 생성"""
        df = df.copy()
        vessel_groups = df.groupby('vessel_id')
        
        # 전체 여행 통계 (선박별)
        vessel_stats = vessel_groups.agg({
            'sog': ['mean', 'std', 'min', 'max', 'median'],
            'cog': ['std'],
            'distance_km': ['sum'],
            'is_night': ['mean'],
            'is_loitering': ['mean'],
            'nk_proximity': ['mean', 'max'],
            'time_diff': ['mean', 'std']
        }).round(4)
        
        # 컬럼명 평탄화
        vessel_stats.columns = ['_'.join(col).strip() for col in vessel_stats.columns]
        vessel_stats = vessel_stats.add_prefix('vessel_')
        
        # 원본 데이터에 병합
        df = df.merge(vessel_stats, left_on='vessel_id', right_index=True, how='left')
        
        # 현재 값과 선박 평균의 차이
        df['speed_vs_vessel_mean'] = df['sog'] - df['vessel_sog_mean']
        df['proximity_vs_vessel_mean'] = df['nk_proximity'] - df['vessel_nk_proximity_mean']
        
        return df
    
    def _create_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """클러스터링 기반 피처"""
        df = df.copy()
        
        # 행동 패턴 클러스터링
        behavior_features = ['sog', 'course_change', 'nk_proximity', 'is_night']
        behavior_data = df[behavior_features].fillna(0)
        
        # 정규화
        behavior_scaled = StandardScaler().fit_transform(behavior_data)
        
        # DBSCAN 클러스터링
        dbscan = DBSCAN(eps=0.5, min_samples=10)
        df['behavior_cluster'] = dbscan.fit_predict(behavior_scaled)
        
        # 클러스터 크기
        cluster_sizes = df['behavior_cluster'].value_counts()
        df['cluster_size'] = df['behavior_cluster'].map(cluster_sizes)
        
        # 이상치 클러스터 (-1)
        df['is_behavior_outlier'] = (df['behavior_cluster'] == -1).astype(int)
        
        return df
    
    def _calculate_haversine_distance(self, df: pd.DataFrame) -> pd.Series:
        """Haversine 거리 계산"""
        def haversine_vectorized(lat1, lon1, lat2, lon2):
            R = 6371  # 지구 반지름 (km)
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            
            return R * c
        
        distances = []
        for vessel_id in df['vessel_id'].unique():
            vessel_data = df[df['vessel_id'] == vessel_id].sort_values('datetime')
            
            if len(vessel_data) > 1:
                dist = haversine_vectorized(
                    vessel_data['lat'].iloc[:-1].values,
                    vessel_data['lon'].iloc[:-1].values,
                    vessel_data['lat'].iloc[1:].values,
                    vessel_data['lon'].iloc[1:].values
                )
                dist = np.concatenate([[0], dist])
            else:
                dist = [0]
            
            distances.extend(dist)
        
        return pd.Series(distances, index=df.index)
    
    def _calculate_bearing(self, df: pd.DataFrame) -> pd.Series:
        """방위각 계산"""
        def bearing_vectorized(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            
            y = np.sin(dlon) * np.cos(lat2)
            x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
            
            bearing = np.arctan2(y, x)
            bearing = np.degrees(bearing)
            bearing = (bearing + 360) % 360
            
            return bearing
        
        bearings = []
        for vessel_id in df['vessel_id'].unique():
            vessel_data = df[df['vessel_id'] == vessel_id].sort_values('datetime')
            
            if len(vessel_data) > 1:
                bear = bearing_vectorized(
                    vessel_data['lat'].iloc[:-1].values,
                    vessel_data['lon'].iloc[:-1].values,
                    vessel_data['lat'].iloc[1:].values,
                    vessel_data['lon'].iloc[1:].values
                )
                bear = np.concatenate([[0], bear])
            else:
                bear = [0]
            
            bearings.extend(bear)
        
        return pd.Series(bearings, index=df.index)
    
    def _calculate_zone_proximity(self, df: pd.DataFrame, zone: Dict) -> pd.Series:
        """특정 구역에 대한 근접도 계산"""
        center_lat = (zone['lat_min'] + zone['lat_max']) / 2
        center_lon = (zone['lon_min'] + zone['lon_max']) / 2
        
        distances = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
        proximity = 1 / (1 + distances)
        
        return proximity
    
    def _detect_outliers(self, df: pd.DataFrame, column: str, groups) -> pd.Series:
        """이상치 탐지 (IQR 방법)"""
        def detect_outliers_group(group):
            Q1 = group[column].quantile(0.25)
            Q3 = group[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            return ((group[column] < lower_bound) | (group[column] > upper_bound)).astype(int)
        
        outliers = groups.apply(detect_outliers_group)
        return outliers.reset_index(level=0, drop=True)
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'is_suspicious') -> pd.DataFrame:
        """피처 선택"""
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        # 수치형 피처만 선택
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟과 ID 컬럼 제외
        exclude_cols = [target_col, 'vessel_id', 'confidence']
        feature_cols = [col for col in numeric_features if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # 피처 선택
        k_best = self.config['training']['feature_selection']['k_best']
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k_best, len(feature_cols)))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        print(f"선택된 피처 수: {len(selected_features)}")
        
        # 선택된 피처로 데이터프레임 재구성
        result_df = df[['vessel_id', target_col, 'confidence'] + selected_features].copy()
        
        return result_df, selected_features
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, selected_features: List[str], 
                                      target_col: str = 'is_suspicious') -> Dict:
        """피처 중요도 분석"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        X = df[selected_features].fillna(0)
        y = df[target_col]
        
        # Random Forest 피처 중요도
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(selected_features, rf.feature_importances_))
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y)
        mi_importance = dict(zip(selected_features, mi_scores))
        
        # 상관관계
        correlation = df[selected_features + [target_col]].corr()[target_col].abs().to_dict()
        
        return {
            'random_forest': rf_importance,
            'mutual_information': mi_importance,
            'correlation': correlation
        } 