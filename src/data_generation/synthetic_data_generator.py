"""
북한 의심 선박 탐지를 위한 가상 데이터 생성기
기존 AIS 데이터를 기반으로 정상/의심 선박 데이터를 생성
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path

class SyntheticVesselDataGenerator:
    """북한 의심 선박 탐지를 위한 가상 데이터 생성기"""
    
    def __init__(self, config_path: str = "config/competition_config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 북한 의심 선박 행동 패턴 정의
        self.suspicious_patterns = {
            'illegal_fishing': {
                'speed_range': (0.5, 3.0),  # 매우 느린 속도
                'course_stability': 'high_variation',  # 코스 변화가 큼
                'loitering_time': 180,  # 3시간 이상 머무름
                'night_activity': True,  # 야간 활동
                'zone_proximity': 'north_korea_waters'
            },
            'smuggling': {
                'speed_range': (8.0, 25.0),  # 빠른 속도
                'course_stability': 'direct',  # 직선 코스
                'meeting_behavior': True,  # 다른 선박과 만남
                'ais_gaps': True,  # AIS 신호 끊김
                'unusual_routes': True
            },
            'surveillance': {
                'speed_range': (5.0, 15.0),  # 중간 속도
                'patrol_pattern': True,  # 순찰 패턴
                'border_proximity': True,  # 경계선 근처
                'long_duration': True  # 장시간 활동
            }
        }
        
        # 정상 선박 행동 패턴
        self.normal_patterns = {
            'commercial_fishing': {
                'speed_range': (2.0, 8.0),
                'course_stability': 'moderate',
                'fishing_zones': True,
                'regular_schedule': True
            },
            'cargo_transport': {
                'speed_range': (10.0, 20.0),
                'course_stability': 'stable',
                'port_to_port': True,
                'scheduled_routes': True
            },
            'passenger_ferry': {
                'speed_range': (15.0, 30.0),
                'course_stability': 'very_stable',
                'fixed_routes': True,
                'regular_schedule': True
            }
        }
    
    def load_existing_data(self, data_path: str) -> pd.DataFrame:
        """기존 AIS 데이터 로드"""
        all_data = []
        
        for csv_file in Path(data_path).glob("*.csv"):
            df = pd.read_csv(csv_file)
            # 파일명에서 vessel_id 추출
            vessel_id = csv_file.stem
            df['vessel_id'] = vessel_id
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        
        return combined_df
    
    def generate_suspicious_labels(self, df: pd.DataFrame, suspicious_ratio: float = 0.3) -> pd.DataFrame:
        """의심 선박 라벨 생성"""
        vessel_ids = df['vessel_id'].unique()
        n_suspicious = max(1, int(len(vessel_ids) * suspicious_ratio))  # 최소 1개는 의심 선박
        
        print(f"전체 선박 수: {len(vessel_ids)}, 의심 선박 수: {n_suspicious}")
        
        # 랜덤하게 의심 선박 선택
        np.random.seed(42)  # 재현 가능한 결과를 위해
        suspicious_vessels = np.random.choice(vessel_ids, n_suspicious, replace=False)
        
        print(f"의심 선박 ID: {suspicious_vessels}")
        
        # 기본적으로 모든 선박을 정상으로 설정
        df['is_suspicious'] = 0
        
        # 의심 선박 라벨 설정
        for vessel_id in suspicious_vessels:
            vessel_mask = df['vessel_id'] == vessel_id
            df.loc[vessel_mask, 'is_suspicious'] = 1
            
            # 의심 선박에 대해 행동 패턴 수정
            pattern_type = np.random.choice(list(self.suspicious_patterns.keys()))
            print(f"선박 {vessel_id}에 {pattern_type} 패턴 적용")
            df = self._apply_suspicious_pattern(df, vessel_id, pattern_type)
        
        # 라벨 분포 확인
        print(f"의심 선박 비율: {df['is_suspicious'].mean():.2%}")
        print(f"의심 선박 레코드 수: {df['is_suspicious'].sum()}")
        
        return df
    
    def _apply_suspicious_pattern(self, df: pd.DataFrame, vessel_id: str, pattern_type: str) -> pd.DataFrame:
        """의심 선박 행동 패턴 적용"""
        vessel_mask = df['vessel_id'] == vessel_id
        pattern = self.suspicious_patterns[pattern_type]
        vessel_data = df[vessel_mask].copy()
        
        print(f"  - {pattern_type} 패턴 적용 중... (레코드 수: {vessel_mask.sum()})")
        
        if pattern_type == 'illegal_fishing':
            # 1. 속도 조정 (매우 느리게, 불규칙하게)
            slow_speeds = np.random.uniform(
                pattern['speed_range'][0], 
                pattern['speed_range'][1], 
                vessel_mask.sum()
            )
            # 일부 구간에서 갑자기 빨라지는 패턴 (도망가는 행동)
            escape_indices = np.random.choice(vessel_mask.sum(), size=int(vessel_mask.sum() * 0.1), replace=False)
            slow_speeds[escape_indices] = np.random.uniform(8, 15, len(escape_indices))
            df.loc[vessel_mask, 'sog'] = slow_speeds
            
            # 2. 코스 변화 증가 (지그재그 패턴)
            original_cog = df.loc[vessel_mask, 'cog'].values
            course_noise = np.random.normal(0, 45, vessel_mask.sum())  # 더 큰 변화
            df.loc[vessel_mask, 'cog'] = (original_cog + course_noise) % 360
            
            # 3. 야간 활동 증가
            night_mask = vessel_mask & (df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]))
            if night_mask.sum() > 0:
                df.loc[night_mask, 'sog'] *= 0.3  # 야간에 매우 느리게
            
            # 4. 특정 지역에서 머무름 (어업 활동 시뮬레이션)
            fishing_spots = np.random.choice(vessel_mask.sum(), size=int(vessel_mask.sum() * 0.3), replace=False)
            df.loc[vessel_mask, 'mean_ship_course_change'] = np.random.uniform(15, 45, vessel_mask.sum())
            
        elif pattern_type == 'smuggling':
            # 1. 빠른 속도로 이동
            fast_speeds = np.random.uniform(
                pattern['speed_range'][0], 
                pattern['speed_range'][1], 
                vessel_mask.sum()
            )
            df.loc[vessel_mask, 'sog'] = fast_speeds
            
            # 2. 직선 코스 (목적지 지향적)
            if 'mean_ship_course_change' in df.columns:
                df.loc[vessel_mask, 'mean_ship_course_change'] *= 0.2  # 코스 변화 최소화
            if 'standard_deviation_of_ship_course_change' in df.columns:
                df.loc[vessel_mask, 'standard_deviation_of_ship_course_change'] *= 0.3
            
            # 3. 특정 시간대에 활동 (새벽 시간)
            smuggling_hours = vessel_mask & (df['hour'].isin([2, 3, 4, 5]))
            if smuggling_hours.sum() > 0:
                df.loc[smuggling_hours, 'sog'] *= 1.3  # 새벽에 더 빠르게
            
            # 4. 북한 수역 근처로 이동
            nk_lat_center = 39.0
            nk_lon_center = 127.5
            df.loc[vessel_mask, 'lat'] = nk_lat_center + np.random.normal(0, 1.0, vessel_mask.sum())
            df.loc[vessel_mask, 'lon'] = nk_lon_center + np.random.normal(0, 1.0, vessel_mask.sum())
            
        elif pattern_type == 'surveillance':
            # 1. 중간 속도로 순찰
            patrol_speeds = np.random.uniform(
                pattern['speed_range'][0], 
                pattern['speed_range'][1], 
                vessel_mask.sum()
            )
            df.loc[vessel_mask, 'sog'] = patrol_speeds
            
            # 2. 순찰 패턴 (반복적인 코스 변화)
            if 'mean_ship_course_change' in df.columns:
                df.loc[vessel_mask, 'mean_ship_course_change'] *= 2.0  # 코스 변화 증가
            
            # 3. 경계선 근처 활동
            border_lat = 38.5  # 북한 경계선 근처
            border_lon = 127.0
            df.loc[vessel_mask, 'lat'] = border_lat + np.random.normal(0, 0.3, vessel_mask.sum())
            df.loc[vessel_mask, 'lon'] = border_lon + np.random.normal(0, 0.3, vessel_mask.sum())
            
            # 4. 장시간 같은 지역에서 활동
            if 'fishery_density' in df.columns:
                df.loc[vessel_mask, 'fishery_density'] *= 0.1  # 어업 밀도 낮은 지역
        
        print(f"  - 패턴 적용 완료: 평균 속도 {df.loc[vessel_mask, 'sog'].mean():.2f} knots")
        
        return df
    
    def generate_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """신뢰도 점수 생성"""
        # 기본 신뢰도는 랜덤
        df['confidence'] = np.random.uniform(0.1, 0.9, len(df))
        
        # 의심 선박의 경우 특정 조건에서 신뢰도 조정
        suspicious_mask = df['is_suspicious'] == 1
        
        # 야간 활동 시 신뢰도 증가
        night_mask = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])
        df.loc[suspicious_mask & night_mask, 'confidence'] += 0.2
        
        # 느린 속도 시 신뢰도 증가
        slow_speed_mask = df['sog'] < 2.0
        df.loc[suspicious_mask & slow_speed_mask, 'confidence'] += 0.15
        
        # 코스 변화가 클 때 신뢰도 증가
        high_course_change = df['mean_ship_course_change'] > 10
        df.loc[suspicious_mask & high_course_change, 'confidence'] += 0.1
        
        # 신뢰도 범위 조정 (0-1)
        df['confidence'] = np.clip(df['confidence'], 0.0, 1.0)
        
        # 정상 선박의 경우 낮은 신뢰도
        normal_mask = df['is_suspicious'] == 0
        df.loc[normal_mask, 'confidence'] = np.random.uniform(0.05, 0.3, normal_mask.sum())
        
        return df
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 피처 추가"""
        # 데이터프레임 복사
        df = df.copy()
        
        # datetime 기반 정렬
        df = df.sort_values(['vessel_id', 'datetime']).reset_index(drop=True)
        
        # 속도 변화율
        df['speed_change_rate'] = 0.0
        df['course_change_rate'] = 0.0
        df['distance_traveled'] = 0.0
        df['is_loitering'] = 0
        df['loitering_duration'] = 0.0
        df['is_night'] = 0
        df['night_activity_ratio'] = 0.0
        
        # 선박별로 처리
        for vessel_id in df['vessel_id'].unique():
            vessel_mask = df['vessel_id'] == vessel_id
            vessel_data = df[vessel_mask].copy()
            
            if len(vessel_data) > 1:
                # 속도 변화율
                speed_diff = vessel_data['sog'].diff().fillna(0)
                df.loc[vessel_mask, 'speed_change_rate'] = speed_diff.values
                
                # 코스 변화율
                course_diff = vessel_data['cog'].diff().fillna(0)
                # 360도 경계 처리
                course_diff[course_diff > 180] -= 360
                course_diff[course_diff < -180] += 360
                df.loc[vessel_mask, 'course_change_rate'] = course_diff.values
                
                # 거리 계산
                if len(vessel_data) > 1:
                    distances = self._calculate_vessel_distance(vessel_data)
                    df.loc[vessel_mask, 'distance_traveled'] = distances
            
            # 머무름 시간
            is_loitering = (vessel_data['sog'] < 1.0).astype(int)
            df.loc[vessel_mask, 'is_loitering'] = is_loitering.values
            
            # 롤링 윈도우 계산
            if len(vessel_data) >= 10:
                loitering_duration = is_loitering.rolling(window=10, min_periods=1).sum()
                df.loc[vessel_mask, 'loitering_duration'] = loitering_duration.values
            
            # 야간 활동
            is_night = vessel_data['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
            df.loc[vessel_mask, 'is_night'] = is_night.values
            
            if len(vessel_data) >= 20:
                night_ratio = is_night.rolling(window=20, min_periods=1).mean()
                df.loc[vessel_mask, 'night_activity_ratio'] = night_ratio.values
        
        # 북한 수역 근접도
        nk_waters = self.config['feature_engineering']['geographic_zones']['north_korea_waters']
        df['nk_proximity'] = self._calculate_nk_proximity(df, nk_waters)
        
        # 어업 구역 근접도
        fishing_zones = self.config['feature_engineering']['geographic_zones']['fishing_zones']
        df['fishing_zone_proximity'] = self._calculate_fishing_zone_proximity(df, fishing_zones)
        
        return df
    
    def _calculate_vessel_distance(self, vessel_data: pd.DataFrame) -> np.ndarray:
        """단일 선박의 위치 간 거리 계산 (Haversine formula)"""
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # 지구 반지름 (km)
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        if len(vessel_data) <= 1:
            return np.array([0.0])
        
        # 시간순 정렬
        vessel_data = vessel_data.sort_values('datetime')
        
        # 거리 계산
        distances = haversine(
            vessel_data['lat'].iloc[:-1].values,
            vessel_data['lon'].iloc[:-1].values,
            vessel_data['lat'].iloc[1:].values,
            vessel_data['lon'].iloc[1:].values
        )
        
        # 첫 번째 포인트는 0
        return np.concatenate([[0], distances])
    
    def _calculate_distance(self, df: pd.DataFrame) -> pd.Series:
        """위치 간 거리 계산 (Haversine formula) - 레거시 함수"""
        distances = []
        for vessel_id in df['vessel_id'].unique():
            vessel_data = df[df['vessel_id'] == vessel_id].sort_values('datetime')
            vessel_distances = self._calculate_vessel_distance(vessel_data)
            distances.extend(vessel_distances)
        
        return pd.Series(distances, index=df.index)
    
    def _calculate_nk_proximity(self, df: pd.DataFrame, nk_waters: Dict) -> pd.Series:
        """북한 수역 근접도 계산"""
        # 북한 수역 중심점
        center_lat = (nk_waters['lat_min'] + nk_waters['lat_max']) / 2
        center_lon = (nk_waters['lon_min'] + nk_waters['lon_max']) / 2
        
        # 거리 계산
        distances = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
        
        # 근접도로 변환 (거리가 가까울수록 높은 값)
        proximity = 1 / (1 + distances)
        
        return proximity
    
    def _calculate_fishing_zone_proximity(self, df: pd.DataFrame, fishing_zones: List[Dict]) -> pd.Series:
        """어업 구역 근접도 계산"""
        max_proximity = 0
        
        for zone in fishing_zones:
            center_lat = (zone['lat_min'] + zone['lat_max']) / 2
            center_lon = (zone['lon_min'] + zone['lon_max']) / 2
            
            distances = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
            proximity = 1 / (1 + distances)
            
            max_proximity = np.maximum(max_proximity, proximity)
        
        return pd.Series(max_proximity, index=df.index)
    
    def split_train_test_data(self, df: pd.DataFrame, train_vessels: List[str], test_vessels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """훈련/테스트 데이터 분할"""
        train_df = df[df['vessel_id'].isin(train_vessels)].copy()
        test_df = df[df['vessel_id'].isin(test_vessels)].copy()
        
        return train_df, test_df
    
    def save_competition_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
        """대회 형식으로 데이터 저장"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 훈련 데이터 저장 (라벨 포함)
        train_df.to_csv(f"{output_dir}/train_data.csv", index=False)
        
        # 테스트 데이터 저장 (라벨 제외)
        test_features = test_df.drop(['is_suspicious', 'confidence'], axis=1)
        test_features.to_csv(f"{output_dir}/test_data.csv", index=False)
        
        # 테스트 라벨 저장 (평가용)
        test_labels = test_df[['vessel_id', 'is_suspicious', 'confidence']].drop_duplicates()
        test_labels.to_csv(f"{output_dir}/test_labels.csv", index=False)
        
        # 제출 템플릿 생성
        submission_template = test_df[['vessel_id']].drop_duplicates()
        submission_template['is_suspicious'] = 0
        submission_template['confidence'] = 0.0
        submission_template.to_csv(f"{output_dir}/submission_template.csv", index=False)
        
        print(f"데이터가 {output_dir}에 저장되었습니다.")
        print(f"훈련 데이터: {len(train_df)} 레코드")
        print(f"테스트 데이터: {len(test_features)} 레코드")
        print(f"의심 선박 비율 (훈련): {train_df['is_suspicious'].mean():.2%}")
        print(f"의심 선박 비율 (테스트): {test_df['is_suspicious'].mean():.2%}")

def main():
    """메인 실행 함수"""
    generator = SyntheticVesselDataGenerator()
    
    # 기존 데이터 로드
    data_path = "train_data/original_data"
    df = generator.load_existing_data(data_path)
    
    print(f"로드된 데이터: {len(df)} 레코드, {df['vessel_id'].nunique()} 선박")
    
    # 의심 선박 라벨 생성
    df = generator.generate_suspicious_labels(df, suspicious_ratio=0.3)
    
    # 신뢰도 점수 생성
    df = generator.generate_confidence_scores(df)
    
    # 고급 피처 추가
    df = generator.add_advanced_features(df)
    
    # 선박 ID 목록
    vessel_ids = df['vessel_id'].unique()
    
    # 첫 3개를 훈련용, 나머지를 테스트용으로 분할
    train_vessels = vessel_ids[:3]
    test_vessels = vessel_ids[3:]
    
    print(f"훈련 선박: {train_vessels}")
    print(f"테스트 선박: {test_vessels}")
    
    # 데이터 분할
    train_df, test_df = generator.split_train_test_data(df, train_vessels, test_vessels)
    
    # 대회 형식으로 저장
    generator.save_competition_data(train_df, test_df, "competition_data")

if __name__ == "__main__":
    main() 