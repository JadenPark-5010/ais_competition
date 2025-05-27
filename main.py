"""
북한 의심 선박 탐지 대회 - 메인 실행 스크립트
End-to-End 파이프라인: 데이터 생성 → 피처 엔지니어링 → 모델 훈련 → 예측 → 제출
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
from src.data_generation.synthetic_data_generator import SyntheticVesselDataGenerator
from src.features.feature_engineering import MaritimeFeatureEngineer
from src.models.ensemble_model import SuspiciousVesselEnsemble
from src.visualization.analysis_plots import CompetitionVisualizer

def load_config(config_path: str = "config/competition_config.yaml") -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_directories(config: dict):
    """필요한 디렉토리 생성"""
    directories = [
        config['data']['output_path'],
        'competition_data',
        'models',
        'results',
        'visualizations'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("디렉토리 설정 완료")

def generate_competition_data(config: dict):
    """기존 데이터를 사용하여 대회 데이터 생성"""
    print("\n=== 1. 기존 데이터 로드 및 처리 ===")
    
    # 기존 데이터 디렉토리 확인
    train_data_dir = "train_data/original_data"
    if not os.path.exists(train_data_dir):
        print(f"기존 데이터 디렉토리를 찾을 수 없습니다: {train_data_dir}")
        print("샘플 데이터를 생성합니다...")
        df = create_sample_data()
    else:
        print(f"기존 데이터 디렉토리: {train_data_dir}")
        
        # 기존 데이터 파일 확인
        data_files = [f for f in os.listdir(train_data_dir) if f.endswith('.csv')]
        if not data_files:
            print(f"CSV 파일을 찾을 수 없습니다: {train_data_dir}")
            print("샘플 데이터를 생성합니다...")
            df = create_sample_data()
        else:
            print(f"발견된 데이터 파일: {len(data_files)}개")
            
            # 모든 데이터 파일 로드 및 결합
            all_data = []
            vessel_id = 1
            
            for file_name in data_files:
                file_path = os.path.join(train_data_dir, file_name)
                print(f"로딩 중: {file_name}")
                
                try:
                    file_df = pd.read_csv(file_path)
                    print(f"  - 데이터 크기: {file_df.shape}")
                    
                    # vessel_id 추가 (파일명에서 추출하거나 순차 생성)
                    file_df['vessel_id'] = f"vessel_{vessel_id:03d}"
                    
                    # 기본 컬럼 확인
                    required_columns = ['datetime', 'lat', 'lon', 'sog', 'cog']
                    missing_columns = [col for col in required_columns if col not in file_df.columns]
                    if missing_columns:
                        print(f"  - 경고: 누락된 컬럼 {missing_columns}")
                        continue
                    
                    # 데이터 타입 변환
                    file_df['datetime'] = pd.to_datetime(file_df['datetime'])
                    file_df['lat'] = pd.to_numeric(file_df['lat'], errors='coerce')
                    file_df['lon'] = pd.to_numeric(file_df['lon'], errors='coerce')
                    file_df['sog'] = pd.to_numeric(file_df['sog'], errors='coerce')
                    file_df['cog'] = pd.to_numeric(file_df['cog'], errors='coerce')
                    
                    # 결측값 제거
                    file_df = file_df.dropna(subset=['lat', 'lon', 'sog', 'cog'])
                    
                    if len(file_df) > 0:
                        all_data.append(file_df)
                        vessel_id += 1
                    
                except Exception as e:
                    print(f"  - 오류: {file_name} 로드 실패: {e}")
                    continue
            
            if not all_data:
                print("유효한 데이터 파일이 없습니다. 샘플 데이터를 생성합니다...")
                df = create_sample_data()
            else:
                # 모든 데이터 결합
                df = pd.concat(all_data, ignore_index=True)
                print(f"\n결합된 데이터: {df.shape}")
                print(f"선박 수: {df['vessel_id'].nunique()}")
    
    # SyntheticVesselDataGenerator를 사용하여 라벨 및 피처 생성
    generator = SyntheticVesselDataGenerator(config_path="config/competition_config.yaml")
    
    # 의심 선박 라벨 생성
    df = generator.generate_suspicious_labels(df, suspicious_ratio=0.3)
    
    # 신뢰도 점수 생성
    df = generator.generate_confidence_scores(df)
    
    # 고급 피처 추가
    df = generator.add_advanced_features(df)
    
    # 선박 ID 목록
    vessel_ids = df['vessel_id'].unique()
    print(f"전체 선박 수: {len(vessel_ids)}")
    
    # 의심 선박과 정상 선박이 훈련/테스트 모두에 포함되도록 분할
    suspicious_vessels = df[df['is_suspicious'] == 1]['vessel_id'].unique()
    normal_vessels = df[df['is_suspicious'] == 0]['vessel_id'].unique()
    
    print(f"의심 선박: {suspicious_vessels}")
    print(f"정상 선박: {normal_vessels}")
    
    # 각 그룹에서 일부를 훈련용, 일부를 테스트용으로 분할
    if len(suspicious_vessels) >= 2:
        train_suspicious = suspicious_vessels[:len(suspicious_vessels)//2 + 1]
        test_suspicious = suspicious_vessels[len(suspicious_vessels)//2 + 1:]
    else:
        train_suspicious = suspicious_vessels[:1] if len(suspicious_vessels) > 0 else []
        test_suspicious = suspicious_vessels[1:] if len(suspicious_vessels) > 1 else suspicious_vessels[:1] if len(suspicious_vessels) > 0 else []
    
    if len(normal_vessels) >= 2:
        train_normal = normal_vessels[:len(normal_vessels)//2 + 1]
        test_normal = normal_vessels[len(normal_vessels)//2 + 1:]
    else:
        train_normal = normal_vessels[:1] if len(normal_vessels) > 0 else []
        test_normal = normal_vessels[1:] if len(normal_vessels) > 1 else normal_vessels[:1] if len(normal_vessels) > 0 else []
    
    # 훈련/테스트 선박 목록 결합
    train_vessels = np.concatenate([train_suspicious, train_normal]) if len(train_suspicious) > 0 and len(train_normal) > 0 else (train_suspicious if len(train_suspicious) > 0 else train_normal)
    test_vessels = np.concatenate([test_suspicious, test_normal]) if len(test_suspicious) > 0 and len(test_normal) > 0 else (test_suspicious if len(test_suspicious) > 0 else test_normal)
    
    # 최소 1개씩은 보장
    if len(train_vessels) == 0:
        train_vessels = vessel_ids[:max(1, len(vessel_ids)//2)]
    if len(test_vessels) == 0:
        test_vessels = vessel_ids[len(vessel_ids)//2:]
    
    print(f"훈련 선박: {train_vessels}")
    print(f"테스트 선박: {test_vessels}")
    
    # 데이터 분할
    train_df, test_df = generator.split_train_test_data(df, train_vessels, test_vessels)
    
    # 대회 형식으로 저장
    generator.save_competition_data(train_df, test_df, "competition_data")
    
    return train_df, test_df

def create_sample_data():
    """샘플 데이터 생성 (기존 데이터가 없는 경우)"""
    print("샘플 AIS 데이터 생성 중...")
    
    # 5개 선박의 샘플 데이터 생성
    sample_data = []
    
    for vessel_idx in range(5):
        vessel_id = f"SAMPLE_VESSEL_{vessel_idx:03d}"
        
        # 각 선박당 100개 포인트
        n_points = 100
        
        # 기본 위치 (한반도 주변)
        base_lat = 35.0 + np.random.uniform(-2, 5)
        base_lon = 126.0 + np.random.uniform(-3, 6)
        
        for i in range(n_points):
            # 시간 생성 (1시간 간격)
            timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i)
            
            # 위치 생성 (랜덤 워크)
            lat = base_lat + np.random.normal(0, 0.01) * i * 0.1
            lon = base_lon + np.random.normal(0, 0.01) * i * 0.1
            
            # 속도 생성
            sog = max(0, np.random.normal(10, 3))
            
            # 코스 생성
            cog = np.random.uniform(0, 360)
            
            # 환경 데이터 (랜덤)
            sample_data.append({
                'vessel_id': vessel_id,
                'datetime': timestamp,
                'lat': lat,
                'lon': lon,
                'sog': sog,
                'cog': cog,
                'sea_surface_temperature': np.random.uniform(10, 25),
                'sea_surface_salinity': np.random.uniform(30, 35),
                'current_speed': np.random.uniform(0, 2),
                'wind': np.random.uniform(0, 15),
                'tide': np.random.uniform(-2, 2),
                'bottom_depth': np.random.uniform(10, 200),
                'chlorophyll': np.random.uniform(0.1, 5),
                'DIN': np.random.uniform(0, 20),
                'DIP': np.random.uniform(0, 2),
                'dissolved_oxygen': np.random.uniform(5, 10),
                'fishery_density': np.random.uniform(0, 100),
                'month': timestamp.month,
                'hour': timestamp.hour,
                'mean_ship_course_change': np.random.uniform(0, 30),
                'standard_deviation_of_ship_course_change': np.random.uniform(0, 15),
                'histogram_of_ship_course_change': np.random.uniform(0, 1),
                'mean_ship_course_change_per_velocity_stage': np.random.uniform(0, 20),
                'mean_velocity_change': np.random.uniform(-5, 5),
                'standard_deviation_of_velocity_change': np.random.uniform(0, 10),
                'mean_velocity': sog,
                'histogram_of_velocity': np.random.uniform(0, 1),
                'histogram_of_velocity_change': np.random.uniform(0, 1),
                'velocity_change_per_velocity_stage': np.random.uniform(0, 10)
            })
    
    return pd.DataFrame(sample_data)

def perform_feature_engineering(train_df: pd.DataFrame, test_df: pd.DataFrame, config: dict):
    """피처 엔지니어링 수행"""
    print("\n=== 2. 피처 엔지니어링 ===")
    
    feature_engineer = MaritimeFeatureEngineer(config)
    
    # 훈련 데이터 피처 생성
    print("훈련 데이터 피처 엔지니어링...")
    train_features = feature_engineer.create_all_features(train_df)
    
    # 테스트 데이터 피처 생성
    print("테스트 데이터 피처 엔지니어링...")
    test_features = feature_engineer.create_all_features(test_df)
    
    # 피처 선택
    print("피처 선택 수행...")
    train_selected, selected_features = feature_engineer.select_features(train_features)
    
    # 테스트 데이터에 동일한 피처 적용
    test_selected = test_features[['vessel_id'] + selected_features].copy()
    if 'is_suspicious' in test_features.columns:
        test_selected['is_suspicious'] = test_features['is_suspicious']
    if 'confidence' in test_features.columns:
        test_selected['confidence'] = test_features['confidence']
    
    # 피처 중요도 분석
    importance_analysis = feature_engineer.get_feature_importance_analysis(
        train_selected, selected_features
    )
    
    print(f"선택된 피처 수: {len(selected_features)}")
    print("상위 10개 중요 피처 (Random Forest):")
    rf_importance = importance_analysis['random_forest']
    top_features = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    return train_selected, test_selected, selected_features, importance_analysis

def train_ensemble_model(train_df: pd.DataFrame, config: dict):
    """앙상블 모델 훈련"""
    print("\n=== 3. 앙상블 모델 훈련 ===")
    
    # 앙상블 모델 생성
    ensemble = SuspiciousVesselEnsemble(config)
    
    # 모델 훈련
    training_results = ensemble.fit(train_df)
    
    print("\n개별 모델 성능:")
    for model_name, scores in training_results['model_scores'].items():
        print(f"  {model_name}: CV AUC = {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})")
    
    # 모델 저장
    model_path = "models/suspicious_vessel_ensemble.pkl"
    ensemble.save_model(model_path)
    print(f"모델 저장: {model_path}")
    
    return ensemble, training_results

def evaluate_model(ensemble: SuspiciousVesselEnsemble, test_df: pd.DataFrame):
    """모델 평가"""
    print("\n=== 4. 모델 평가 ===")
    
    # 평가 수행
    evaluation_results = ensemble.evaluate(test_df)
    
    print("평가 결과:")
    metrics = evaluation_results['metrics']
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print("\n혼동 행렬:")
    print(evaluation_results['confusion_matrix'])
    
    # 피처 중요도
    feature_importance = ensemble.get_feature_importance()
    print("\n상위 10개 중요 피처:")
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in top_features:
        print(f"  {feature}: {importance:.4f}")
    
    return evaluation_results

def generate_predictions_and_submission(ensemble: SuspiciousVesselEnsemble, test_df: pd.DataFrame):
    """예측 및 제출 파일 생성"""
    print("\n=== 5. 예측 및 제출 파일 생성 ===")
    
    # 제출 파일 생성
    submission_path = "results/submission.csv"
    submission_df = ensemble.generate_submission(test_df, submission_path)
    
    # 예측 결과 분석
    print("\n예측 결과 분석:")
    print(f"총 선박 수: {len(submission_df)}")
    print(f"의심 선박 예측 수: {submission_df['is_suspicious'].sum()}")
    print(f"의심 선박 비율: {submission_df['is_suspicious'].mean():.2%}")
    print(f"평균 신뢰도: {submission_df['confidence'].mean():.4f}")
    print(f"신뢰도 표준편차: {submission_df['confidence'].std():.4f}")
    
    return submission_df

def create_visualizations(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                         evaluation_results: dict, submission_df: pd.DataFrame, config: dict):
    """시각화 생성"""
    print("\n=== 6. 시각화 생성 ===")
    
    try:
        visualizer = CompetitionVisualizer(config)
        
        # 데이터 분포 시각화
        visualizer.plot_data_distribution(train_df, "visualizations/data_distribution.png")
        
        # 모델 성능 시각화
        visualizer.plot_model_performance(evaluation_results, "visualizations/model_performance.png")
        
        # 예측 결과 시각화
        visualizer.plot_prediction_results(submission_df, "visualizations/prediction_results.png")
        
        # 궤적 시각화 (샘플)
        sample_vessels = train_df['vessel_id'].unique()[:3]
        for vessel_id in sample_vessels:
            vessel_data = train_df[train_df['vessel_id'] == vessel_id]
            visualizer.plot_vessel_trajectory(
                vessel_data, 
                f"visualizations/trajectory_{vessel_id}.png"
            )
        
        print("시각화 파일이 visualizations/ 폴더에 저장되었습니다.")
        
    except Exception as e:
        print(f"시각화 생성 중 오류 발생: {e}")

def generate_technical_report(config: dict, training_results: dict, evaluation_results: dict, 
                            submission_df: pd.DataFrame, selected_features: list):
    """기술 문서 생성"""
    print("\n=== 7. 기술 문서 생성 ===")
    
    report_path = "results/technical_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 북한 의심 선박 탐지 시스템 기술 보고서\n\n")
        f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 1. 프로젝트 개요\n")
        f.write("AIS 데이터를 활용하여 북한 의심 선박을 탐지하는 머신러닝 시스템\n\n")
        
        f.write("## 2. 데이터 분석\n")
        f.write(f"- 훈련 데이터: {training_results['training_samples']} 샘플\n")
        f.write(f"- 클래스 분포: {training_results['class_distribution']}\n")
        f.write(f"- 피처 수: {len(selected_features)}\n\n")
        
        f.write("## 3. 모델 성능\n")
        f.write("### 개별 모델 성능 (Cross Validation)\n")
        for model_name, scores in training_results['model_scores'].items():
            f.write(f"- {model_name}: {scores['cv_mean']:.4f} (+/- {scores['cv_std']:.4f})\n")
        
        f.write("\n### 최종 모델 성능 (Test Set)\n")
        metrics = evaluation_results['metrics']
        for metric_name, value in metrics.items():
            f.write(f"- {metric_name}: {value:.4f}\n")
        
        f.write("\n## 4. 예측 결과\n")
        f.write(f"- 총 예측 선박 수: {len(submission_df)}\n")
        f.write(f"- 의심 선박 예측 수: {submission_df['is_suspicious'].sum()}\n")
        f.write(f"- 의심 선박 비율: {submission_df['is_suspicious'].mean():.2%}\n")
        f.write(f"- 평균 신뢰도: {submission_df['confidence'].mean():.4f}\n\n")
        
        f.write("## 5. 주요 피처\n")
        f.write("### 선택된 피처 목록\n")
        for i, feature in enumerate(selected_features[:20], 1):
            f.write(f"{i}. {feature}\n")
        
        f.write("\n## 6. 모델 구성\n")
        f.write("### 앙상블 모델\n")
        for model_config in config['models']['ensemble']:
            f.write(f"- {model_config['name']}: {model_config['type']}\n")
        
        f.write("\n## 7. 피처 엔지니어링\n")
        f.write("- 시계열 피처: 시간 패턴, 주기성 분석\n")
        f.write("- 궤적 피처: 이동 패턴, 방향 변화\n")
        f.write("- 행동 피처: 속도 패턴, 머무름 행동\n")
        f.write("- 지리적 피처: 위치 기반 특성\n")
        f.write("- 이상 탐지 피처: 비정상 행동 패턴\n\n")
        
        f.write("## 8. 결론\n")
        f.write("본 시스템은 다양한 머신러닝 기법을 조합하여 북한 의심 선박을 효과적으로 탐지할 수 있는 ")
        f.write("앙상블 모델을 구축하였습니다. 특히 시계열 분석과 지리적 특성을 활용한 피처 엔지니어링을 ")
        f.write("통해 높은 성능을 달성하였습니다.\n")
    
    print(f"기술 보고서가 {report_path}에 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("북한 의심 선박 탐지 대회 - End-to-End 파이프라인")
    print("=" * 60)
    
    # 설정 로드
    config = load_config()
    
    # 디렉토리 설정
    setup_directories(config)
    
    try:
        # 1. 데이터 생성
        train_df, test_df = generate_competition_data(config)
        
        # 2. 피처 엔지니어링
        train_features, test_features, selected_features, importance_analysis = perform_feature_engineering(
            train_df, test_df, config
        )
        
        # 3. 모델 훈련
        ensemble, training_results = train_ensemble_model(train_features, config)
        
        # 4. 모델 평가
        evaluation_results = evaluate_model(ensemble, test_features)
        
        # 5. 예측 및 제출
        submission_df = generate_predictions_and_submission(ensemble, test_features)
        
        # 6. 시각화
        create_visualizations(train_df, test_df, evaluation_results, submission_df, config)
        
        # 7. 기술 문서 생성
        generate_technical_report(config, training_results, evaluation_results, 
                                submission_df, selected_features)
        
        print("\n" + "=" * 60)
        print("파이프라인 실행 완료!")
        print("=" * 60)
        print("생성된 파일:")
        print("- competition_data/: 대회 데이터")
        print("- models/: 훈련된 모델")
        print("- results/: 제출 파일 및 보고서")
        print("- visualizations/: 시각화 결과")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 