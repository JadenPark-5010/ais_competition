# 북한 의심 선박 탐지 시스템 (North Korean Suspicious Vessel Detection)

AIS 데이터를 활용하여 북한 의심 선박을 탐지하는 머신러닝 시스템입니다. 이 프로젝트는 해양 안보와 불법 활동 탐지를 위한 고급 분석 도구를 제공합니다.

## 🎯 프로젝트 개요

### 목표
- AIS(Automatic Identification System) 데이터를 분석하여 북한 의심 선박 탐지
- 시계열 분석과 지리적 특성을 활용한 고급 피처 엔지니어링
- 앙상블 머신러닝 모델을 통한 높은 정확도의 분류 및 신뢰도 예측

### 주요 특징
- **End-to-End 파이프라인**: 데이터 생성부터 모델 배포까지 완전 자동화
- **고급 피처 엔지니어링**: 100+ 개의 도메인 특화 피처
- **앙상블 모델**: XGBoost, LightGBM, CatBoost, Random Forest, Neural Network 조합
- **실시간 시각화**: 궤적 분석, 성능 지표, 예측 결과 대시보드
- **모델 해석**: SHAP을 활용한 예측 설명

## 🏗️ 시스템 아키텍처

```
maritime_anomaly_detection/
├── config/                     # 설정 파일
│   └── competition_config.yaml
├── src/                        # 소스 코드
│   ├── data_generation/        # 데이터 생성
│   ├── features/              # 피처 엔지니어링
│   ├── models/                # 머신러닝 모델
│   └── visualization/         # 시각화
├── train_data/                # 훈련 데이터
├── competition_data/          # 대회 데이터
├── models/                    # 저장된 모델
├── results/                   # 결과 파일
├── visualizations/           # 시각화 결과
├── main.py                   # 메인 실행 스크립트
└── requirements.txt          # 패키지 의존성
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd maritime_anomaly_detection

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행

```bash
# 메인 스크립트 실행 (End-to-End)
python main.py
```

이 명령어는 다음 과정을 자동으로 수행합니다:
1. 데이터 생성 및 전처리
2. 피처 엔지니어링
3. 모델 훈련
4. 성능 평가
5. 예측 및 제출 파일 생성
6. 시각화 및 보고서 생성

### 3. 결과 확인

실행 완료 후 다음 파일들이 생성됩니다:
- `results/submission.csv`: 대회 제출 파일
- `results/technical_report.md`: 기술 보고서
- `visualizations/`: 시각화 결과
- `models/`: 훈련된 모델

## 📊 데이터 구조

### 입력 데이터
```
vessel_id, datetime, lat, lon, sog, cog, [환경 데이터...]
```

### 출력 데이터
```
vessel_id, is_suspicious, confidence
```

## 🔧 주요 구성 요소

### 1. 데이터 생성기 (`SyntheticVesselDataGenerator`)
- 기존 AIS 데이터 로드 및 전처리
- 의심 선박 라벨 생성 (3가지 패턴: 불법어업, 밀수, 감시)
- 신뢰도 점수 생성

### 2. 피처 엔지니어링 (`MaritimeFeatureEngineer`)
- **시계열 피처**: 시간 패턴, 주기성 분석
- **궤적 피처**: 이동 패턴, 방향 변화, 복잡도
- **행동 피처**: 속도 패턴, 머무름 행동, 급격한 변화
- **지리적 피처**: 북한 수역 근접도, 어업 구역 분석
- **이상 탐지 피처**: 비정상 행동 패턴
- **통계적 피처**: 선박별 집계 통계
- **클러스터링 피처**: 행동 패턴 클러스터링

### 3. 앙상블 모델 (`SuspiciousVesselEnsemble`)
- **개별 모델**: XGBoost, LightGBM, CatBoost, Random Forest, MLP
- **앙상블 전략**: Soft Voting with Calibration
- **클래스 불균형 처리**: SMOTE, ADASYN
- **하이퍼파라미터 튜닝**: Grid Search CV

### 4. 시각화 (`CompetitionVisualizer`)
- 데이터 분포 분석
- 모델 성능 시각화
- 선박 궤적 분석
- 예측 결과 대시보드

## 📈 성능 지표

모델은 다음 지표로 평가됩니다:
- **Accuracy**: 전체 정확도
- **Precision**: 의심 선박 예측 정밀도
- **Recall**: 의심 선박 탐지율
- **F1-Score**: 정밀도와 재현율의 조화평균
- **ROC-AUC**: ROC 곡선 아래 면적
- **Average Precision**: PR 곡선 아래 면적

## 🎛️ 설정 옵션

`config/competition_config.yaml`에서 다음을 설정할 수 있습니다:

```yaml
# 모델 설정
models:
  ensemble:
    - name: "xgboost"
      type: "XGBClassifier"
      params:
        n_estimators: 1000
        max_depth: 8

# 피처 엔지니어링 설정
feature_engineering:
  time_windows: [10, 30, 60, 120]
  anomaly_thresholds:
    speed_change_rate: 5.0
    course_change_rate: 30.0

# 훈련 설정
training:
  cv_folds: 5
  class_balance:
    method: "SMOTE"
```

## 🔍 의심 선박 탐지 패턴

시스템은 다음과 같은 의심 행동 패턴을 탐지합니다:

### 1. 불법 어업 (Illegal Fishing)
- 매우 느린 속도 (0.5-3.0 knots)
- 빈번한 코스 변화
- 장시간 머무름 (3시간 이상)
- 야간 활동 증가

### 2. 밀수 (Smuggling)
- 빠른 속도 (8.0-25.0 knots)
- 직선 코스
- 다른 선박과의 만남
- AIS 신호 끊김

### 3. 감시 (Surveillance)
- 중간 속도 (5.0-15.0 knots)
- 순찰 패턴
- 경계선 근처 활동
- 장시간 지속

## 📋 사용 예시

### 개별 모듈 사용

```python
# 피처 엔지니어링만 수행
from src.features.feature_engineering import MaritimeFeatureEngineer

engineer = MaritimeFeatureEngineer(config)
features_df = engineer.create_all_features(raw_data)

# 모델 훈련만 수행
from src.models.ensemble_model import SuspiciousVesselEnsemble

ensemble = SuspiciousVesselEnsemble(config)
ensemble.fit(train_data)
predictions = ensemble.predict(test_data)
```

### 시각화 생성

```python
from src.visualization.analysis_plots import CompetitionVisualizer

visualizer = CompetitionVisualizer(config)
visualizer.plot_vessel_trajectory(vessel_data, "trajectory.png")
visualizer.create_dashboard(train_df, test_df, results, "dashboard.png")
```

## 🛠️ 개발 및 기여

### 코드 스타일
- PEP 8 준수
- Type hints 사용
- Docstring 작성

### 테스트 실행
```bash
pytest tests/
```

### 새로운 피처 추가
1. `src/features/feature_engineering.py`에 새 메서드 추가
2. `config/competition_config.yaml`에 설정 추가
3. 테스트 코드 작성

## 📚 기술 문서

자세한 기술 문서는 실행 후 `results/technical_report.md`에서 확인할 수 있습니다.

## 🔒 보안 및 윤리

이 시스템은 해양 안보 목적으로 개발되었으며, 다음 원칙을 준수합니다:
- 개인정보 보호
- 국제법 준수
- 투명한 알고리즘
- 편향 방지

## 📞 지원 및 문의

- 이슈 리포트: GitHub Issues
- 기술 문의: [이메일 주소]
- 문서: [문서 링크]

## 📄 라이선스

이 프로젝트는 [라이선스 유형] 하에 배포됩니다.

---

**주의**: 이 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 운영 환경에서 사용하기 전에 충분한 검증이 필요합니다.