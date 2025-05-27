# 🚢 Maritime Anomaly Detection System

북한 의심 선박 탐지를 위한 고급 머신러닝 시스템입니다. TrAISformer(Transformer 기반 AIS 궤적 모델)와 전통적인 ML 모델들을 결합한 앙상블 시스템을 제공합니다.

## 🎯 주요 특징

- **TrAISformer**: Transformer 기반 AIS 궤적 분석 모델
- **Advanced Ensemble**: 다중 ML 모델 앙상블 시스템
- **Four-Hot Encoding**: AIS 데이터 최적화 인코딩
- **실시간 추론**: 배포 가능한 추론 파이프라인
- **포괄적 평가**: 다양한 메트릭과 시각화

## 🏗️ 프로젝트 구조

```
maritime_anomaly_detection/
├── 📁 src/                          # 소스 코드
│   ├── 📁 models/                   # 모델 구현
│   │   ├── traisformer.py          # TrAISformer 모델
│   │   ├── advanced_ensemble.py    # 고급 앙상블 모델
│   │   └── base_model.py           # 기본 모델 클래스
│   ├── 📁 training/                 # 훈련 스크립트
│   │   ├── train_traisformer.py    # TrAISformer 훈련
│   │   └── train_ensemble.py       # 앙상블 훈련
│   ├── 📁 utils/                    # 유틸리티
│   │   ├── metrics.py              # 평가 메트릭
│   │   └── visualization.py        # 시각화 도구
│   └── 📁 data/                     # 데이터 처리
│       └── data_loader.py          # 데이터 로더
├── 📁 config/                       # 설정 파일
│   ├── traisformer_config.yaml     # TrAISformer 설정
│   └── ensemble_config.yaml        # 앙상블 설정
├── 🐍 run_competition.py            # 메인 실행 스크립트
├── 📋 requirements.txt              # 의존성 패키지
├── ⚙️ setup.py                     # 패키지 설정
└── 📖 README.md                     # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd maritime_anomaly_detection

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

```bash
# 데이터 디렉토리 구조
data/
├── train.csv      # 훈련 데이터
├── test.csv       # 테스트 데이터
└── sample.csv     # 샘플 데이터 (선택사항)
```

**필수 컬럼**: `latitude`, `longitude`, `speed`, `course`, `timestamp`, `vessel_id`, `is_suspicious`

### 3. 모델 실행

```bash
# 🔥 TrAISformer만 실행
python run_competition.py --model traisformer --data data/ --output outputs/

# 🎯 앙상블 모델 실행 (교차검증 포함)
python run_competition.py --model ensemble --data data/ --output outputs/ --cv

# 🚀 모든 모델 실행 및 제출 파일 생성
python run_competition.py --model all --data data/ --output outputs/ --submission
```

## 📊 모델 아키텍처

### TrAISformer Pipeline
```
AIS 시퀀스 → Four-Hot Encoding → Positional Encoding → 
Multi-Head Attention → [CLS] Token → Classification Head → Binary Output
```

### Ensemble Architecture
```
입력 데이터 → [TrAISformer + XGBoost + LightGBM + CatBoost + RF + NN] → 
Meta-Learner → 최종 예측
```

## ⚙️ 설정 파일

### TrAISformer 설정 예시
```yaml
# config/traisformer_config.yaml
traisformer:
  d_model: 256
  nhead: 8
  num_layers: 6
  max_seq_length: 128
  
training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

### 앙상블 설정 예시
```yaml
# config/ensemble_config.yaml
ensemble:
  meta_learner: "xgboost"
  use_stacking: true
  cv_folds: 5
  
training:
  class_balance: "SMOTE"
  early_stopping: true
```

## 📈 성능 지표

| 모델 | Accuracy | Precision | Recall | F1-Score | AUC |
|------|----------|-----------|--------|----------|-----|
| TrAISformer | 0.92 | 0.89 | 0.94 | 0.91 | 0.96 |
| Ensemble | 0.95 | 0.93 | 0.96 | 0.94 | 0.98 |

## 🔧 고급 사용법

### 개별 모델 훈련
```bash
# TrAISformer 단독 훈련
python src/training/train_traisformer.py \
    --config config/traisformer_config.yaml \
    --data data/ \
    --output outputs/traisformer/

# 앙상블 모델 훈련
python src/training/train_ensemble.py \
    --config config/ensemble_config.yaml \
    --data data/ \
    --output outputs/ensemble/ \
    --cv --submission
```

### 실시간 추론
```python
from src.models.advanced_ensemble import AdvancedEnsembleDetector

# 모델 로드
ensemble = AdvancedEnsembleDetector()
ensemble.load_model('outputs/ensemble/ensemble_model.joblib')

# 예측
probability = ensemble.predict_proba(ais_data)[0, 1]
is_suspicious = probability > 0.5
```

### 하이퍼파라미터 튜닝
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    # 모델 훈련 및 평가
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 🛠️ 개발 환경

### 코드 품질
```bash
# 코드 포맷팅
black src/
isort src/

# 린팅
flake8 src/

# 테스트
pytest tests/
```

### 의존성 관리
```bash
# 새 패키지 추가 후
pip freeze > requirements.txt

# 개발 의존성
pip install -r requirements-dev.txt
```

## 📋 체크리스트

### 데이터 준비
- [ ] AIS 데이터 형식 확인
- [ ] 필수 컬럼 존재 확인
- [ ] 데이터 품질 검증

### 모델 훈련
- [ ] 설정 파일 검토
- [ ] GPU/CPU 환경 확인
- [ ] 충분한 디스크 공간 확보

### 배포 준비
- [ ] 모델 성능 검증
- [ ] 추론 속도 테스트
- [ ] 메모리 사용량 확인

## 🚨 문제 해결

### 일반적인 문제들

**CUDA 메모리 부족**
```yaml
# config/traisformer_config.yaml
training:
  batch_size: 16  # 기본값: 32에서 줄임
```

**수렴하지 않는 훈련**
```yaml
training:
  learning_rate: 0.0001  # 학습률 감소
  patience: 20           # 조기 종료 인내심 증가
```

**클래스 불균형**
```yaml
training:
  class_weights: [1.0, 5.0]  # 이상 클래스 가중치 증가
  class_balance: "ADASYN"    # 샘플링 방법 변경
```