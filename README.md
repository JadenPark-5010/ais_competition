# AIS 기반 해상 이상 탐지 시스템

## 📋 프로젝트 개요

이 프로젝트는 AIS(Automatic Identification System) 데이터를 활용하여 선박의 이상 행동을 탐지하는 머신러닝 시스템입니다. TrAISformer, 클러스터링, 앙상블 모델을 조합하여 높은 정확도의 이상 탐지를 수행합니다.

## 🗃️ 데이터 스키마

AIS 데이터는 26개 컬럼으로 구성되어 있습니다:

1. **Timestamp** - 시간 정보 (31/12/2015 23:59:59)
2. **Type of mobile** - AIS 장비 유형 (Class A/B AIS Vessel)
3. **MMSI** - 선박 고유번호
4. **Latitude** - 위도 (57.8794)
5. **Longitude** - 경도 (17.9125)
6. **Navigational status** - 항해 상태
7. **ROT** - 회전율 (Rate of Turn)
8. **SOG** - 대지속도 (Speed Over Ground)
9. **COG** - 대지침로 (Course Over Ground)
10. **Heading** - 선수방위
11. **IMO** - IMO 번호
12. **Callsign** - 호출부호
13. **Name** - 선박명
14. **Ship type** - 선박 유형
15. **Cargo type** - 화물 유형
16. **Width** - 선박 폭
17. **Length** - 선박 길이
18. **Type of position fixing device** - GPS 장치 유형
19. **Draught** - 흘수
20. **Destination** - 목적지
21. **ETA** - 예상 도착시간
22. **Data source type** - 데이터 소스
23. **Size A** - GPS~선수 길이
24. **Size B** - GPS~선미 길이
25. **Size C** - GPS~우현 길이
26. **Size D** - GPS~좌현 길이

## 🏗️ 프로젝트 구조

```
maritime_anomaly_detection/
├── README.md                        # 프로젝트 설명
├── requirements.txt                 # 의존성
├── setup.py                         # 패키지 설치
├── .gitignore                       # Git 무시 파일
├── config/
│   ├── config.yaml                  # 전역 설정
│   └── model_configs.yaml           # 모델별 하이퍼파라미터
├── src/
│   ├── data/
│   │   ├── data_loader.py           # 데이터 로딩
│   │   └── preprocessing.py         # 전처리
│   ├── features/
│   │   ├── feature_engineering.py   # 특징 추출
│   │   └── feature_selection.py     # 특징 선택
│   ├── models/
│   │   ├── traisformer.py           # TrAISformer 모델
│   │   ├── clustering_model.py      # 클러스터링 모델
│   │   ├── ensemble_model.py        # 앙상블 모델
│   │   └── base_model.py            # 베이스 클래스
│   ├── training/
│   │   ├── trainer.py               # 학습 로직
│   │   └── validator.py             # 검증 로직
│   └── utils/
│       ├── metrics.py               # 평가 지표
│       ├── visualization.py         # 시각화
│       └── logging_utils.py         # 로깅
├── scripts/
│   ├── train.py                     # 학습 실행
│   ├── predict.py                   # 예측 실행
│   └── submit.py                    # 제출 파일 생성
├── notebooks/                       # EDA, 실험 노트북
├── tests/                           # 단위 테스트
└── models/                          # 저장된 모델
```

## 🚀 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd maritime_anomaly_detection
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 패키지 설치
```bash
pip install -e .
```

## 🎯 사용 방법

### 1. 전체 파이프라인 실행
```bash
# 기본 설정으로 학습
python scripts/train.py --config config/config.yaml

# 커스텀 설정으로 학습
python scripts/train.py --config config/custom_config.yaml --output_dir results/experiment_1
```

### 2. 예측 실행
```bash
# 단일 모델 예측
python scripts/predict.py --model_path models/final_model.pkl --data_path data/test.csv

# 앙상블 예측
python scripts/predict.py --model_path models/ensemble_model.pkl --data_path data/test.csv --ensemble
```

### 3. 제출 파일 생성
```bash
python scripts/submit.py --test_data data/test/ --model_path models/final_model.pkl --output submissions/submission.csv
```

## 🔧 주요 기능

### Feature Engineering
- **운동학적 특징**: 속도, 가속도, 방향 변화 통계량
- **지리적 특징**: 이동거리, 궤적 복잡도, 해상구역 분석
- **시간적 특징**: 항해 지속시간, 주기성 분석
- **행동적 특징**: 항해상태별 분석, 급격한 기동 탐지
- **TrAISformer 특징**: Four-hot encoding, 궤적 엔트로피

### 모델 아키텍처
1. **TrAISformer**: Transformer 기반 이상 탐지
2. **Clustering Model**: DBSCAN + Isolation Forest
3. **Ensemble Model**: 동적 가중치 앙상블

### 성능 최적화
- 배치 처리를 통한 메모리 효율성
- 멀티프로세싱 병렬 처리
- GPU 지원 (CUDA 사용 가능 시)

## 📊 실험 관리

### 설정 파일 수정
`config/config.yaml`에서 하이퍼파라미터 조정:
```yaml
model:
  traisformer:
    d_model: 256
    nhead: 8
    num_layers: 6
  ensemble:
    weights: [0.4, 0.3, 0.3]
```

### 로그 확인
```bash
# 학습 로그 확인
tail -f logs/training.log

# 실험 결과 확인
cat results/experiment_summary.json
```

## 🧪 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/

# 특정 모듈 테스트
python -m pytest tests/test_feature_engineering.py -v
```

## 📈 성능 지표

- **정확도 (Accuracy)**
- **정밀도 (Precision)**
- **재현율 (Recall)**
- **F1-Score**
- **AUC-ROC**