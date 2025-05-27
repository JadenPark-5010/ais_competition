"""
북한 의심 선박 탐지를 위한 앙상블 모델
여러 머신러닝 모델을 조합하여 높은 성능과 신뢰도를 달성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 모델들
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# 불균형 데이터 처리
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# 개별 모델들
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# 모델 해석
import shap
from sklearn.inspection import permutation_importance

class SuspiciousVesselEnsemble:
    """북한 의심 선박 탐지 앙상블 모델"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 모델 설정 딕셔너리
        """
        self.config = config
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # 개별 모델 초기화
        self._initialize_models()
        
    def _initialize_models(self):
        """개별 모델들 초기화"""
        model_configs = self.config['models']['ensemble']
        
        for model_config in model_configs:
            model_name = model_config['name']
            model_type = model_config['type']
            params = model_config['params']
            
            if model_type == 'XGBClassifier':
                self.models[model_name] = xgb.XGBClassifier(**params)
            elif model_type == 'LGBMClassifier':
                self.models[model_name] = lgb.LGBMClassifier(**params)
            elif model_type == 'CatBoostClassifier':
                self.models[model_name] = CatBoostClassifier(**params)
            elif model_type == 'RandomForestClassifier':
                self.models[model_name] = RandomForestClassifier(**params)
            elif model_type == 'MLPClassifier':
                self.models[model_name] = MLPClassifier(**params)
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
        
        print(f"초기화된 모델: {list(self.models.keys())}")
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'is_suspicious') -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리 및 준비"""
        # 피처와 타겟 분리
        feature_cols = [col for col in df.columns if col not in ['vessel_id', target_col, 'confidence']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 결측값 처리
        X = X.fillna(X.median())
        
        # 피처명 저장
        self.feature_names = feature_cols
        
        return X.values, y.values
    
    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """클래스 불균형 처리"""
        balance_config = self.config['training']['class_balance']
        method = balance_config['method']
        sampling_strategy = balance_config['sampling_strategy']
        
        print(f"클래스 불균형 처리: {method}")
        print(f"원본 클래스 분포: {np.bincount(y)}")
        
        if method == 'SMOTE':
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'ADASYN':
            sampler = ADASYN(sampling_strategy=sampling_strategy, random_state=42)
        elif method == 'RandomOverSampler':
            sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        else:
            print("클래스 불균형 처리 없음")
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"리샘플링 후 클래스 분포: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def train_individual_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """개별 모델 훈련"""
        print("개별 모델 훈련 시작...")
        
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 교차 검증 설정
        cv_folds = self.config['training']['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        model_scores = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name} 훈련 중...")
            
            # 모델별로 스케일링 필요 여부 결정
            if model_name in ['neural_network']:
                X_train = X_scaled
            else:
                X_train = X
            
            # 교차 검증
            cv_scores = cross_val_score(model, X_train, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            print(f"{model_name} CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # 전체 데이터로 훈련
            model.fit(X_train, y)
            
            model_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
        
        return model_scores
    
    def create_ensemble(self, model_scores: Dict[str, Any], X: np.ndarray, y: np.ndarray):
        """앙상블 모델 생성"""
        print("\n앙상블 모델 생성 중...")
        
        # 성능 기반으로 모델 선택 (상위 모델들만 사용)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['cv_mean'], reverse=True)
        top_models = sorted_models[:4]  # 상위 4개 모델 사용
        
        print("앙상블에 포함된 모델:")
        for model_name, scores in top_models:
            print(f"  {model_name}: {scores['cv_mean']:.4f}")
        
        # VotingClassifier 생성
        estimators = []
        for model_name, scores in top_models:
            model = scores['model']
            
            # 신뢰도 보정 적용
            calibrated_model = CalibratedClassifierCV(
                model, 
                method=self.config['evaluation']['calibration']['method'],
                cv=self.config['evaluation']['calibration']['cv_folds']
            )
            
            estimators.append((model_name, calibrated_model))
        
        # 소프트 보팅 앙상블
        self.ensemble_model = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # 앙상블 훈련
        X_scaled = self.scaler.transform(X)
        self.ensemble_model.fit(X_scaled, y)
        
        self.is_fitted = True
        print("앙상블 모델 훈련 완료")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'is_suspicious') -> Dict[str, Any]:
        """전체 훈련 파이프라인"""
        print("=== 북한 의심 선박 탐지 모델 훈련 시작 ===")
        
        # 데이터 준비
        X, y = self.prepare_data(df, target_col)
        
        # 클래스 불균형 처리
        X_balanced, y_balanced = self.handle_class_imbalance(X, y)
        
        # 개별 모델 훈련
        model_scores = self.train_individual_models(X_balanced, y_balanced)
        
        # 앙상블 생성
        self.create_ensemble(model_scores, X_balanced, y_balanced)
        
        # 훈련 결과 반환
        return {
            'model_scores': model_scores,
            'feature_names': self.feature_names,
            'training_samples': len(X_balanced),
            'class_distribution': np.bincount(y_balanced)
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """예측 수행"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 피처 준비
        feature_cols = [col for col in df.columns if col not in ['vessel_id', 'is_suspicious', 'confidence']]
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        
        # 정규화
        X_scaled = self.scaler.transform(X)
        
        # 예측
        predictions = self.ensemble_model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """확률 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 피처 준비
        feature_cols = [col for col in df.columns if col not in ['vessel_id', 'is_suspicious', 'confidence']]
        X = df[feature_cols].fillna(df[feature_cols].median()).values
        
        # 정규화
        X_scaled = self.scaler.transform(X)
        
        # 확률 예측
        probabilities = self.ensemble_model.predict_proba(X_scaled)
        
        return probabilities
    
    def evaluate(self, df: pd.DataFrame, target_col: str = 'is_suspicious') -> Dict[str, Any]:
        """모델 평가"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 예측
        y_true = df[target_col].values
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)[:, 1]  # 양성 클래스 확률
        
        # 평가 지표 계산
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba)
        }
        
        # 분류 리포트
        classification_rep = classification_report(y_true, y_pred, output_dict=True)
        
        # 혼동 행렬
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            'metrics': metrics,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_proba
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """피처 중요도 분석"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 앙상블의 각 모델에서 피처 중요도 추출
        feature_importance = {}
        
        for name, estimator in self.ensemble_model.named_estimators_.items():
            # CalibratedClassifierCV에서 base_estimator 추출 (sklearn 버전 호환성)
            try:
                if hasattr(estimator, 'calibrated_classifiers_'):
                    # sklearn 1.0+ 버전
                    if hasattr(estimator.calibrated_classifiers_[0], 'estimator'):
                        base_estimator = estimator.calibrated_classifiers_[0].estimator
                    else:
                        base_estimator = estimator.calibrated_classifiers_[0].base_estimator
                else:
                    base_estimator = estimator
            except:
                # 직접 모델에서 중요도 추출 시도
                base_estimator = estimator
            
            if hasattr(base_estimator, 'feature_importances_'):
                importance = base_estimator.feature_importances_
                for i, feature_name in enumerate(self.feature_names):
                    if feature_name not in feature_importance:
                        feature_importance[feature_name] = []
                    feature_importance[feature_name].append(importance[i])
        
        # 평균 중요도 계산
        avg_importance = {}
        for feature_name, importances in feature_importance.items():
            avg_importance[feature_name] = np.mean(importances)
        
        # 정규화
        total_importance = sum(avg_importance.values())
        if total_importance > 0:
            for feature_name in avg_importance:
                avg_importance[feature_name] /= total_importance
        
        return avg_importance
    
    def explain_predictions(self, df: pd.DataFrame, sample_size: int = 100) -> Dict[str, Any]:
        """SHAP을 사용한 예측 설명"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 샘플 데이터 준비
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        feature_cols = [col for col in sample_df.columns if col not in ['vessel_id', 'is_suspicious', 'confidence']]
        X_sample = sample_df[feature_cols].fillna(sample_df[feature_cols].median()).values
        X_sample_scaled = self.scaler.transform(X_sample)
        
        try:
            # SHAP Explainer 생성 (TreeExplainer 시도)
            explainer = shap.TreeExplainer(self.ensemble_model.estimators_[0])
            shap_values = explainer.shap_values(X_sample_scaled)
            
            return {
                'shap_values': shap_values,
                'feature_names': self.feature_names,
                'sample_data': X_sample,
                'explainer_type': 'tree'
            }
        except:
            try:
                # KernelExplainer 사용
                explainer = shap.KernelExplainer(
                    self.ensemble_model.predict_proba, 
                    X_sample_scaled[:10]  # 배경 데이터
                )
                shap_values = explainer.shap_values(X_sample_scaled[:20])  # 설명할 샘플
                
                return {
                    'shap_values': shap_values,
                    'feature_names': self.feature_names,
                    'sample_data': X_sample[:20],
                    'explainer_type': 'kernel'
                }
            except Exception as e:
                print(f"SHAP 분석 실패: {e}")
                return None
    
    def hyperparameter_tuning(self, df: pd.DataFrame, target_col: str = 'is_suspicious') -> Dict[str, Any]:
        """하이퍼파라미터 튜닝"""
        print("하이퍼파라미터 튜닝 시작...")
        
        X, y = self.prepare_data(df, target_col)
        X_balanced, y_balanced = self.handle_class_imbalance(X, y)
        
        # 주요 모델들에 대해 그리드 서치 수행
        tuning_results = {}
        
        # XGBoost 튜닝
        if 'xgboost' in self.models:
            print("XGBoost 하이퍼파라미터 튜닝...")
            xgb_params = {
                'n_estimators': [500, 1000],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            }
            
            xgb_grid = GridSearchCV(
                xgb.XGBClassifier(random_state=42),
                xgb_params,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            xgb_grid.fit(X_balanced, y_balanced)
            tuning_results['xgboost'] = {
                'best_params': xgb_grid.best_params_,
                'best_score': xgb_grid.best_score_
            }
        
        # LightGBM 튜닝
        if 'lightgbm' in self.models:
            print("LightGBM 하이퍼파라미터 튜닝...")
            lgb_params = {
                'n_estimators': [500, 1000],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9]
            }
            
            lgb_grid = GridSearchCV(
                lgb.LGBMClassifier(random_state=42),
                lgb_params,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            lgb_grid.fit(X_balanced, y_balanced)
            tuning_results['lightgbm'] = {
                'best_params': lgb_grid.best_params_,
                'best_score': lgb_grid.best_score_
            }
        
        return tuning_results
    
    def save_model(self, filepath: str):
        """모델 저장"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"모델이 {filepath}에 저장되었습니다.")
    
    def load_model(self, filepath: str):
        """모델 로드"""
        model_data = joblib.load(filepath)
        
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.config = model_data['config']
        self.is_fitted = True
        
        print(f"모델이 {filepath}에서 로드되었습니다.")
    
    def generate_submission(self, test_df: pd.DataFrame, output_path: str):
        """대회 제출 파일 생성"""
        if not self.is_fitted:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 예측 수행
        predictions = self.predict(test_df)
        probabilities = self.predict_proba(test_df)[:, 1]  # 의심 선박일 확률
        
        # 선박별로 집계 (여러 시점의 데이터가 있는 경우)
        vessel_predictions = test_df.groupby('vessel_id').agg({
            'vessel_id': 'first'
        }).reset_index(drop=True)
        
        # 선박별 예측 결과 집계
        vessel_results = []
        for vessel_id in test_df['vessel_id'].unique():
            vessel_mask = test_df['vessel_id'] == vessel_id
            vessel_preds = predictions[vessel_mask]
            vessel_probs = probabilities[vessel_mask]
            
            # 다수결 투표 또는 평균 확률 사용
            final_prediction = int(vessel_probs.mean() > 0.5)
            final_confidence = vessel_probs.mean()
            
            vessel_results.append({
                'vessel_id': vessel_id,
                'is_suspicious': final_prediction,
                'confidence': final_confidence
            })
        
        # 제출 파일 생성
        submission_df = pd.DataFrame(vessel_results)
        submission_df.to_csv(output_path, index=False)
        
        print(f"제출 파일이 {output_path}에 저장되었습니다.")
        print(f"총 {len(submission_df)} 선박 예측 완료")
        print(f"의심 선박 예측 수: {submission_df['is_suspicious'].sum()}")
        print(f"평균 신뢰도: {submission_df['confidence'].mean():.4f}")
        
        return submission_df

def create_ensemble_pipeline(config: Dict) -> SuspiciousVesselEnsemble:
    """앙상블 파이프라인 생성"""
    return SuspiciousVesselEnsemble(config) 