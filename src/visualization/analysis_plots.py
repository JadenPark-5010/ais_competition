"""
북한 의심 선박 탐지 대회를 위한 시각화 모듈
데이터 분포, 모델 성능, 예측 결과 등을 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class CompetitionVisualizer:
    """북한 의심 선박 탐지 대회 시각화 클래스"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.colors = {
            'suspicious': '#FF6B6B',
            'normal': '#4ECDC4',
            'primary': '#45B7D1',
            'secondary': '#96CEB4',
            'accent': '#FFEAA7'
        }
        
        # 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_data_distribution(self, df: pd.DataFrame, save_path: str):
        """데이터 분포 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('North Korean Suspicious Vessel Detection - Data Distribution', fontsize=16, fontweight='bold')
        
        # 1. 클래스 분포
        if 'is_suspicious' in df.columns:
            class_counts = df['is_suspicious'].value_counts()
            axes[0, 0].pie(class_counts.values, 
                          labels=['Normal', 'Suspicious'], 
                          colors=[self.colors['normal'], self.colors['suspicious']],
                          autopct='%1.1f%%',
                          startangle=90)
            axes[0, 0].set_title('Class Distribution')
        
        # 2. 속도 분포
        if 'sog' in df.columns:
            for label, color in [(0, self.colors['normal']), (1, self.colors['suspicious'])]:
                if 'is_suspicious' in df.columns:
                    data = df[df['is_suspicious'] == label]['sog']
                    label_name = 'Normal' if label == 0 else 'Suspicious'
                else:
                    data = df['sog']
                    label_name = 'All Vessels'
                    color = self.colors['primary']
                
                axes[0, 1].hist(data, bins=30, alpha=0.7, color=color, label=label_name)
            
            axes[0, 1].set_xlabel('Speed Over Ground (knots)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Speed Distribution')
            axes[0, 1].legend()
        
        # 3. 시간별 활동 패턴
        if 'hour' in df.columns:
            hour_activity = df.groupby(['hour', 'is_suspicious']).size().unstack(fill_value=0) if 'is_suspicious' in df.columns else df['hour'].value_counts().sort_index()
            
            if 'is_suspicious' in df.columns:
                hour_activity.plot(kind='bar', ax=axes[0, 2], 
                                 color=[self.colors['normal'], self.colors['suspicious']])
                axes[0, 2].legend(['Normal', 'Suspicious'])
            else:
                hour_activity.plot(kind='bar', ax=axes[0, 2], color=self.colors['primary'])
            
            axes[0, 2].set_xlabel('Hour of Day')
            axes[0, 2].set_ylabel('Number of Records')
            axes[0, 2].set_title('Activity by Hour')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. 지리적 분포
        if 'lat' in df.columns and 'lon' in df.columns:
            if 'is_suspicious' in df.columns:
                for label, color, marker in [(0, self.colors['normal'], 'o'), (1, self.colors['suspicious'], '^')]:
                    vessel_data = df[df['is_suspicious'] == label]
                    if len(vessel_data) > 0:
                        axes[1, 0].scatter(vessel_data['lon'], vessel_data['lat'], 
                                         c=color, alpha=0.6, s=10, marker=marker,
                                         label='Normal' if label == 0 else 'Suspicious')
            else:
                axes[1, 0].scatter(df['lon'], df['lat'], c=self.colors['primary'], alpha=0.6, s=10)
            
            axes[1, 0].set_xlabel('Longitude')
            axes[1, 0].set_ylabel('Latitude')
            axes[1, 0].set_title('Geographic Distribution')
            if 'is_suspicious' in df.columns:
                axes[1, 0].legend()
        
        # 5. 코스 변화 분포
        if 'cog' in df.columns:
            # 코스 변화 계산
            vessel_groups = df.groupby('vessel_id')
            course_changes = []
            
            for vessel_id, group in vessel_groups:
                if len(group) > 1:
                    cog_diff = group['cog'].diff().dropna()
                    # 360도 경계 처리
                    cog_diff = np.where(cog_diff > 180, cog_diff - 360, cog_diff)
                    cog_diff = np.where(cog_diff < -180, cog_diff + 360, cog_diff)
                    course_changes.extend(cog_diff.tolist())
            
            if course_changes:
                axes[1, 1].hist(course_changes, bins=50, color=self.colors['accent'], alpha=0.7)
                axes[1, 1].set_xlabel('Course Change (degrees)')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Course Change Distribution')
        
        # 6. 선박별 데이터 포인트 수
        if 'vessel_id' in df.columns:
            vessel_counts = df['vessel_id'].value_counts()
            axes[1, 2].bar(range(len(vessel_counts)), vessel_counts.values, 
                          color=self.colors['secondary'])
            axes[1, 2].set_xlabel('Vessel Index')
            axes[1, 2].set_ylabel('Number of Data Points')
            axes[1, 2].set_title('Data Points per Vessel')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"데이터 분포 시각화 저장: {save_path}")
    
    def plot_model_performance(self, evaluation_results: Dict, save_path: str):
        """모델 성능 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. 성능 지표 바 차트
        metrics = evaluation_results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[0, 0].bar(metric_names, metric_values, color=self.colors['primary'])
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 2. 혼동 행렬
        conf_matrix = evaluation_results['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Suspicious'],
                   yticklabels=['Normal', 'Suspicious'],
                   ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        # 3. 예측 확률 분포
        if 'probabilities' in evaluation_results:
            probabilities = evaluation_results['probabilities']
            predictions = evaluation_results['predictions']
            
            # 정상 선박과 의심 선박의 확률 분포
            normal_probs = probabilities[predictions == 0]
            suspicious_probs = probabilities[predictions == 1]
            
            axes[1, 0].hist(normal_probs, bins=30, alpha=0.7, 
                           color=self.colors['normal'], label='Normal')
            axes[1, 0].hist(suspicious_probs, bins=30, alpha=0.7, 
                           color=self.colors['suspicious'], label='Suspicious')
            
            axes[1, 0].set_xlabel('Prediction Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Prediction Probability Distribution')
            axes[1, 0].legend()
        
        # 4. ROC 곡선 (간단한 버전)
        if 'probabilities' in evaluation_results:
            from sklearn.metrics import roc_curve, auc
            
            # 실제 라벨이 있다고 가정 (evaluation_results에서 추출)
            y_true = np.concatenate([np.zeros(len(normal_probs)), np.ones(len(suspicious_probs))])
            y_scores = np.concatenate([normal_probs, suspicious_probs])
            
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            axes[1, 1].plot(fpr, tpr, color=self.colors['primary'], 
                           label=f'ROC Curve (AUC = {roc_auc:.3f})')
            axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[1, 1].set_xlabel('False Positive Rate')
            axes[1, 1].set_ylabel('True Positive Rate')
            axes[1, 1].set_title('ROC Curve')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"모델 성능 시각화 저장: {save_path}")
    
    def plot_prediction_results(self, submission_df: pd.DataFrame, save_path: str):
        """예측 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Prediction Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. 예측 클래스 분포
        pred_counts = submission_df['is_suspicious'].value_counts()
        axes[0, 0].pie(pred_counts.values, 
                      labels=['Normal', 'Suspicious'], 
                      colors=[self.colors['normal'], self.colors['suspicious']],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[0, 0].set_title('Predicted Class Distribution')
        
        # 2. 신뢰도 분포
        axes[0, 1].hist(submission_df['confidence'], bins=30, 
                       color=self.colors['accent'], alpha=0.7)
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Confidence Score Distribution')
        
        # 3. 클래스별 신뢰도 박스플롯
        normal_conf = submission_df[submission_df['is_suspicious'] == 0]['confidence']
        suspicious_conf = submission_df[submission_df['is_suspicious'] == 1]['confidence']
        
        box_data = [normal_conf, suspicious_conf]
        box_labels = ['Normal', 'Suspicious']
        box_colors = [self.colors['normal'], self.colors['suspicious']]
        
        bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_ylabel('Confidence Score')
        axes[1, 0].set_title('Confidence by Predicted Class')
        
        # 4. 신뢰도 vs 예측 클래스 산점도
        colors = [self.colors['normal'] if x == 0 else self.colors['suspicious'] 
                 for x in submission_df['is_suspicious']]
        
        axes[1, 1].scatter(submission_df['is_suspicious'], submission_df['confidence'], 
                          c=colors, alpha=0.6)
        axes[1, 1].set_xlabel('Predicted Class')
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].set_title('Confidence vs Prediction')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_xticklabels(['Normal', 'Suspicious'])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"예측 결과 시각화 저장: {save_path}")
    
    def plot_vessel_trajectory(self, vessel_data: pd.DataFrame, save_path: str):
        """개별 선박 궤적 시각화"""
        if len(vessel_data) == 0:
            print("선박 데이터가 비어있습니다.")
            return
        
        vessel_id = vessel_data['vessel_id'].iloc[0]
        is_suspicious = vessel_data['is_suspicious'].iloc[0] if 'is_suspicious' in vessel_data.columns else 0
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Vessel Trajectory Analysis - {vessel_id}', fontsize=16, fontweight='bold')
        
        # 색상 설정
        color = self.colors['suspicious'] if is_suspicious else self.colors['normal']
        status = 'Suspicious' if is_suspicious else 'Normal'
        
        # 1. 지리적 궤적
        if 'lat' in vessel_data.columns and 'lon' in vessel_data.columns:
            # 시간 순서대로 정렬
            vessel_data_sorted = vessel_data.sort_values('datetime')
            
            # 궤적 라인
            axes[0, 0].plot(vessel_data_sorted['lon'], vessel_data_sorted['lat'], 
                           color=color, alpha=0.7, linewidth=2)
            
            # 시작점과 끝점 표시
            axes[0, 0].scatter(vessel_data_sorted['lon'].iloc[0], vessel_data_sorted['lat'].iloc[0], 
                              color='green', s=100, marker='o', label='Start', zorder=5)
            axes[0, 0].scatter(vessel_data_sorted['lon'].iloc[-1], vessel_data_sorted['lat'].iloc[-1], 
                              color='red', s=100, marker='s', label='End', zorder=5)
            
            axes[0, 0].set_xlabel('Longitude')
            axes[0, 0].set_ylabel('Latitude')
            axes[0, 0].set_title(f'Geographic Trajectory ({status})')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 속도 시계열
        if 'sog' in vessel_data.columns and 'datetime' in vessel_data.columns:
            vessel_data_sorted = vessel_data.sort_values('datetime')
            axes[0, 1].plot(vessel_data_sorted['datetime'], vessel_data_sorted['sog'], 
                           color=color, linewidth=2)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Speed Over Ground (knots)')
            axes[0, 1].set_title('Speed Time Series')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 코스 시계열
        if 'cog' in vessel_data.columns and 'datetime' in vessel_data.columns:
            vessel_data_sorted = vessel_data.sort_values('datetime')
            axes[1, 0].plot(vessel_data_sorted['datetime'], vessel_data_sorted['cog'], 
                           color=color, linewidth=2)
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Course Over Ground (degrees)')
            axes[1, 0].set_title('Course Time Series')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 속도-코스 관계
        if 'sog' in vessel_data.columns and 'cog' in vessel_data.columns:
            scatter = axes[1, 1].scatter(vessel_data['sog'], vessel_data['cog'], 
                                       c=range(len(vessel_data)), cmap='viridis', 
                                       alpha=0.7, s=30)
            axes[1, 1].set_xlabel('Speed Over Ground (knots)')
            axes[1, 1].set_ylabel('Course Over Ground (degrees)')
            axes[1, 1].set_title('Speed vs Course (colored by time)')
            plt.colorbar(scatter, ax=axes[1, 1], label='Time sequence')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"선박 궤적 시각화 저장: {save_path}")
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], save_path: str, top_n: int = 20):
        """피처 중요도 시각화"""
        # 상위 N개 피처 선택
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color=self.colors['primary'])
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        
        # 값 표시
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"피처 중요도 시각화 저장: {save_path}")
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: str, features: Optional[List[str]] = None):
        """상관관계 매트릭스 시각화"""
        if features is None:
            # 수치형 컬럼만 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # 너무 많으면 상위 20개만
            features = numeric_cols[:20]
        
        correlation_matrix = df[features].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"상관관계 매트릭스 시각화 저장: {save_path}")
    
    def create_dashboard(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                        evaluation_results: Dict, submission_df: pd.DataFrame, save_path: str):
        """종합 대시보드 생성"""
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('North Korean Suspicious Vessel Detection - Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # 1. 클래스 분포 (훈련 데이터)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'is_suspicious' in train_df.columns:
            class_counts = train_df['is_suspicious'].value_counts()
            ax1.pie(class_counts.values, labels=['Normal', 'Suspicious'], 
                   colors=[self.colors['normal'], self.colors['suspicious']],
                   autopct='%1.1f%%')
            ax1.set_title('Training Data\nClass Distribution')
        
        # 2. 예측 결과 분포
        ax2 = fig.add_subplot(gs[0, 1])
        pred_counts = submission_df['is_suspicious'].value_counts()
        ax2.pie(pred_counts.values, labels=['Normal', 'Suspicious'], 
               colors=[self.colors['normal'], self.colors['suspicious']],
               autopct='%1.1f%%')
        ax2.set_title('Prediction Results\nClass Distribution')
        
        # 3. 모델 성능 지표
        ax3 = fig.add_subplot(gs[0, 2:])
        metrics = evaluation_results['metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = ax3.bar(metric_names, metric_values, color=self.colors['primary'])
        ax3.set_title('Model Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for bar, value in zip(bars, metric_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 지리적 분포
        ax4 = fig.add_subplot(gs[1, :2])
        if 'lat' in train_df.columns and 'lon' in train_df.columns:
            if 'is_suspicious' in train_df.columns:
                for label, color, marker in [(0, self.colors['normal'], 'o'), (1, self.colors['suspicious'], '^')]:
                    vessel_data = train_df[train_df['is_suspicious'] == label]
                    if len(vessel_data) > 0:
                        ax4.scatter(vessel_data['lon'], vessel_data['lat'], 
                                   c=color, alpha=0.6, s=20, marker=marker,
                                   label='Normal' if label == 0 else 'Suspicious')
            ax4.set_xlabel('Longitude')
            ax4.set_ylabel('Latitude')
            ax4.set_title('Geographic Distribution of Vessels')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. 혼동 행렬
        ax5 = fig.add_subplot(gs[1, 2:])
        conf_matrix = evaluation_results['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Suspicious'],
                   yticklabels=['Normal', 'Suspicious'],
                   ax=ax5)
        ax5.set_title('Confusion Matrix')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        
        # 6. 속도 분포
        ax6 = fig.add_subplot(gs[2, :2])
        if 'sog' in train_df.columns:
            if 'is_suspicious' in train_df.columns:
                for label, color in [(0, self.colors['normal']), (1, self.colors['suspicious'])]:
                    data = train_df[train_df['is_suspicious'] == label]['sog']
                    label_name = 'Normal' if label == 0 else 'Suspicious'
                    ax6.hist(data, bins=30, alpha=0.7, color=color, label=label_name)
                ax6.legend()
            ax6.set_xlabel('Speed Over Ground (knots)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Speed Distribution by Class')
        
        # 7. 신뢰도 분포
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.hist(submission_df['confidence'], bins=30, 
                color=self.colors['accent'], alpha=0.7)
        ax7.set_xlabel('Confidence Score')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Prediction Confidence Distribution')
        
        # 8. 시간별 활동 패턴
        ax8 = fig.add_subplot(gs[3, :])
        if 'hour' in train_df.columns:
            if 'is_suspicious' in train_df.columns:
                hour_activity = train_df.groupby(['hour', 'is_suspicious']).size().unstack(fill_value=0)
                hour_activity.plot(kind='bar', ax=ax8, 
                                 color=[self.colors['normal'], self.colors['suspicious']])
                ax8.legend(['Normal', 'Suspicious'])
            else:
                hour_activity = train_df['hour'].value_counts().sort_index()
                hour_activity.plot(kind='bar', ax=ax8, color=self.colors['primary'])
            
            ax8.set_xlabel('Hour of Day')
            ax8.set_ylabel('Number of Records')
            ax8.set_title('Vessel Activity by Hour of Day')
            ax8.tick_params(axis='x', rotation=0)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"종합 대시보드 저장: {save_path}") 