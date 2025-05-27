"""
Maritime Anomaly Detection Package

AIS 데이터를 활용한 해상 이상 탐지 시스템
"""

__version__ = "0.1.0"
__author__ = "Maritime AI Team"
__email__ = "team@maritime-ai.com"

from . import data
from . import features
from . import models
# from . import training  # 불필요한 의존성으로 인해 주석 처리
from . import utils

__all__ = [
    "data",
    "features", 
    "models",
    "utils"
] 