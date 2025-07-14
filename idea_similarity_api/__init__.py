"""
아이디어 유사도 측정 API 패키지

창업 아이디어의 의미적 유사도를 측정하고, 사용자 피드백을 반영한 지능형 추천 시스템입니다.
"""

__version__ = "1.0.0"
__author__ = "AI Study Team"

from .core import IdeaSimilarityEngine
from .api_server import app

__all__ = ["IdeaSimilarityEngine", "app"] 