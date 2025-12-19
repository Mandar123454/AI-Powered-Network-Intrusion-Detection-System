"""
AI-NIDS Machine Learning Package
Complete ML pipeline for network intrusion detection
"""

from .preprocessing import DataPreprocessor, FeatureEngineer, create_preprocessor
from .models import (
    XGBoostClassifier, create_xgboost_classifier,
    AnomalyAutoencoder, create_autoencoder,
    LSTMDetector, create_lstm_detector,
    EnsembleDetector, create_ensemble
)
from .explainability import SHAPExplainer, create_explainer

__version__ = '1.0.0'

__all__ = [
    # Preprocessing
    'DataPreprocessor', 'FeatureEngineer', 'create_preprocessor',
    # Models
    'XGBoostClassifier', 'create_xgboost_classifier',
    'AnomalyAutoencoder', 'create_autoencoder',
    'LSTMDetector', 'create_lstm_detector',
    'EnsembleDetector', 'create_ensemble',
    # Explainability
    'SHAPExplainer', 'create_explainer'
]
