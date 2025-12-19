"""
ML Models Package
=================
Commercial-Grade AI Network Intrusion Detection Models

Includes:
- XGBoost: Fast, accurate classification
- Autoencoder: Unsupervised anomaly detection
- LSTM: Temporal pattern detection
- GNN: Network topology-aware detection (Graph Neural Networks)
- Temporal Windows: Multi-scale temporal inference
- Adaptive Ensemble: LSTM-controlled dynamic weight fusion
"""

from .xgboost_classifier import XGBoostClassifier, create_xgboost_classifier
from .autoencoder import AnomalyAutoencoder, create_autoencoder
from .lstm_detector import LSTMDetector, create_lstm_detector
from .ensemble import EnsembleDetector, create_ensemble
from .gnn_detector import (
    GNNIntrusionDetector, 
    create_gnn_detector,
    NetworkGraphBuilder,
    GNNTrainer,
    ATTACK_LABELS
)
from .temporal_windows import (
    MultiWindowTemporalDetector,
    TemporalAnomalyAnalyzer,
    create_temporal_detector,
    TemporalDataBuffer,
    WINDOW_1MIN,
    WINDOW_15MIN,
    WINDOW_1HOUR,
    WINDOW_24HOUR
)
from .adaptive_ensemble import (
    AdaptiveEnsemble,
    create_adaptive_ensemble,
    LSTMWeightController,
    ModelPerformanceTracker,
    ContextFeatures,
    NetworkState
)

__all__ = [
    # Classic models
    'XGBoostClassifier', 'create_xgboost_classifier',
    'AnomalyAutoencoder', 'create_autoencoder',
    'LSTMDetector', 'create_lstm_detector',
    'EnsembleDetector', 'create_ensemble',
    
    # GNN models
    'GNNIntrusionDetector', 'create_gnn_detector',
    'NetworkGraphBuilder', 'GNNTrainer', 'ATTACK_LABELS',
    
    # Temporal models
    'MultiWindowTemporalDetector', 'TemporalAnomalyAnalyzer',
    'create_temporal_detector', 'TemporalDataBuffer',
    'WINDOW_1MIN', 'WINDOW_15MIN', 'WINDOW_1HOUR', 'WINDOW_24HOUR',
    
    # Adaptive ensemble
    'AdaptiveEnsemble', 'create_adaptive_ensemble',
    'LSTMWeightController', 'ModelPerformanceTracker',
    'ContextFeatures', 'NetworkState'
]
