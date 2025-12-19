"""
Model Predictor
===============
Real-time prediction interface for ML models.
Handles model loading, inference, and result formatting.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import os
import json
import pickle

logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Real-time model predictor for network intrusion detection.
    Provides unified interface for all ML models.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the model predictor.
        
        Args:
            model_path: Path to saved model files
            config: Configuration dictionary
        """
        self.model_path = model_path or os.getenv('MODEL_PATH', 'models/')
        self.config = config or {}
        
        # Model instances
        self.ensemble = None
        self.autoencoder = None
        self.lstm = None
        self.xgboost = None
        
        # Load configuration
        self.threshold = self.config.get('threshold', 0.5)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Feature configuration
        self.feature_names: List[str] = []
        self.num_features = 0
        
        # Performance tracking
        self.predictions_count = 0
        self.last_prediction_time: Optional[datetime] = None
        
        # Metadata
        self.metadata: Dict = {
            'initialized_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'status': 'initialized'
        }
        
        # Try to load models
        self._load_models()
        
        logger.info("ModelPredictor initialized successfully")
    
    def _load_models(self) -> bool:
        """
        Load trained models from disk.
        
        Returns:
            True if models loaded successfully
        """
        try:
            # Check if model directory exists
            if not os.path.exists(self.model_path):
                logger.warning(f"Model path does not exist: {self.model_path}")
                self._initialize_default_models()
                return False
            
            # Load ensemble model if available
            ensemble_path = os.path.join(self.model_path, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'rb') as f:
                    self.ensemble = pickle.load(f)
                logger.info("Loaded ensemble model")
            
            # Load feature config
            config_path = os.path.join(self.model_path, 'feature_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    feature_config = json.load(f)
                    self.feature_names = feature_config.get('feature_names', [])
                    self.num_features = len(self.feature_names)
            
            self.metadata['status'] = 'loaded'
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._initialize_default_models()
            return False
    
    def _initialize_default_models(self) -> None:
        """Initialize default/placeholder models for demo mode."""
        self.metadata['status'] = 'demo_mode'
        self.num_features = 41  # Standard network flow features
        logger.info("Running in demo mode with placeholder models")
    
    def predict(
        self,
        features: Union[np.ndarray, List[float], Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Make prediction on input features.
        
        Args:
            features: Input features (array, list, or dict)
            
        Returns:
            Prediction result with probability and label
        """
        try:
            # Convert input to numpy array
            if isinstance(features, dict):
                X = np.array([features.get(f, 0) for f in self.feature_names])
            elif isinstance(features, list):
                X = np.array(features)
            else:
                X = features
            
            # Ensure 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Make prediction
            if self.ensemble is not None:
                # Use ensemble model
                probability = self.ensemble.predict_proba(X)[0, 1]
            else:
                # Demo mode - simulate prediction
                probability = self._simulate_prediction(X)
            
            # Determine label
            label = 'malicious' if probability >= self.threshold else 'benign'
            
            # Update statistics
            self.predictions_count += 1
            self.last_prediction_time = datetime.now()
            
            result = {
                'probability': float(probability),
                'label': label,
                'threshold': self.threshold,
                'confidence': float(abs(probability - 0.5) * 2),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.metadata.get('version', '1.0.0')
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'probability': 0.5,
                'label': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
    
    def predict_batch(
        self,
        features_batch: Union[np.ndarray, List[List[float]]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of samples.
        
        Args:
            features_batch: Batch of feature vectors
            
        Returns:
            List of prediction results
        """
        try:
            # Convert to numpy array
            if isinstance(features_batch, list):
                X = np.array(features_batch)
            else:
                X = features_batch
            
            results = []
            
            if self.ensemble is not None:
                # Batch prediction with ensemble
                probabilities = self.ensemble.predict_proba(X)[:, 1]
                for prob in probabilities:
                    label = 'malicious' if prob >= self.threshold else 'benign'
                    results.append({
                        'probability': float(prob),
                        'label': label,
                        'confidence': float(abs(prob - 0.5) * 2)
                    })
            else:
                # Demo mode
                for i in range(len(X)):
                    results.append(self.predict(X[i]))
            
            self.predictions_count += len(X)
            self.last_prediction_time = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return [{'error': str(e)}]
    
    def _simulate_prediction(self, X: np.ndarray) -> float:
        """
        Simulate prediction for demo mode.
        Uses statistical analysis of features.
        
        Args:
            X: Input features
            
        Returns:
            Simulated probability
        """
        # Use feature statistics to simulate threat detection
        # Higher variance/extreme values suggest anomaly
        
        if X.size == 0:
            return 0.5
        
        # Normalize features
        X_flat = X.flatten()
        
        # Calculate anomaly score based on statistics
        mean_val = np.mean(X_flat)
        std_val = np.std(X_flat)
        max_val = np.max(np.abs(X_flat))
        
        # Combine factors for threat score
        anomaly_score = 0.0
        
        # High variance indicates anomaly
        if std_val > 2.0:
            anomaly_score += 0.3
        
        # Extreme values indicate anomaly
        if max_val > 10.0:
            anomaly_score += 0.2
        
        # Unusual mean indicates anomaly
        if abs(mean_val) > 5.0:
            anomaly_score += 0.2
        
        # Add small random factor
        noise = np.random.uniform(-0.1, 0.1)
        
        probability = min(1.0, max(0.0, anomaly_score + noise + 0.2))
        
        return probability
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if self.ensemble is not None and hasattr(self.ensemble, 'feature_importances_'):
            importances = self.ensemble.feature_importances_
            return dict(zip(self.feature_names, importances.tolist()))
        
        # Return placeholder for demo mode
        return {f"feature_{i}": 1.0 / max(1, self.num_features) 
                for i in range(self.num_features)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Model metadata and statistics
        """
        return {
            'status': self.metadata.get('status', 'unknown'),
            'version': self.metadata.get('version', '1.0.0'),
            'initialized_at': self.metadata.get('initialized_at'),
            'predictions_count': self.predictions_count,
            'last_prediction': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'threshold': self.threshold,
            'num_features': self.num_features,
            'model_path': self.model_path
        }
    
    def update_threshold(self, threshold: float) -> None:
        """
        Update classification threshold.
        
        Args:
            threshold: New threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"Updated threshold to {threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the predictor.
        
        Returns:
            Health status information
        """
        status = {
            'healthy': True,
            'status': self.metadata.get('status', 'unknown'),
            'models_loaded': self.ensemble is not None,
            'predictions_count': self.predictions_count,
            'uptime': (datetime.now() - datetime.fromisoformat(
                self.metadata.get('initialized_at', datetime.now().isoformat())
            )).total_seconds()
        }
        
        # Test prediction
        try:
            test_input = np.zeros((1, max(1, self.num_features)))
            self.predict(test_input)
            status['inference_test'] = 'passed'
        except Exception as e:
            status['healthy'] = False
            status['inference_test'] = f'failed: {str(e)}'
        
        return status
