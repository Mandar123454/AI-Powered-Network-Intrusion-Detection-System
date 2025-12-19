"""
Test ML Models
==============
Tests for machine learning models
"""

import pytest
import numpy as np
import pandas as pd


class TestPreprocessor:
    """Test data preprocessor."""
    
    def test_preprocessor_init(self):
        """Test preprocessor initialization."""
        from ml.preprocessing.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        assert preprocessor is not None
    
    def test_preprocessor_fit_transform(self):
        """Test fit_transform method."""
        from ml.preprocessing.preprocessor import DataPreprocessor
        
        # Create sample data
        df = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'category': ['a', 'b', 'a', 'c', 'b']
        })
        
        preprocessor = DataPreprocessor()
        result = preprocessor.fit_transform(df)
        
        assert result is not None
        assert len(result) == len(df)
    
    def test_feature_engineer(self):
        """Test feature engineering."""
        from ml.preprocessing.preprocessor import FeatureEngineer
        
        engineer = FeatureEngineer()
        
        # Create sample network flow data
        df = pd.DataFrame({
            'duration': [1.0, 2.0, 3.0],
            'src_bytes': [100, 200, 300],
            'dst_bytes': [50, 100, 150],
            'src_port': [54321, 54322, 54323],
            'dst_port': [80, 443, 22]
        })
        
        result = engineer.create_features(df)
        assert result is not None


class TestXGBoostClassifier:
    """Test XGBoost classifier."""
    
    def test_xgboost_init(self):
        """Test XGBoost initialization."""
        from ml.models.xgboost_classifier import XGBoostClassifier
        
        model = XGBoostClassifier()
        assert model is not None
    
    def test_xgboost_train_predict(self, temp_model_dir):
        """Test XGBoost training and prediction."""
        from ml.models.xgboost_classifier import XGBoostClassifier
        
        # Create sample data
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.randn(20, 10))
        
        model = XGBoostClassifier()
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_xgboost_predict_proba(self, temp_model_dir):
        """Test XGBoost probability prediction."""
        from ml.models.xgboost_classifier import XGBoostClassifier
        
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.randn(20, 10))
        
        model = XGBoostClassifier()
        model.train(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)
        
        assert probabilities.shape == (20, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestAutoencoder:
    """Test Autoencoder model."""
    
    def test_autoencoder_init(self):
        """Test Autoencoder initialization."""
        from ml.models.autoencoder import AnomalyAutoencoder
        
        model = AnomalyAutoencoder(input_dim=10)
        assert model is not None
    
    def test_autoencoder_train(self, temp_model_dir):
        """Test Autoencoder training."""
        from ml.models.autoencoder import AnomalyAutoencoder
        
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        
        model = AnomalyAutoencoder(input_dim=10, encoding_dim=4)
        history = model.train(X_train, epochs=5, batch_size=32)
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 5
    
    def test_autoencoder_detect(self, temp_model_dir):
        """Test Autoencoder anomaly detection."""
        from ml.models.autoencoder import AnomalyAutoencoder
        
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        X_test = np.random.randn(20, 10).astype(np.float32)
        
        model = AnomalyAutoencoder(input_dim=10, encoding_dim=4)
        model.train(X_train, epochs=5, batch_size=32)
        
        scores = model.get_reconstruction_error(X_test)
        
        assert len(scores) == 20
        assert all(s >= 0 for s in scores)


class TestLSTMDetector:
    """Test LSTM detector."""
    
    def test_lstm_init(self):
        """Test LSTM initialization."""
        from ml.models.lstm_detector import LSTMDetector
        
        model = LSTMDetector(input_size=10)
        assert model is not None
    
    def test_lstm_train(self, temp_model_dir):
        """Test LSTM training."""
        from ml.models.lstm_detector import LSTMDetector
        
        np.random.seed(42)
        X_train = np.random.randn(100, 10).astype(np.float32)
        y_train = np.random.randint(0, 2, 100)
        
        model = LSTMDetector(input_size=10, sequence_length=5)
        history = model.train(X_train, y_train, epochs=3, batch_size=16)
        
        assert 'train_loss' in history


class TestEnsemble:
    """Test Ensemble detector."""
    
    def test_ensemble_init(self, temp_model_dir):
        """Test Ensemble initialization."""
        from ml.models.ensemble import EnsembleDetector
        
        ensemble = EnsembleDetector(model_dir=temp_model_dir)
        assert ensemble is not None
    
    @pytest.mark.slow
    def test_ensemble_train(self, temp_model_dir):
        """Test Ensemble training."""
        from ml.models.ensemble import EnsembleDetector
        
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_val = pd.DataFrame(np.random.randn(20, 10))
        y_val = pd.Series(np.random.randint(0, 2, 20))
        
        ensemble = EnsembleDetector(model_dir=temp_model_dir)
        ensemble.train(X_train, y_train, X_val, y_val)
        
        assert ensemble.xgboost is not None
