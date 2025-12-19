"""
Ensemble Model for Network Intrusion Detection
Combines XGBoost, Autoencoder, and LSTM for robust detection
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EnsembleDetector:
    """
    Ensemble detector combining multiple models for robust intrusion detection.
    Implements voting, weighted averaging, and stacking strategies.
    """
    
    def __init__(
        self,
        models: Optional[Dict] = None,
        weights: Optional[Dict[str, float]] = None,
        strategy: str = 'weighted_average',
        threshold: float = 0.5
    ):
        """
        Initialize ensemble detector.
        
        Args:
            models: Dictionary of model instances {'name': model}
            weights: Model weights for ensemble {'name': weight}
            strategy: Ensemble strategy ('voting', 'weighted_average', 'stacking')
            threshold: Classification threshold
        """
        self.models = models or {}
        self.weights = weights or {}
        self.strategy = strategy
        self.threshold = threshold
        
        # Normalize weights if provided
        if self.weights:
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Meta-learner for stacking (if applicable)
        self.meta_learner = None
        
        # Performance tracking
        self.model_performance: Dict[str, Dict] = {}
        
        # Metadata
        self.metadata: Dict = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'model_type': 'ensemble_detector',
            'strategy': strategy
        }
        
        logger.info(f"Initialized EnsembleDetector with strategy: {strategy}")
    
    def add_model(self, name: str, model: object, weight: float = 1.0) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Model instance (must have predict and predict_proba methods)
            weight: Model weight
        """
        self.models[name] = model
        self.weights[name] = weight
        
        # Renormalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        
        logger.info(f"Added model '{name}' with weight {weight}")
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.weights[name]
            
            # Renormalize weights
            if self.weights:
                total = sum(self.weights.values())
                self.weights = {k: v / total for k, v in self.weights.items()}
            
            logger.info(f"Removed model '{name}'")
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
            **kwargs: Additional arguments passed to individual models
            
        Returns:
            Predicted labels
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")
        
        proba = self.predict_proba(X, **kwargs)
        
        if proba.ndim == 1:
            return (proba >= self.threshold).astype(int)
        else:
            return np.argmax(proba, axis=1)
    
    def predict_proba(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Get ensemble probability predictions.
        
        Args:
            X: Input features
            **kwargs: Additional arguments passed to individual models
            
        Returns:
            Probability predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble. Add models first.")
        
        if self.strategy == 'voting':
            return self._voting_predict(X, **kwargs)
        elif self.strategy == 'weighted_average':
            return self._weighted_average_predict(X, **kwargs)
        elif self.strategy == 'stacking':
            return self._stacking_predict(X, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _voting_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Hard voting ensemble."""
        predictions = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X, **kwargs.get(name, {}))
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
        
        if not predictions:
            raise ValueError("All model predictions failed")
        
        # Majority voting
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)
    
    def _weighted_average_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Weighted average of probabilities."""
        all_proba = []
        all_weights = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X, **kwargs.get(name, {}))
                
                # Ensure 2D array
                if proba.ndim == 1:
                    proba = np.column_stack([1 - proba, proba])
                
                all_proba.append(proba)
                all_weights.append(self.weights.get(name, 1.0))
                
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")
        
        if not all_proba:
            raise ValueError("All model predictions failed")
        
        # Weighted average
        all_proba = np.array(all_proba)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        weighted_proba = np.tensordot(all_weights, all_proba, axes=([0], [0]))
        
        return weighted_proba
    
    def _stacking_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Stacking with meta-learner."""
        if self.meta_learner is None:
            logger.warning("Meta-learner not trained. Falling back to weighted average.")
            return self._weighted_average_predict(X, **kwargs)
        
        # Get base model predictions
        meta_features = self._get_meta_features(X, **kwargs)
        
        # Meta-learner prediction
        return self.meta_learner.predict_proba(meta_features)
    
    def _get_meta_features(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Get meta-features from base models for stacking."""
        meta_features = []
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X, **kwargs.get(name, {}))
                
                if proba.ndim == 1:
                    meta_features.append(proba)
                else:
                    meta_features.append(proba[:, 1])  # Use positive class probability
                    
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                meta_features.append(np.zeros(len(X)))
        
        return np.column_stack(meta_features)
    
    def train_meta_learner(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta_model: str = 'logistic',
        **kwargs
    ) -> None:
        """
        Train meta-learner for stacking ensemble.
        
        Args:
            X: Training features
            y: Training labels
            meta_model: Type of meta-learner ('logistic', 'xgboost')
            **kwargs: Additional arguments for model prediction
        """
        # Get meta-features
        meta_features = self._get_meta_features(X, **kwargs)
        
        # Train meta-learner
        if meta_model == 'logistic':
            from sklearn.linear_model import LogisticRegression
            self.meta_learner = LogisticRegression(max_iter=1000)
        elif meta_model == 'xgboost':
            import xgboost as xgb
            self.meta_learner = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        else:
            raise ValueError(f"Unknown meta-model: {meta_model}")
        
        self.meta_learner.fit(meta_features, y)
        logger.info(f"Trained {meta_model} meta-learner")
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """
        Evaluate ensemble and individual models.
        
        Args:
            X: Test features
            y: True labels
            **kwargs: Additional arguments for model prediction
            
        Returns:
            Dictionary with ensemble and individual model metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        results = {'ensemble': {}, 'individual': {}}
        
        # Ensemble metrics
        predictions = self.predict(X, **kwargs)
        proba = self.predict_proba(X, **kwargs)
        
        results['ensemble'] = {
            'accuracy': float(accuracy_score(y, predictions)),
            'precision': float(precision_score(y, predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, predictions, average='weighted', zero_division=0))
        }
        
        # Add AUC if binary classification
        if proba.ndim == 2 and proba.shape[1] == 2:
            results['ensemble']['auc_roc'] = float(roc_auc_score(y, proba[:, 1]))
        
        # Individual model metrics
        for name, model in self.models.items():
            try:
                pred = model.predict(X, **kwargs.get(name, {}))
                
                results['individual'][name] = {
                    'accuracy': float(accuracy_score(y, pred)),
                    'precision': float(precision_score(y, pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y, pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y, pred, average='weighted', zero_division=0)),
                    'weight': self.weights.get(name, 1.0)
                }
                
            except Exception as e:
                logger.warning(f"Evaluation failed for {name}: {e}")
        
        self.model_performance = results
        return results
    
    def optimize_weights(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'f1',
        **kwargs
    ) -> Dict[str, float]:
        """
        Optimize model weights using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
            **kwargs: Additional arguments for model prediction
            
        Returns:
            Optimized weights dictionary
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from scipy.optimize import minimize
        
        metric_funcs = {
            'accuracy': lambda y, p: accuracy_score(y, (p >= 0.5).astype(int)),
            'f1': lambda y, p: f1_score(y, (p >= 0.5).astype(int), average='weighted', zero_division=0),
            'precision': lambda y, p: precision_score(y, (p >= 0.5).astype(int), average='weighted', zero_division=0),
            'recall': lambda y, p: recall_score(y, (p >= 0.5).astype(int), average='weighted', zero_division=0)
        }
        
        metric_func = metric_funcs.get(metric, metric_funcs['f1'])
        
        # Get individual probabilities
        model_proba = {}
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X_val, **kwargs.get(name, {}))
                if proba.ndim == 2:
                    proba = proba[:, 1]
                model_proba[name] = proba
            except:
                pass
        
        if not model_proba:
            return self.weights
        
        model_names = list(model_proba.keys())
        proba_matrix = np.column_stack([model_proba[n] for n in model_names])
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_proba = np.dot(proba_matrix, weights)
            return -metric_func(y_val, ensemble_proba)  # Negative for minimization
        
        # Initial weights
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, all non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimized_weights = result.x / result.x.sum()
        self.weights = {name: float(w) for name, w in zip(model_names, optimized_weights)}
        
        logger.info(f"Optimized weights: {self.weights}")
        return self.weights
    
    def get_model_contributions(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to predictions.
        
        Args:
            X: Input features
            **kwargs: Additional arguments for model prediction
            
        Returns:
            Dictionary of model contributions
        """
        contributions = {}
        
        for name, model in self.models.items():
            try:
                proba = model.predict_proba(X, **kwargs.get(name, {}))
                if proba.ndim == 2:
                    proba = proba[:, 1]
                
                weighted = proba * self.weights.get(name, 1.0)
                contributions[name] = weighted
                
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        return contributions
    
    def save(self, path: str, save_models: bool = False) -> None:
        """
        Save ensemble configuration.
        
        Args:
            path: Path to save configuration
            save_models: Whether to also save individual models
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        config = {
            'weights': self.weights,
            'strategy': self.strategy,
            'threshold': self.threshold,
            'model_names': list(self.models.keys()),
            'model_performance': self.model_performance,
            'metadata': self.metadata
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save individual models if requested
        if save_models:
            models_dir = os.path.join(os.path.dirname(path), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            for name, model in self.models.items():
                if hasattr(model, 'save'):
                    model.save(os.path.join(models_dir, f'{name}.pkl'))
        
        logger.info(f"Saved ensemble configuration to {path}")
    
    @classmethod
    def load(cls, path: str, models: Optional[Dict] = None) -> 'EnsembleDetector':
        """
        Load ensemble from configuration.
        
        Args:
            path: Path to configuration file
            models: Dictionary of model instances to use
            
        Returns:
            EnsembleDetector instance
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        ensemble = cls(
            models=models or {},
            weights=config.get('weights', {}),
            strategy=config.get('strategy', 'weighted_average'),
            threshold=config.get('threshold', 0.5)
        )
        
        ensemble.model_performance = config.get('model_performance', {})
        ensemble.metadata = config.get('metadata', ensemble.metadata)
        
        logger.info(f"Loaded ensemble configuration from {path}")
        return ensemble


def create_ensemble(
    xgboost_model=None,
    autoencoder_model=None,
    lstm_model=None,
    weights: Optional[Dict[str, float]] = None,
    strategy: str = 'weighted_average'
) -> EnsembleDetector:
    """
    Factory function to create an EnsembleDetector with common models.
    
    Args:
        xgboost_model: XGBoost classifier instance
        autoencoder_model: Autoencoder instance
        lstm_model: LSTM detector instance
        weights: Model weights
        strategy: Ensemble strategy
        
    Returns:
        Configured EnsembleDetector
    """
    models = {}
    default_weights = {}
    
    if xgboost_model is not None:
        models['xgboost'] = xgboost_model
        default_weights['xgboost'] = 0.4
    
    if autoencoder_model is not None:
        models['autoencoder'] = autoencoder_model
        default_weights['autoencoder'] = 0.3
    
    if lstm_model is not None:
        models['lstm'] = lstm_model
        default_weights['lstm'] = 0.3
    
    return EnsembleDetector(
        models=models,
        weights=weights or default_weights,
        strategy=strategy
    )
