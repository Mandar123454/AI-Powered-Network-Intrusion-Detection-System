"""
XGBoost Classifier for Network Intrusion Detection
High-performance gradient boosting model with built-in feature importance
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from typing import Dict, List, Optional, Tuple, Union
import pickle
import os
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """
    XGBoost-based network intrusion detection classifier.
    Optimized for both binary and multi-class classification.
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        min_child_weight: int = 3,
        gamma: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        objective: str = 'binary:logistic',
        eval_metric: str = 'auc',
        use_gpu: bool = False,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize XGBoost classifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_child_weight: Minimum sum of instance weight
            gamma: Minimum loss reduction for split
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            scale_pos_weight: Balance positive/negative weights
            objective: Learning objective
            eval_metric: Evaluation metric
            use_gpu: Whether to use GPU acceleration
            random_state: Random seed
            n_jobs: Number of parallel threads
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'objective': objective,
            'eval_metric': eval_metric,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'tree_method': 'gpu_hist' if use_gpu else 'hist',
            'verbosity': 0
        }
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.training_history: Dict = {}
        self.metadata: Dict = {
            'created_at': None,
            'trained_at': None,
            'version': '1.0.0',
            'model_type': 'xgboost_classifier'
        }
        
    def build_model(self, n_classes: int = 2) -> xgb.XGBClassifier:
        """
        Build XGBoost model.
        
        Args:
            n_classes: Number of classes
            
        Returns:
            XGBoost classifier instance
        """
        params = self.params.copy()
        
        # Adjust for multi-class
        if n_classes > 2:
            params['objective'] = 'multi:softprob'
            params['num_class'] = n_classes
            params['eval_metric'] = 'mlogloss'
        
        self.model = xgb.XGBClassifier(**params)
        self.metadata['created_at'] = datetime.now().isoformat()
        
        logger.info(f"Built XGBoost model with params: {params}")
        return self.model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 20,
        verbose: bool = True
    ) -> Dict:
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Names of features
            early_stopping_rounds: Rounds without improvement before stopping
            verbose: Whether to print training progress
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Training XGBoost on {X_train.shape[0]} samples...")
        
        # Determine number of classes
        n_classes = len(np.unique(y_train))
        
        # Build model if not already built
        if self.model is None:
            self.build_model(n_classes)
        
        # Store feature names
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Training metrics
        train_preds = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_preds)
        
        # Validation metrics
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_preds)
            val_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        
        self.training_history = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'n_estimators': self.model.n_estimators,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.model.n_estimators
        }
        
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        logger.info(f"Training complete. Train F1: {train_metrics['f1']:.4f}")
        if val_metrics:
            logger.info(f"Validation F1: {val_metrics['val_f1']:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        detailed: bool = True
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            detailed: Whether to include detailed report
            
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        metrics = self._calculate_metrics(y, predictions)
        
        # Add AUC-ROC
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            metrics['auc_roc'] = roc_auc_score(y, probabilities[:, 1])
        elif probabilities.ndim == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y, probabilities, multi_class='ovr')
            except:
                pass
        
        if detailed:
            metrics['confusion_matrix'] = confusion_matrix(y, predictions).tolist()
            metrics['classification_report'] = classification_report(y, predictions, output_dict=True)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate classification metrics."""
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0))
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save model state
        state = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names,
            'feature_importances': self.feature_importances_,
            'training_history': self.training_history,
            'metadata': self.metadata
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved XGBoost model to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'XGBoostClassifier':
        """
        Load model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded XGBoostClassifier instance
        """
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        classifier = cls()
        classifier.model = state['model']
        classifier.params = state['params']
        classifier.feature_names = state['feature_names']
        classifier.feature_importances_ = state['feature_importances']
        classifier.training_history = state['training_history']
        classifier.metadata = state['metadata']
        
        logger.info(f"Loaded XGBoost model from {path}")
        return classifier
    
    def export_to_json(self, path: str) -> None:
        """Export model metadata to JSON."""
        export_data = {
            'metadata': self.metadata,
            'params': {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                      for k, v in self.params.items()},
            'training_history': self.training_history,
            'feature_names': self.feature_names[:20] if self.feature_names else None,
            'top_features': self.get_feature_importance(10).to_dict('records') if self.feature_importances_ is not None else None
        }
        
        with open(path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported model metadata to {path}")


def create_xgboost_classifier(**kwargs) -> XGBoostClassifier:
    """Factory function to create XGBoostClassifier."""
    return XGBoostClassifier(**kwargs)
