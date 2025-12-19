"""
AI-NIDS Training Utilities
==========================
Helper functions for model training and evaluation
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training and evaluation utility.
    Handles training, validation, and model persistence.
    """
    
    def __init__(
        self,
        model_dir: str = 'models',
        random_state: int = 42
    ):
        """
        Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Training history
        self.history: Dict[str, List[Dict]] = {}
    
    def prepare_data(
        self,
        data_path: str,
        target_column: str = 'label',
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load and split data for training.
        
        Args:
            data_path: Path to CSV data file
            target_column: Name of target column
            test_size: Proportion for test set
            val_size: Proportion for validation set
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Second split: train and val
        val_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        params: Optional[Dict] = None
    ) -> Any:
        """Train XGBoost classifier"""
        from ml.models.xgboost_classifier import XGBoostClassifier
        
        logger.info("Training XGBoost classifier...")
        model = XGBoostClassifier(params=params)
        
        model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val
        )
        
        # Save model
        model_path = self.model_dir / 'xgboost_model.json'
        model.save(str(model_path))
        logger.info(f"Saved XGBoost model to {model_path}")
        
        return model
    
    def train_autoencoder(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        epochs: int = 100,
        batch_size: int = 256
    ) -> Any:
        """Train Autoencoder for anomaly detection"""
        from ml.models.autoencoder import AnomalyAutoencoder
        
        logger.info("Training Autoencoder...")
        
        input_dim = X_train.shape[1]
        model = AnomalyAutoencoder(input_dim=input_dim)
        
        history = model.train(
            X_train.values,
            X_val=X_val.values if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Save model
        model_path = self.model_dir / 'autoencoder.pt'
        model.save(str(model_path))
        logger.info(f"Saved Autoencoder model to {model_path}")
        
        return model, history
    
    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: pd.Series,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[pd.Series] = None,
        sequence_length: int = 10,
        epochs: int = 50
    ) -> Any:
        """Train LSTM detector"""
        from ml.models.lstm_detector import LSTMDetector
        
        logger.info("Training LSTM detector...")
        
        n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
        model = LSTMDetector(
            input_size=n_features,
            sequence_length=sequence_length
        )
        
        history = model.train(
            X_train, y_train.values,
            X_val=X_val, y_val=y_val.values if y_val is not None else None,
            epochs=epochs
        )
        
        # Save model
        model_path = self.model_dir / 'lstm_detector.pt'
        model.save(str(model_path))
        logger.info(f"Saved LSTM model to {model_path}")
        
        return model, history
    
    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train full ensemble"""
        from ml.models.ensemble import EnsembleDetector
        
        logger.info("Training Ensemble detector...")
        
        ensemble = EnsembleDetector(model_dir=str(self.model_dir))
        ensemble.train(X_train, y_train, X_val, y_val)
        ensemble.save()
        
        logger.info("Ensemble training complete")
        return ensemble
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = 'model'
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            raise ValueError("Model must have predict method")
        
        # Get probabilities if available
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Binary classification metrics
        if len(np.unique(y_test)) == 2:
            metrics['precision_binary'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall_binary'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1_binary'] = f1_score(y_test, y_pred, zero_division=0)
            
            if y_prob is not None:
                if len(y_prob.shape) > 1:
                    y_prob_positive = y_prob[:, 1]
                else:
                    y_prob_positive = y_prob
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob_positive)
        
        # Log results
        logger.info(f"\n{model_name} Evaluation Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Store in history
        if model_name not in self.history:
            self.history[model_name] = []
        self.history[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = 'Confusion Matrix',
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        title: str = 'ROC Curve',
        save_path: Optional[str] = None
    ):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved ROC curve to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_prob: np.ndarray,
        title: str = 'Precision-Recall Curve',
        save_path: Optional[str] = None
    ):
        """Plot precision-recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved PR curve to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
        title: str = 'Feature Importance',
        save_path: Optional[str] = None
    ):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        else:
            logger.warning("Model doesn't have feature importance")
            return
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(indices)),
            importances[indices],
            align='center'
        )
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved feature importance to {save_path}")
        
        plt.show()
    
    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """Perform cross-validation"""
        logger.info(f"Running {cv}-fold cross-validation...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        results = {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
        
        logger.info(f"CV F1 Score: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def save_training_report(
        self,
        report_path: Optional[str] = None
    ):
        """Save training history report"""
        if report_path is None:
            report_path = self.model_dir / 'training_report.json'
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'models': self.history
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved training report to {report_path}")


def load_cicids2017(data_dir: str) -> pd.DataFrame:
    """
    Load and combine CICIDS2017 dataset files.
    
    Args:
        data_dir: Directory containing CICIDS2017 CSV files
        
    Returns:
        Combined DataFrame
    """
    logger.info(f"Loading CICIDS2017 from {data_dir}")
    
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob('*.csv'))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file_path in all_files:
        logger.info(f"  Loading {file_path.name}")
        df = pd.read_csv(file_path, low_memory=False)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total samples")
    
    return combined


def load_unsw_nb15(data_dir: str) -> pd.DataFrame:
    """
    Load and combine UNSW-NB15 dataset files.
    
    Args:
        data_dir: Directory containing UNSW-NB15 CSV files
        
    Returns:
        Combined DataFrame
    """
    logger.info(f"Loading UNSW-NB15 from {data_dir}")
    
    data_dir = Path(data_dir)
    
    # UNSW-NB15 training and testing sets
    train_files = list(data_dir.glob('UNSW_NB15_training*.csv'))
    test_files = list(data_dir.glob('UNSW_NB15_testing*.csv'))
    
    all_files = train_files + test_files
    
    if not all_files:
        # Try alternative naming
        all_files = list(data_dir.glob('*.csv'))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file_path in all_files:
        logger.info(f"  Loading {file_path.name}")
        df = pd.read_csv(file_path, low_memory=False)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} total samples")
    
    return combined
