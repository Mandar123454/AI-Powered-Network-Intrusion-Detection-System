"""
Adaptive Ensemble with Dynamic LSTM-Controlled Weights
======================================================
Next-generation ensemble that adapts model weights in real-time based on:
- Current network state and traffic patterns
- Time of day / day of week patterns
- Historical model performance
- Baseline deviation levels
- Threat intelligence context

This represents state-of-the-art ensemble learning for security:
- No static weights - weights change every prediction
- Context-aware model selection
- Automatic model performance tracking
- Self-optimizing based on feedback loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import json
import threading
from enum import Enum, auto

logger = logging.getLogger(__name__)


class NetworkState(Enum):
    """Network state categories for context-aware weighting."""
    NORMAL = auto()
    HIGH_TRAFFIC = auto()
    LOW_TRAFFIC = auto()
    ATTACK_SUSPECTED = auto()
    ATTACK_CONFIRMED = auto()
    MAINTENANCE = auto()
    UNKNOWN = auto()


@dataclass
class ContextFeatures:
    """Features describing current network context."""
    # Temporal features
    hour_of_day: int = 0  # 0-23
    day_of_week: int = 0  # 0-6 (Monday=0)
    is_weekend: bool = False
    is_business_hours: bool = False
    
    # Traffic features
    current_traffic_rate: float = 0.0  # packets/sec
    traffic_vs_baseline: float = 1.0  # ratio to baseline
    unique_sources: int = 0
    unique_destinations: int = 0
    
    # State features
    network_state: NetworkState = NetworkState.NORMAL
    baseline_deviation: float = 0.0  # how far from normal
    alert_count_1h: int = 0  # alerts in last hour
    alert_count_24h: int = 0  # alerts in last 24 hours
    
    # Threat intel features
    known_malicious_ips: int = 0
    ioc_hits: int = 0
    threat_level: float = 0.0  # 0-1 scale
    
    # Model performance features
    model_accuracies: Dict[str, float] = field(default_factory=dict)
    model_recent_errors: Dict[str, int] = field(default_factory=dict)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert context to feature tensor."""
        # Temporal encoding (cyclical)
        hour_sin = np.sin(2 * np.pi * self.hour_of_day / 24)
        hour_cos = np.cos(2 * np.pi * self.hour_of_day / 24)
        day_sin = np.sin(2 * np.pi * self.day_of_week / 7)
        day_cos = np.cos(2 * np.pi * self.day_of_week / 7)
        
        features = [
            hour_sin, hour_cos, day_sin, day_cos,
            float(self.is_weekend),
            float(self.is_business_hours),
            np.log1p(self.current_traffic_rate),
            self.traffic_vs_baseline,
            np.log1p(self.unique_sources),
            np.log1p(self.unique_destinations),
            self.network_state.value / 7.0,  # Normalize state
            self.baseline_deviation,
            np.log1p(self.alert_count_1h),
            np.log1p(self.alert_count_24h),
            np.log1p(self.known_malicious_ips),
            np.log1p(self.ioc_hits),
            self.threat_level
        ]
        
        return torch.tensor(features, dtype=torch.float32)


class LSTMWeightController(nn.Module):
    """
    LSTM-based controller that generates dynamic model weights
    based on current network context and historical patterns.
    """
    
    def __init__(
        self,
        context_dim: int = 17,
        hidden_dim: int = 64,
        num_models: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 20
    ):
        super().__init__()
        
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        self.sequence_length = sequence_length
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention over history
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_models)
        )
        
        # Temperature for softmax sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
        # History buffer
        self.register_buffer('hidden', None)
        self.register_buffer('cell', None)
        
    def reset_state(self):
        """Reset LSTM hidden state."""
        self.hidden = None
        self.cell = None
    
    def forward(
        self,
        context: torch.Tensor,
        history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate dynamic weights based on context.
        
        Args:
            context: Current context features [batch, context_dim]
            history: Optional historical context [batch, seq_len, context_dim]
            
        Returns:
            Tuple of (weights [batch, num_models], attention_weights)
        """
        batch_size = context.size(0)
        
        # Encode current context
        current_encoded = self.context_encoder(context)  # [batch, hidden]
        
        if history is not None:
            # Encode history
            history_encoded = self.context_encoder(
                history.view(-1, self.context_dim)
            ).view(batch_size, -1, self.hidden_dim)
            
            # LSTM over history
            lstm_out, (h_n, c_n) = self.lstm(history_encoded)
            
            # Attention: query with current, attend to history
            query = current_encoded.unsqueeze(1)  # [batch, 1, hidden]
            attended, attn_weights = self.attention(query, lstm_out, lstm_out)
            attended = attended.squeeze(1)  # [batch, hidden]
            
        else:
            # No history, use only current
            attended = current_encoded
            attn_weights = None
        
        # Combine current and attended history
        combined = torch.cat([current_encoded, attended], dim=-1)
        
        # Generate weights
        raw_weights = self.weight_generator(combined)
        
        # Apply temperature-controlled softmax
        weights = F.softmax(raw_weights / self.temperature, dim=-1)
        
        return weights, attn_weights
    
    def get_weight_explanations(
        self,
        weights: torch.Tensor,
        model_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Generate explanations for weight assignments."""
        weights_np = weights.detach().cpu().numpy()
        
        explanations = {}
        for i, name in enumerate(model_names):
            w = float(weights_np[0, i]) if weights_np.ndim > 1 else float(weights_np[i])
            
            # Determine contribution level
            if w > 0.4:
                contribution = "dominant"
            elif w > 0.2:
                contribution = "significant"
            elif w > 0.1:
                contribution = "moderate"
            else:
                contribution = "minimal"
            
            explanations[name] = {
                'weight': w,
                'contribution': contribution,
                'percentage': f"{w * 100:.1f}%"
            }
        
        return explanations


class ModelPerformanceTracker:
    """Tracks model performance over time for adaptive weighting."""
    
    def __init__(
        self,
        model_names: List[str],
        window_size: int = 1000,
        decay_factor: float = 0.99
    ):
        self.model_names = model_names
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        # Performance tracking
        self.predictions: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in model_names
        }
        self.ground_truth: deque = deque(maxlen=window_size)
        
        # Cumulative scores with exponential decay
        self.accuracy_scores: Dict[str, float] = {name: 0.5 for name in model_names}
        self.precision_scores: Dict[str, float] = {name: 0.5 for name in model_names}
        self.recall_scores: Dict[str, float] = {name: 0.5 for name in model_names}
        
        # Error tracking
        self.recent_errors: Dict[str, int] = {name: 0 for name in model_names}
        self.false_positives: Dict[str, int] = {name: 0 for name in model_names}
        self.false_negatives: Dict[str, int] = {name: 0 for name in model_names}
        
        self.lock = threading.Lock()
    
    def record_prediction(
        self,
        model_name: str,
        prediction: int,
        probability: float,
        ground_truth: Optional[int] = None
    ):
        """Record a prediction for tracking."""
        with self.lock:
            if model_name in self.predictions:
                self.predictions[model_name].append({
                    'pred': prediction,
                    'prob': probability,
                    'timestamp': datetime.now()
                })
            
            if ground_truth is not None:
                self.ground_truth.append({
                    'label': ground_truth,
                    'timestamp': datetime.now()
                })
    
    def update_with_feedback(
        self,
        model_name: str,
        was_correct: bool,
        was_false_positive: bool = False,
        was_false_negative: bool = False
    ):
        """Update scores with ground truth feedback."""
        with self.lock:
            # Update accuracy with exponential moving average
            self.accuracy_scores[model_name] = (
                self.decay_factor * self.accuracy_scores[model_name] +
                (1 - self.decay_factor) * (1.0 if was_correct else 0.0)
            )
            
            if not was_correct:
                self.recent_errors[model_name] += 1
            
            if was_false_positive:
                self.false_positives[model_name] += 1
                self.precision_scores[model_name] = (
                    self.decay_factor * self.precision_scores[model_name] +
                    (1 - self.decay_factor) * 0.0
                )
            elif was_false_negative:
                self.false_negatives[model_name] += 1
                self.recall_scores[model_name] = (
                    self.decay_factor * self.recall_scores[model_name] +
                    (1 - self.decay_factor) * 0.0
                )
            else:
                # Correct prediction
                self.precision_scores[model_name] = (
                    self.decay_factor * self.precision_scores[model_name] +
                    (1 - self.decay_factor) * 1.0
                )
                self.recall_scores[model_name] = (
                    self.decay_factor * self.recall_scores[model_name] +
                    (1 - self.decay_factor) * 1.0
                )
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get current performance summary for all models."""
        with self.lock:
            return {
                name: {
                    'accuracy': self.accuracy_scores[name],
                    'precision': self.precision_scores[name],
                    'recall': self.recall_scores[name],
                    'f1': 2 * self.precision_scores[name] * self.recall_scores[name] /
                          (self.precision_scores[name] + self.recall_scores[name] + 1e-10),
                    'recent_errors': self.recent_errors[name],
                    'false_positives': self.false_positives[name],
                    'false_negatives': self.false_negatives[name]
                }
                for name in self.model_names
            }
    
    def get_reliability_weights(self) -> Dict[str, float]:
        """Get reliability-based weight suggestions."""
        summary = self.get_performance_summary()
        
        # Combine accuracy, precision, recall with penalties for errors
        weights = {}
        for name, metrics in summary.items():
            # Base weight from F1
            base = metrics['f1']
            
            # Penalty for recent errors
            error_penalty = min(metrics['recent_errors'] / 10, 0.5)
            
            weights[name] = max(0.01, base - error_penalty)
        
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble with LSTM-controlled dynamic weights.
    Combines multiple detection models with context-aware weighting.
    """
    
    # Model registry
    MODEL_TYPES = [
        'xgboost',
        'autoencoder', 
        'lstm',
        'gnn',
        'temporal'
    ]
    
    def __init__(
        self,
        model_names: List[str] = None,
        hidden_dim: int = 64,
        context_dim: int = 17,
        min_weight: float = 0.05,
        device: str = 'cpu'
    ):
        super().__init__()
        
        if model_names is None:
            model_names = self.MODEL_TYPES
        
        self.model_names = model_names
        self.num_models = len(model_names)
        self.min_weight = min_weight
        self.device = device
        
        # LSTM weight controller
        self.weight_controller = LSTMWeightController(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            num_models=self.num_models
        )
        
        # Performance tracker
        self.performance_tracker = ModelPerformanceTracker(model_names)
        
        # Context history buffer
        self.context_history: deque = deque(maxlen=50)
        
        # Model references (set externally)
        self.models: Dict[str, nn.Module] = {}
        
        # Prediction history for analysis
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'weight_distributions': [],
            'context_states': defaultdict(int)
        }
    
    def register_model(self, name: str, model: nn.Module):
        """Register a model with the ensemble."""
        if name not in self.model_names:
            logger.warning(f"Model {name} not in model_names, adding it")
            self.model_names.append(name)
            self.num_models = len(self.model_names)
        
        self.models[name] = model
        logger.info(f"Registered model: {name}")
    
    def _build_context(
        self,
        traffic_features: Optional[Dict[str, float]] = None,
        threat_intel: Optional[Dict[str, Any]] = None,
        baseline_deviation: float = 0.0
    ) -> ContextFeatures:
        """Build context features from current state."""
        now = datetime.now()
        
        context = ContextFeatures(
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            is_weekend=now.weekday() >= 5,
            is_business_hours=9 <= now.hour <= 17 and now.weekday() < 5,
            baseline_deviation=baseline_deviation
        )
        
        if traffic_features:
            context.current_traffic_rate = traffic_features.get('rate', 0)
            context.traffic_vs_baseline = traffic_features.get('vs_baseline', 1.0)
            context.unique_sources = traffic_features.get('unique_sources', 0)
            context.unique_destinations = traffic_features.get('unique_destinations', 0)
        
        if threat_intel:
            context.known_malicious_ips = threat_intel.get('malicious_ips', 0)
            context.ioc_hits = threat_intel.get('ioc_hits', 0)
            context.threat_level = threat_intel.get('threat_level', 0.0)
        
        # Add model performance
        perf_summary = self.performance_tracker.get_performance_summary()
        context.model_accuracies = {
            name: data['accuracy'] for name, data in perf_summary.items()
        }
        context.model_recent_errors = {
            name: data['recent_errors'] for name, data in perf_summary.items()
        }
        
        return context
    
    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        context: Optional[ContextFeatures] = None,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        Combine model outputs with adaptive weights.
        
        Args:
            model_outputs: Dict mapping model names to their probability outputs
            context: Current context features
            return_details: Whether to return detailed weight information
            
        Returns:
            Dictionary with combined predictions and optional details
        """
        # Build context if not provided
        if context is None:
            context = self._build_context()
        
        context_tensor = context.to_tensor().unsqueeze(0).to(self.device)
        
        # Get history tensor if available
        history_tensor = None
        if len(self.context_history) >= 5:
            history_list = [c.to_tensor() for c in self.context_history]
            history_tensor = torch.stack(history_list).unsqueeze(0).to(self.device)
        
        # Generate adaptive weights
        weights, attn_weights = self.weight_controller(context_tensor, history_tensor)
        
        # Ensure minimum weight for all models
        weights = weights + self.min_weight
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Combine model outputs
        combined_proba = torch.zeros_like(next(iter(model_outputs.values())))
        
        for i, name in enumerate(self.model_names):
            if name in model_outputs:
                w = weights[0, i]
                combined_proba = combined_proba + w * model_outputs[name]
        
        # Record context for history
        self.context_history.append(context)
        
        # Update statistics
        self.stats['total_predictions'] += 1
        self.stats['weight_distributions'].append(
            weights.detach().cpu().numpy().tolist()
        )
        self.stats['context_states'][context.network_state.name] += 1
        
        result = {
            'probabilities': combined_proba,
            'predictions': (combined_proba > 0.5).long() if combined_proba.ndim == 1 
                          else combined_proba.argmax(dim=-1),
            'weights': weights.detach()
        }
        
        if return_details:
            result['weight_explanations'] = self.weight_controller.get_weight_explanations(
                weights, self.model_names
            )
            result['context'] = context
            result['attention_weights'] = attn_weights
            result['model_performance'] = self.performance_tracker.get_performance_summary()
        
        return result
    
    def predict_with_all_models(
        self,
        X: torch.Tensor,
        context: Optional[ContextFeatures] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Run all registered models and combine with adaptive weights.
        
        Args:
            X: Input features
            context: Optional context features
            **model_kwargs: Additional kwargs for specific models
            
        Returns:
            Combined prediction with detailed breakdown
        """
        model_outputs = {}
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                with torch.no_grad():
                    kwargs = model_kwargs.get(name, {})
                    
                    # Handle different model types
                    if hasattr(model, 'predict_proba'):
                        # Sklearn-style model
                        output = model.predict_proba(X.cpu().numpy())
                        output = torch.tensor(output, dtype=torch.float32)
                        if output.ndim == 2:
                            output = output[:, 1]  # Get positive class probability
                    elif hasattr(model, 'forward'):
                        # PyTorch model
                        output = model(X, **kwargs)
                        if isinstance(output, dict):
                            output = output.get('probabilities', output.get('probs', output.get('logits')))
                            if 'logits' in str(type(output)):
                                output = F.softmax(output, dim=-1)
                    else:
                        continue
                    
                    model_outputs[name] = output.to(self.device)
                    individual_predictions[name] = {
                        'probabilities': output.detach().cpu(),
                        'prediction': (output > 0.5).long() if output.ndim == 1
                                     else output.argmax(dim=-1)
                    }
                    
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        
        if not model_outputs:
            raise ValueError("All models failed to produce output")
        
        # Get adaptive combination
        result = self.forward(model_outputs, context, return_details=True)
        result['individual_predictions'] = individual_predictions
        
        return result
    
    def update_from_feedback(
        self,
        prediction_id: str,
        ground_truth: int,
        individual_predictions: Dict[str, int]
    ):
        """Update model performance from ground truth feedback."""
        for name, pred in individual_predictions.items():
            was_correct = pred == ground_truth
            was_false_positive = pred == 1 and ground_truth == 0
            was_false_negative = pred == 0 and ground_truth == 1
            
            self.performance_tracker.update_with_feedback(
                name, was_correct, was_false_positive, was_false_negative
            )
        
        logger.debug(f"Updated performance from feedback for prediction {prediction_id}")
    
    def get_weight_analysis(self) -> Dict[str, Any]:
        """Analyze weight distribution patterns."""
        if not self.stats['weight_distributions']:
            return {'status': 'no_data'}
        
        weights_array = np.array(self.stats['weight_distributions'][-100:])
        
        analysis = {
            'num_predictions': self.stats['total_predictions'],
            'model_statistics': {}
        }
        
        for i, name in enumerate(self.model_names):
            model_weights = weights_array[:, 0, i] if weights_array.ndim == 3 else weights_array[:, i]
            analysis['model_statistics'][name] = {
                'mean_weight': float(np.mean(model_weights)),
                'std_weight': float(np.std(model_weights)),
                'min_weight': float(np.min(model_weights)),
                'max_weight': float(np.max(model_weights)),
                'current_weight': float(model_weights[-1]) if len(model_weights) > 0 else 0
            }
        
        # Context state distribution
        total_states = sum(self.stats['context_states'].values())
        analysis['context_distribution'] = {
            state: count / total_states
            for state, count in self.stats['context_states'].items()
        }
        
        return analysis
    
    def explain_current_weights(self) -> str:
        """Generate human-readable explanation of current weight selection."""
        if not self.stats['weight_distributions']:
            return "No predictions made yet."
        
        current_weights = self.stats['weight_distributions'][-1]
        if isinstance(current_weights[0], list):
            current_weights = current_weights[0]
        
        sorted_models = sorted(
            zip(self.model_names, current_weights),
            key=lambda x: x[1],
            reverse=True
        )
        
        explanation_parts = [
            f"Current ensemble configuration:",
            f"- Primary model: {sorted_models[0][0]} ({sorted_models[0][1]:.1%})"
        ]
        
        if len(sorted_models) > 1:
            secondary = [f"{m[0]} ({m[1]:.1%})" for m in sorted_models[1:3]]
            explanation_parts.append(f"- Secondary: {', '.join(secondary)}")
        
        # Add context reasoning
        if self.context_history:
            last_context = self.context_history[-1]
            if last_context.baseline_deviation > 0.5:
                explanation_parts.append("- Elevated due to baseline deviation")
            if last_context.threat_level > 0.5:
                explanation_parts.append("- Heightened due to threat intelligence")
            if not last_context.is_business_hours:
                explanation_parts.append("- Off-hours weighting applied")
        
        return "\n".join(explanation_parts)
    
    def save_state(self, path: str):
        """Save ensemble state including weight controller."""
        state = {
            'weight_controller': self.weight_controller.state_dict(),
            'model_names': self.model_names,
            'stats': {
                'total_predictions': self.stats['total_predictions'],
                'context_states': dict(self.stats['context_states'])
            },
            'performance': self.performance_tracker.get_performance_summary()
        }
        torch.save(state, path)
        logger.info(f"Saved adaptive ensemble state to {path}")
    
    def load_state(self, path: str):
        """Load ensemble state."""
        state = torch.load(path, map_location=self.device)
        self.weight_controller.load_state_dict(state['weight_controller'])
        self.model_names = state.get('model_names', self.model_names)
        self.stats['total_predictions'] = state.get('stats', {}).get('total_predictions', 0)
        logger.info(f"Loaded adaptive ensemble state from {path}")


class AdaptiveEnsembleTrainer:
    """Trainer for the adaptive ensemble weight controller."""
    
    def __init__(
        self,
        ensemble: AdaptiveEnsemble,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        self.ensemble = ensemble
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            ensemble.weight_controller.parameters(),
            lr=learning_rate
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.history = {
            'loss': [],
            'accuracy': []
        }
    
    def train_step(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        context: ContextFeatures
    ) -> Dict[str, float]:
        """Single training step for the weight controller."""
        self.ensemble.train()
        self.optimizer.zero_grad()
        
        # Get adaptive predictions
        result = self.ensemble.forward(
            {k: v.to(self.device) for k, v in model_outputs.items()},
            context
        )
        
        proba = result['probabilities']
        labels = labels.to(self.device)
        
        # Loss: combination of cross-entropy and diversity regularization
        if proba.ndim == 1 or proba.size(-1) == 1:
            proba = proba.view(-1)
            ce_loss = F.binary_cross_entropy(proba, labels.float())
        else:
            ce_loss = F.cross_entropy(proba, labels)
        
        # Diversity loss: penalize all weights going to one model
        weights = result['weights']
        entropy = -torch.sum(weights * torch.log(weights + 1e-10), dim=-1)
        diversity_loss = -entropy.mean()  # Maximize entropy = diversify weights
        
        total_loss = ce_loss + 0.1 * diversity_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute accuracy
        if proba.ndim == 1:
            preds = (proba > 0.5).long()
        else:
            preds = proba.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        
        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'accuracy': accuracy
        }
    
    def train_epoch(
        self,
        data_loader,
        model_predict_fn: Callable
    ) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0
        total_acc = 0
        n_batches = 0
        
        for batch in data_loader:
            X, labels, context = batch
            
            # Get model outputs
            model_outputs = model_predict_fn(X)
            
            metrics = self.train_step(model_outputs, labels, context)
            
            total_loss += metrics['loss']
            total_acc += metrics['accuracy']
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        
        self.history['loss'].append(avg_loss)
        self.history['accuracy'].append(avg_acc)
        
        self.scheduler.step(avg_loss)
        
        return {'loss': avg_loss, 'accuracy': avg_acc}


def create_adaptive_ensemble(
    model_names: List[str] = None,
    pretrained_path: Optional[str] = None,
    device: str = 'cpu'
) -> AdaptiveEnsemble:
    """
    Factory function to create an adaptive ensemble.
    
    Args:
        model_names: List of model names to include
        pretrained_path: Path to pretrained weights
        device: Device to use
        
    Returns:
        Configured AdaptiveEnsemble
    """
    ensemble = AdaptiveEnsemble(
        model_names=model_names,
        device=device
    )
    
    if pretrained_path:
        ensemble.load_state(pretrained_path)
    
    return ensemble.to(device)


if __name__ == "__main__":
    print("Adaptive Ensemble Demo")
    print("=" * 50)
    
    # Create ensemble
    ensemble = AdaptiveEnsemble(
        model_names=['xgboost', 'autoencoder', 'lstm', 'gnn', 'temporal'],
        device='cpu'
    )
    
    print(f"Ensemble models: {ensemble.model_names}")
    print(f"Weight controller parameters: {sum(p.numel() for p in ensemble.weight_controller.parameters()):,}")
    
    # Simulate model outputs
    batch_size = 10
    model_outputs = {
        'xgboost': torch.rand(batch_size),
        'autoencoder': torch.rand(batch_size),
        'lstm': torch.rand(batch_size),
        'gnn': torch.rand(batch_size),
        'temporal': torch.rand(batch_size)
    }
    
    # Create context
    context = ContextFeatures(
        hour_of_day=14,
        day_of_week=2,
        is_business_hours=True,
        current_traffic_rate=1500,
        baseline_deviation=0.3,
        threat_level=0.2
    )
    
    print(f"\nContext: Business hours, moderate traffic, low threat")
    
    # Get adaptive prediction
    result = ensemble.forward(model_outputs, context, return_details=True)
    
    print(f"\nWeight Distribution:")
    for name, info in result['weight_explanations'].items():
        print(f"  {name}: {info['percentage']} ({info['contribution']})")
    
    print(f"\nCombined probabilities shape: {result['probabilities'].shape}")
    print(f"Predictions: {result['predictions']}")
    
    # Simulate different context (attack scenario)
    attack_context = ContextFeatures(
        hour_of_day=3,
        day_of_week=6,
        is_weekend=True,
        is_business_hours=False,
        current_traffic_rate=10000,
        traffic_vs_baseline=5.0,
        baseline_deviation=0.9,
        threat_level=0.8,
        alert_count_1h=15,
        ioc_hits=3
    )
    
    print(f"\n--- Attack Context ---")
    print(f"Context: Weekend night, high traffic, elevated threat")
    
    result2 = ensemble.forward(model_outputs, attack_context, return_details=True)
    
    print(f"\nWeight Distribution (Attack Mode):")
    for name, info in result2['weight_explanations'].items():
        print(f"  {name}: {info['percentage']} ({info['contribution']})")
    
    print(f"\n{ensemble.explain_current_weights()}")
    
    # Weight analysis
    print(f"\nWeight Analysis:")
    analysis = ensemble.get_weight_analysis()
    for name, stats in analysis.get('model_statistics', {}).items():
        print(f"  {name}: mean={stats['mean_weight']:.3f}, std={stats['std_weight']:.3f}")
    
    print("\n✅ Adaptive Ensemble ready for dynamic weight control!")
