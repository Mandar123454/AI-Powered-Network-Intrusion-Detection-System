"""
Federated Client - Local Training Node
=======================================
Each client represents a subnet/site/organization that:
1. Trains on local traffic (privacy preserved)
2. Computes gradient updates
3. Sends only gradients to coordinator (never raw data)
4. Receives global model updates

This is how 1,000 networks teach 1 model without exposing secrets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import hashlib
import json
import time
from datetime import datetime, timedelta
import threading
import copy
import os

logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for a federated client."""
    client_id: str
    organization: str = "unknown"
    subnet: str = "0.0.0.0/0"
    
    # Training config
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Privacy config
    differential_privacy: bool = True
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    
    # Communication config
    server_url: str = "localhost:8080"
    heartbeat_interval: int = 60
    sync_interval: int = 300
    
    # Resource limits
    max_memory_mb: int = 1024
    max_cpu_percent: float = 50.0


@dataclass
class TrainingMetrics:
    """Metrics from a local training round."""
    client_id: str
    round_number: int
    samples_trained: int
    loss: float
    accuracy: float
    training_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Attack distribution seen
    attack_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Privacy budget spent
    epsilon_spent: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'client_id': self.client_id,
            'round': self.round_number,
            'samples': self.samples_trained,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'time': self.training_time,
            'timestamp': self.timestamp.isoformat(),
            'attacks': self.attack_distribution,
            'epsilon': self.epsilon_spent
        }


class LocalModel(nn.Module):
    """
    Local detection model that trains on site-specific traffic.
    Lightweight enough to run on edge devices.
    """
    
    def __init__(
        self,
        input_dim: int = 78,
        hidden_dims: List[int] = [128, 64, 32],
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)
        
        # For anomaly detection (autoencoder path)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-2] if len(hidden_dims) > 1 else input_dim),
            nn.ReLU(),
            nn.Linear(hidden_dims[-2] if len(hidden_dims) > 1 else input_dim, input_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_reconstruction: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional feature/reconstruction output."""
        features = self.features(x)
        logits = self.classifier(features)
        
        result = {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1)
        }
        
        if return_features:
            result['features'] = features
        
        if return_reconstruction:
            reconstruction = self.decoder(features)
            result['reconstruction'] = reconstruction
            result['reconstruction_error'] = F.mse_loss(reconstruction, x, reduction='none').mean(dim=-1)
        
        return result
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embedding for transfer/aggregation."""
        return self.features(x)


class GradientCompressor:
    """
    Compresses gradients for efficient transmission.
    Implements Top-K sparsification and quantization.
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.1,
        quantization_bits: int = 8
    ):
        self.compression_ratio = compression_ratio
        self.quantization_bits = quantization_bits
        self.residuals: Dict[str, torch.Tensor] = {}
    
    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        add_residuals: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compress gradients using Top-K sparsification.
        
        Args:
            gradients: Dictionary of parameter gradients
            add_residuals: Whether to add previous residuals
            
        Returns:
            Compressed gradient representation
        """
        compressed = {}
        
        for name, grad in gradients.items():
            flat_grad = grad.flatten()
            
            # Add residual from previous round
            if add_residuals and name in self.residuals:
                flat_grad = flat_grad + self.residuals[name]
            
            # Top-K selection
            k = max(1, int(len(flat_grad) * self.compression_ratio))
            top_k_values, top_k_indices = torch.topk(flat_grad.abs(), k)
            
            # Store residual
            mask = torch.zeros_like(flat_grad)
            mask[top_k_indices] = 1
            self.residuals[name] = flat_grad * (1 - mask)
            
            # Quantize selected values
            selected_values = flat_grad[top_k_indices]
            
            # Scale to quantization range
            max_val = selected_values.abs().max()
            if max_val > 0:
                scale = (2 ** (self.quantization_bits - 1) - 1) / max_val
                quantized = (selected_values * scale).round().to(torch.int8)
            else:
                scale = 1.0
                quantized = torch.zeros(k, dtype=torch.int8)
            
            compressed[name] = {
                'indices': top_k_indices.cpu().numpy().tolist(),
                'values': quantized.cpu().numpy().tolist(),
                'scale': float(scale),
                'shape': list(grad.shape)
            }
        
        return compressed
    
    def decompress(
        self,
        compressed: Dict[str, Dict[str, Any]],
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """Decompress gradients back to full tensors."""
        gradients = {}
        
        for name, data in compressed.items():
            shape = data['shape']
            flat_size = np.prod(shape)
            
            # Reconstruct sparse gradient
            flat_grad = torch.zeros(flat_size, device=device)
            
            indices = torch.tensor(data['indices'], dtype=torch.long, device=device)
            values = torch.tensor(data['values'], dtype=torch.float32, device=device)
            
            # Dequantize
            values = values / data['scale']
            
            flat_grad[indices] = values
            gradients[name] = flat_grad.reshape(shape)
        
        return gradients


class LocalTrainer:
    """
    Handles local training on client data.
    Implements differential privacy and gradient compression.
    """
    
    def __init__(
        self,
        model: LocalModel,
        config: ClientConfig,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        self.compressor = GradientCompressor()
        
        # Privacy accounting
        self.total_epsilon = 0.0
        self.privacy_budget = 10.0  # Total epsilon budget
        
        # Training history
        self.history: List[TrainingMetrics] = []
        self.round_number = 0
    
    def train_round(
        self,
        data_loader: DataLoader,
        global_model_state: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics]:
        """
        Execute one round of local training.
        
        Args:
            data_loader: Local training data
            global_model_state: Global model weights to start from
            
        Returns:
            Gradient updates and training metrics
        """
        start_time = time.time()
        self.round_number += 1
        
        # Load global model if provided
        if global_model_state is not None:
            self.model.load_state_dict(global_model_state)
        
        # Store initial weights for computing updates
        initial_weights = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        attack_counts: Dict[str, int] = {}
        
        for epoch in range(self.config.local_epochs):
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(batch_x)
                loss = F.cross_entropy(output['logits'], batch_y)
                
                loss.backward()
                
                # Apply differential privacy (gradient clipping + noise)
                if self.config.differential_privacy:
                    self._apply_differential_privacy()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                preds = output['logits'].argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)
                
                # Track attack distribution
                for label in batch_y.cpu().numpy():
                    label_name = str(label)
                    attack_counts[label_name] = attack_counts.get(label_name, 0) + 1
        
        # Compute gradient updates (delta from initial weights)
        gradients = {}
        for name, param in self.model.named_parameters():
            gradients[name] = param.data - initial_weights[name]
        
        # Compress gradients
        compressed_grads = self.compressor.compress(gradients)
        
        training_time = time.time() - start_time
        
        metrics = TrainingMetrics(
            client_id=self.config.client_id,
            round_number=self.round_number,
            samples_trained=total,
            loss=total_loss / max(len(data_loader), 1),
            accuracy=correct / max(total, 1),
            training_time=training_time,
            attack_distribution=attack_counts,
            epsilon_spent=self._compute_epsilon_spent(total)
        )
        
        self.history.append(metrics)
        
        logger.info(
            f"Client {self.config.client_id} round {self.round_number}: "
            f"loss={metrics.loss:.4f}, acc={metrics.accuracy:.4f}, "
            f"samples={total}, time={training_time:.2f}s"
        )
        
        return gradients, metrics
    
    def _apply_differential_privacy(self):
        """Apply gradient clipping and noise for differential privacy."""
        # Clip gradients
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coef = self.config.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Add Gaussian noise
        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * (
                    self.config.noise_multiplier * self.config.max_grad_norm
                )
                param.grad.data.add_(noise)
    
    def _compute_epsilon_spent(self, num_samples: int) -> float:
        """Compute privacy budget spent in this round."""
        # Simplified privacy accounting (Rényi DP)
        if not self.config.differential_privacy:
            return 0.0
        
        sigma = self.config.noise_multiplier
        delta = 1e-5
        
        # Approximate epsilon for Gaussian mechanism
        epsilon = (self.config.max_grad_norm ** 2) / (2 * sigma ** 2)
        epsilon += np.sqrt(2 * np.log(1.25 / delta)) * self.config.max_grad_norm / sigma
        
        self.total_epsilon += epsilon
        return epsilon
    
    def get_privacy_budget_remaining(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.privacy_budget - self.total_epsilon)


class FederatedClient:
    """
    Complete federated learning client.
    Manages communication with server and local training.
    """
    
    def __init__(
        self,
        config: ClientConfig,
        model: Optional[LocalModel] = None,
        device: str = 'cpu'
    ):
        self.config = config
        self.device = device
        
        # Initialize model
        if model is None:
            model = LocalModel()
        self.model = model.to(device)
        
        # Initialize trainer
        self.trainer = LocalTrainer(model, config, device)
        
        # Communication state
        self.is_connected = False
        self.last_sync = None
        self.server_round = 0
        
        # Local data buffer
        self.data_buffer: List[Tuple[np.ndarray, int]] = []
        self.max_buffer_size = 10000
        
        # Background threads
        self._stop_event = threading.Event()
        self._heartbeat_thread = None
    
    def add_sample(self, features: np.ndarray, label: int):
        """Add a training sample to local buffer."""
        self.data_buffer.append((features, label))
        
        # Trim buffer if too large
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer[-self.max_buffer_size:]
    
    def add_batch(self, features: np.ndarray, labels: np.ndarray):
        """Add a batch of training samples."""
        for i in range(len(labels)):
            self.add_sample(features[i], int(labels[i]))
    
    def get_data_loader(self) -> Optional[DataLoader]:
        """Create DataLoader from buffered data."""
        if len(self.data_buffer) < self.config.batch_size:
            return None
        
        features = np.array([x[0] for x in self.data_buffer])
        labels = np.array([x[1] for x in self.data_buffer])
        
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
    
    def participate_in_round(
        self,
        global_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], TrainingMetrics]:
        """
        Participate in a federated learning round.
        
        Args:
            global_weights: Current global model weights
            
        Returns:
            Gradient updates and training metrics
        """
        data_loader = self.get_data_loader()
        if data_loader is None:
            raise ValueError("Insufficient data for training")
        
        return self.trainer.train_round(data_loader, global_weights)
    
    def update_model(self, global_weights: Dict[str, torch.Tensor]):
        """Update local model with global weights."""
        self.model.load_state_dict(global_weights)
        self.last_sync = datetime.now()
    
    def predict(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions using local model."""
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32, device=self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            output = self.model(x, return_reconstruction=True)
            
            return {
                'predictions': output['probs'].argmax(dim=-1).cpu().numpy(),
                'probabilities': output['probs'].cpu().numpy(),
                'anomaly_scores': output['reconstruction_error'].cpu().numpy()
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status for reporting."""
        return {
            'client_id': self.config.client_id,
            'organization': self.config.organization,
            'subnet': self.config.subnet,
            'is_connected': self.is_connected,
            'last_sync': self.last_sync.isoformat() if self.last_sync else None,
            'buffer_size': len(self.data_buffer),
            'rounds_completed': self.trainer.round_number,
            'privacy_remaining': self.trainer.get_privacy_budget_remaining(),
            'total_samples_trained': sum(m.samples_trained for m in self.trainer.history)
        }
    
    def save_state(self, path: str):
        """Save client state to disk."""
        state = {
            'model': self.model.state_dict(),
            'config': self.config.__dict__,
            'round_number': self.trainer.round_number,
            'total_epsilon': self.trainer.total_epsilon,
            'history': [m.to_dict() for m in self.trainer.history]
        }
        torch.save(state, path)
        logger.info(f"Saved client state to {path}")
    
    def load_state(self, path: str):
        """Load client state from disk."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.trainer.round_number = state['round_number']
        self.trainer.total_epsilon = state['total_epsilon']
        logger.info(f"Loaded client state from {path}")


def create_federated_client(
    client_id: str,
    organization: str = "default",
    subnet: str = "0.0.0.0/0",
    device: str = 'cpu',
    **kwargs
) -> FederatedClient:
    """
    Factory function to create a federated client.
    
    Args:
        client_id: Unique identifier for this client
        organization: Organization name
        subnet: Network subnet this client monitors
        device: Device to run on
        **kwargs: Additional ClientConfig parameters
        
    Returns:
        Configured FederatedClient
    """
    config = ClientConfig(
        client_id=client_id,
        organization=organization,
        subnet=subnet,
        **kwargs
    )
    
    return FederatedClient(config, device=device)


if __name__ == "__main__":
    print("Federated Client Demo")
    print("=" * 50)
    
    # Create client
    client = create_federated_client(
        client_id="site-001",
        organization="AcmeCorp",
        subnet="192.168.1.0/24"
    )
    
    print(f"Client ID: {client.config.client_id}")
    print(f"Organization: {client.config.organization}")
    
    # Simulate adding training data
    np.random.seed(42)
    for _ in range(500):
        features = np.random.randn(78).astype(np.float32)
        label = np.random.randint(0, 10)
        client.add_sample(features, label)
    
    print(f"Buffer size: {len(client.data_buffer)}")
    
    # Participate in training round
    gradients, metrics = client.participate_in_round()
    
    print(f"\nTraining Metrics:")
    print(f"  Samples: {metrics.samples_trained}")
    print(f"  Loss: {metrics.loss:.4f}")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Time: {metrics.training_time:.2f}s")
    print(f"  Privacy spent: ε={metrics.epsilon_spent:.4f}")
    
    # Make predictions
    test_features = np.random.randn(10, 78).astype(np.float32)
    predictions = client.predict(test_features)
    
    print(f"\nPredictions: {predictions['predictions']}")
    print(f"Anomaly scores: {predictions['anomaly_scores'][:5]}")
    
    print(f"\nClient Status:")
    for key, value in client.get_status().items():
        print(f"  {key}: {value}")
    
    print("\n✅ Federated Client ready for distributed learning!")
