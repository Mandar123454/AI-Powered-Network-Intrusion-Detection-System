"""
Autoencoder-based Anomaly Detection Model
Deep learning approach for detecting network anomalies through reconstruction error
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
import pickle
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoencoderNetwork(nn.Module):
    """
    Deep Autoencoder neural network architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        use_batch_norm: bool = True
    ):
        """
        Initialize the autoencoder network.
        
        Args:
            input_dim: Number of input features
            encoding_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = encoding_dims[-1]
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = encoding_dims[-1]
        
        for dim in reversed(encoding_dims[:-1]):
            decoder_layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed output, latent encoding)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)


class AnomalyAutoencoder:
    """
    Autoencoder-based anomaly detection for network intrusion detection.
    Detects anomalies based on reconstruction error.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
        threshold_percentile: float = 95.0,
        device: str = 'auto'
    ):
        """
        Initialize the anomaly autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Number of training epochs
            threshold_percentile: Percentile for anomaly threshold
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold_percentile = threshold_percentile
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build model
        self.model = AutoencoderNetwork(
            input_dim=input_dim,
            encoding_dims=encoding_dims,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Threshold for anomaly detection
        self.threshold: Optional[float] = None
        
        # Training history
        self.history: Dict = {'train_loss': [], 'val_loss': []}
        
        # Metadata
        self.metadata: Dict = {
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'version': '1.0.0',
            'model_type': 'autoencoder_anomaly_detector'
        }
        
        logger.info(f"Initialized Autoencoder on {self.device}")
    
    def _create_dataloader(self, X: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create PyTorch DataLoader from numpy array."""
        tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(tensor, tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Train the autoencoder on normal traffic data.
        
        Args:
            X_train: Training data (should be normal/benign traffic only)
            X_val: Validation data
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        logger.info(f"Training Autoencoder on {X_train.shape[0]} samples...")
        
        train_loader = self._create_dataloader(X_train)
        val_loader = self._create_dataloader(X_val, shuffle=False) if X_val is not None else None
        
        self.model.train()
        
        for epoch in range(self.epochs):
            train_loss = 0.0
            
            for batch_x, _ in train_loader:
                self.optimizer.zero_grad()
                
                reconstructed, _ = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader)
                self.history['val_loss'].append(val_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs} - Loss: {train_loss:.6f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                logger.info(msg)
        
        # Set anomaly threshold based on training data
        self._set_threshold(X_train)
        
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        logger.info(f"Training complete. Threshold: {self.threshold:.6f}")
        
        return self.history
    
    def _evaluate_loss(self, dataloader: DataLoader) -> float:
        """Evaluate loss on a dataloader."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                reconstructed, _ = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(dataloader)
    
    def _set_threshold(self, X: np.ndarray) -> None:
        """Set anomaly detection threshold based on training data."""
        errors = self.get_reconstruction_error(X)
        self.threshold = np.percentile(errors, self.threshold_percentile)
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction error for input data.
        
        Args:
            X: Input data
            
        Returns:
            Array of reconstruction errors
        """
        self.model.eval()
        
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            reconstructed, _ = self.model(tensor)
            errors = torch.mean((tensor - reconstructed) ** 2, dim=1)
            
        return errors.cpu().numpy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 for anomaly, 0 for normal).
        
        Args:
            X: Input data
            
        Returns:
            Binary predictions
        """
        if self.threshold is None:
            raise ValueError("Model not trained. Call train() first.")
        
        errors = self.get_reconstruction_error(X)
        return (errors > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (higher = more likely anomaly).
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores normalized to [0, 1]
        """
        errors = self.get_reconstruction_error(X)
        
        # Normalize scores using sigmoid-like transformation
        scores = 1 / (1 + np.exp(-(errors - self.threshold) / (self.threshold / 4)))
        
        return scores
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model on labeled data.
        
        Args:
            X: Input features
            y: True labels (0: normal, 1: anomaly)
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        
        metrics = {
            'accuracy': float(accuracy_score(y, predictions)),
            'precision': float(precision_score(y, predictions, zero_division=0)),
            'recall': float(recall_score(y, predictions, zero_division=0)),
            'f1': float(f1_score(y, predictions, zero_division=0)),
            'threshold': float(self.threshold)
        }
        
        # Calculate optimal threshold
        errors = self.get_reconstruction_error(X)
        best_f1 = 0
        best_threshold = self.threshold
        
        for percentile in np.arange(90, 99.5, 0.5):
            thresh = np.percentile(errors, percentile)
            preds = (errors > thresh).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        metrics['optimal_threshold'] = float(best_threshold)
        metrics['optimal_f1'] = float(best_f1)
        
        return metrics
    
    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """
        Get latent space representation.
        
        Args:
            X: Input data
            
        Returns:
            Latent representations
        """
        self.model.eval()
        
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            encoded = self.model.encode(tensor)
            
        return encoded.cpu().numpy()
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'threshold_percentile': self.threshold_percentile,
            'threshold': self.threshold,
            'history': self.history,
            'metadata': self.metadata
        }
        
        torch.save(state, path)
        logger.info(f"Saved Autoencoder model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'AnomalyAutoencoder':
        """Load model from disk."""
        state = torch.load(path, map_location='cpu')
        
        model = cls(
            input_dim=state['input_dim'],
            encoding_dims=state['encoding_dims'],
            dropout_rate=state['dropout_rate'],
            learning_rate=state['learning_rate'],
            batch_size=state['batch_size'],
            epochs=state['epochs'],
            threshold_percentile=state['threshold_percentile'],
            device=device
        )
        
        model.model.load_state_dict(state['model_state_dict'])
        model.optimizer.load_state_dict(state['optimizer_state_dict'])
        model.threshold = state['threshold']
        model.history = state['history']
        model.metadata = state['metadata']
        
        logger.info(f"Loaded Autoencoder model from {path}")
        return model


def create_autoencoder(input_dim: int, **kwargs) -> AnomalyAutoencoder:
    """Factory function to create AnomalyAutoencoder."""
    return AnomalyAutoencoder(input_dim=input_dim, **kwargs)
