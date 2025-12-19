"""
LSTM-based Sequence Model for Network Intrusion Detection
Captures temporal patterns in network traffic for attack detection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """
    Bidirectional LSTM network for sequence classification.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM network.
        
        Args:
            input_dim: Number of input features per timestep
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (logits, attention_weights)
        """
        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * num_directions)
        
        # Attention
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_dim * num_directions)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights.squeeze(-1)


class LSTMDetector:
    """
    LSTM-based sequence detector for network intrusion detection.
    Captures temporal dependencies in network flows.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 128,
        epochs: int = 50,
        device: str = 'auto'
    ):
        """
        Initialize LSTM detector.
        
        Args:
            input_dim: Number of input features per timestep
            sequence_length: Length of input sequences
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            learning_rate: Learning rate
            batch_size: Training batch size
            epochs: Number of training epochs
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Build model
        self.model = LSTMNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        self.history: Dict = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        # Metadata
        self.metadata: Dict = {
            'created_at': datetime.now().isoformat(),
            'trained_at': None,
            'version': '1.0.0',
            'model_type': 'lstm_sequence_detector'
        }
        
        logger.info(f"Initialized LSTM Detector on {self.device}")
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from flat data.
        
        Args:
            X: Features array
            y: Labels array
            
        Returns:
            Tuple of (sequences, labels)
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])
        
        return np.array(X_seq), np.array(y_seq)
    
    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        create_sequences: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            create_sequences: Whether to create sequences from flat data
            verbose: Whether to print training progress
            
        Returns:
            Training history
        """
        logger.info(f"Training LSTM on {X_train.shape[0]} samples...")
        
        # Create sequences if needed
        if create_sequences and X_train.ndim == 2:
            X_train, y_train = self._create_sequences(X_train, y_train)
            if X_val is not None:
                X_val, y_val = self._create_sequences(X_val, y_val)
        
        train_loader = self._create_dataloader(X_train, y_train)
        val_loader = self._create_dataloader(X_val, y_val, shuffle=False) if X_val is not None else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 10
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                logits, _ = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            if verbose and (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch + 1}/{self.epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
                if val_loader is not None:
                    msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
                logger.info(msg)
        
        self.metadata['trained_at'] = datetime.now().isoformat()
        
        return self.history
    
    def _evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a dataloader."""
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                logits, _ = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return total_loss / len(dataloader), correct / total
    
    def predict(self, X: np.ndarray, create_sequences: bool = True) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            create_sequences: Whether to create sequences from flat data
            
        Returns:
            Predicted labels
        """
        if create_sequences and X.ndim == 2:
            # Create dummy labels for sequence creation
            dummy_labels = np.zeros(len(X))
            X, _ = self._create_sequences(X, dummy_labels)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            for i in range(0, len(tensor), self.batch_size):
                batch = tensor[i:i + self.batch_size]
                logits, _ = self.model(batch)
                _, preds = torch.max(logits, 1)
                predictions.extend(preds.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray, create_sequences: bool = True) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            create_sequences: Whether to create sequences from flat data
            
        Returns:
            Class probabilities
        """
        if create_sequences and X.ndim == 2:
            dummy_labels = np.zeros(len(X))
            X, _ = self._create_sequences(X, dummy_labels)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            for i in range(0, len(tensor), self.batch_size):
                batch = tensor[i:i + self.batch_size]
                logits, _ = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(probabilities)
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention weights for sequences.
        
        Args:
            X: Input sequences (batch, seq_len, features)
            
        Returns:
            Attention weights (batch, seq_len)
        """
        self.model.eval()
        
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            _, attention_weights = self.model(tensor)
        
        return attention_weights.cpu().numpy()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, create_sequences: bool = True) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Test features
            y: True labels
            create_sequences: Whether to create sequences
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        
        if create_sequences and X.ndim == 2:
            X, y = self._create_sequences(X, y)
        
        predictions = self.predict(X, create_sequences=False)
        
        metrics = {
            'accuracy': float(accuracy_score(y, predictions)),
            'precision': float(precision_score(y, predictions, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, predictions, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, predictions, average='weighted', zero_division=0)),
            'classification_report': classification_report(y, predictions, output_dict=True)
        }
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'history': self.history,
            'metadata': self.metadata
        }
        
        torch.save(state, path)
        logger.info(f"Saved LSTM model to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'auto') -> 'LSTMDetector':
        """Load model from disk."""
        state = torch.load(path, map_location='cpu')
        
        model = cls(
            input_dim=state['input_dim'],
            sequence_length=state['sequence_length'],
            hidden_dim=state['hidden_dim'],
            num_layers=state['num_layers'],
            num_classes=state['num_classes'],
            dropout=state['dropout'],
            bidirectional=state['bidirectional'],
            learning_rate=state['learning_rate'],
            batch_size=state['batch_size'],
            epochs=state['epochs'],
            device=device
        )
        
        model.model.load_state_dict(state['model_state_dict'])
        model.optimizer.load_state_dict(state['optimizer_state_dict'])
        model.history = state['history']
        model.metadata = state['metadata']
        
        logger.info(f"Loaded LSTM model from {path}")
        return model


def create_lstm_detector(input_dim: int, **kwargs) -> LSTMDetector:
    """Factory function to create LSTMDetector."""
    return LSTMDetector(input_dim=input_dim, **kwargs)
