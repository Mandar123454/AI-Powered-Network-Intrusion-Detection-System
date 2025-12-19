"""
Multi-Window Temporal Inference Engine
======================================
Implements sliding window analysis at multiple time scales:
- 1 minute: Burst detection, rapid attacks
- 15 minutes: Session analysis, reconnaissance  
- 1 hour: Slow-and-low attacks, data exfiltration

This captures attack patterns that manifest at different time scales:
- DDoS floods appear in 1-minute windows
- Port scans appear in 15-minute windows
- APT activity appears in 1-hour+ windows

Architecture:
- Temporal Convolutional Networks (TCN) for sequence modeling
- Transformer encoders for long-range dependencies
- Multi-scale fusion for comprehensive detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class TemporalWindow:
    """Configuration for a temporal analysis window."""
    name: str
    duration: timedelta
    resolution: timedelta  # Time step granularity
    max_steps: int = 100  # Maximum time steps to keep
    
    def __post_init__(self):
        self.steps = int(self.duration / self.resolution)


# Standard window configurations
WINDOW_1MIN = TemporalWindow(
    name="1min",
    duration=timedelta(minutes=1),
    resolution=timedelta(seconds=1),
    max_steps=60
)

WINDOW_15MIN = TemporalWindow(
    name="15min", 
    duration=timedelta(minutes=15),
    resolution=timedelta(seconds=15),
    max_steps=60
)

WINDOW_1HOUR = TemporalWindow(
    name="1hour",
    duration=timedelta(hours=1),
    resolution=timedelta(minutes=1),
    max_steps=60
)

WINDOW_24HOUR = TemporalWindow(
    name="24hour",
    duration=timedelta(hours=24),
    resolution=timedelta(minutes=15),
    max_steps=96
)


@dataclass
class TimeStepData:
    """Aggregated data for a single time step."""
    timestamp: datetime
    
    # Traffic metrics
    total_bytes: int = 0
    total_packets: int = 0
    connection_count: int = 0
    unique_sources: int = 0
    unique_destinations: int = 0
    unique_ports: int = 0
    
    # Protocol distribution
    tcp_ratio: float = 0.0
    udp_ratio: float = 0.0
    icmp_ratio: float = 0.0
    other_ratio: float = 0.0
    
    # Connection characteristics
    avg_packet_size: float = 0.0
    avg_duration: float = 0.0
    syn_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    
    # Error indicators
    failed_connections: int = 0
    retransmissions: int = 0
    
    # Behavioral indicators
    new_hosts: int = 0
    entropy_src_ip: float = 0.0
    entropy_dst_ip: float = 0.0
    entropy_dst_port: float = 0.0
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor."""
        features = [
            np.log1p(self.total_bytes),
            np.log1p(self.total_packets),
            np.log1p(self.connection_count),
            np.log1p(self.unique_sources),
            np.log1p(self.unique_destinations),
            np.log1p(self.unique_ports),
            self.tcp_ratio,
            self.udp_ratio,
            self.icmp_ratio,
            self.other_ratio,
            np.log1p(self.avg_packet_size),
            np.log1p(self.avg_duration),
            np.log1p(self.syn_count),
            np.log1p(self.rst_count),
            np.log1p(self.fin_count),
            np.log1p(self.failed_connections),
            np.log1p(self.retransmissions),
            np.log1p(self.new_hosts),
            self.entropy_src_ip,
            self.entropy_dst_ip,
            self.entropy_dst_port
        ]
        return torch.tensor(features, dtype=torch.float32)


class TemporalConvBlock(nn.Module):
    """
    Temporal Convolutional Block with dilated causal convolutions.
    Enables modeling of long-range dependencies efficiently.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, seq_len]
            
        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        residual = self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # Causal trimming
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        return F.relu(out + residual)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for sequence modeling.
    Uses dilated causal convolutions for efficient long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Stacked temporal conv blocks with exponential dilation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(
                TemporalConvBlock(
                    hidden_dim, hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)  # [batch, seq_len, hidden_dim]
        
        # Transpose for conv: [batch, hidden_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply TCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Transpose back: [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        return self.output_proj(x)


class TemporalTransformer(nn.Module):
    """
    Transformer-based temporal encoder for sequence modeling.
    Better for capturing complex temporal patterns and long-range dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Create causal mask for autoregressive modeling
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Apply transformer
        x = self.transformer(x, mask=mask)
        x = self.layer_norm(x)
        
        return x


class SingleWindowDetector(nn.Module):
    """
    Detector for a single temporal window.
    Combines TCN and Transformer for robust pattern detection.
    """
    
    def __init__(
        self,
        window_config: TemporalWindow,
        input_dim: int = 21,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_config = window_config
        
        # TCN for local patterns
        self.tcn = TemporalConvNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=4,
            dropout=dropout
        )
        
        # Transformer for global patterns
        self.transformer = TemporalTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout,
            max_seq_len=window_config.max_steps
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Temporal attention for sequence aggregation
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Anomaly score head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            return_embeddings: Whether to return intermediate embeddings
            
        Returns:
            Dictionary with predictions
        """
        # Process through both branches
        tcn_out = self.tcn(x)          # [batch, seq_len, hidden]
        trans_out = self.transformer(x)  # [batch, seq_len, hidden]
        
        # Fuse representations
        fused = torch.cat([tcn_out, trans_out], dim=-1)
        fused = self.fusion(fused)  # [batch, seq_len, hidden]
        
        # Aggregate sequence using attention
        batch_size = fused.size(0)
        query = self.query.expand(batch_size, -1, -1)
        aggregated, attention_weights = self.temporal_attention(
            query, fused, fused
        )
        aggregated = aggregated.squeeze(1)  # [batch, hidden]
        
        # Predictions
        logits = self.classifier(aggregated)
        anomaly_score = self.anomaly_head(aggregated)
        
        result = {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'anomaly_score': anomaly_score,
            'attention_weights': attention_weights
        }
        
        if return_embeddings:
            result['sequence_embeddings'] = fused
            result['aggregated_embedding'] = aggregated
        
        return result


class MultiWindowTemporalDetector(nn.Module):
    """
    Multi-scale temporal detector using multiple time windows.
    Captures patterns at different time scales and fuses them.
    """
    
    def __init__(
        self,
        windows: List[TemporalWindow] = None,
        input_dim: int = 21,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if windows is None:
            windows = [WINDOW_1MIN, WINDOW_15MIN, WINDOW_1HOUR]
        
        self.windows = windows
        self.num_windows = len(windows)
        self.hidden_dim = hidden_dim
        
        # Create detector for each window
        self.window_detectors = nn.ModuleDict({
            w.name: SingleWindowDetector(
                window_config=w,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
            for w in windows
        })
        
        # Cross-window attention
        self.cross_window_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        
        # Multi-scale fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_windows, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Window importance weights (learnable)
        self.window_importance = nn.Parameter(torch.ones(self.num_windows) / self.num_windows)
        
    def forward(
        self,
        window_inputs: Dict[str, torch.Tensor],
        return_window_results: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-window inputs.
        
        Args:
            window_inputs: Dict mapping window name to input tensor
            return_window_results: Whether to return per-window results
            
        Returns:
            Dictionary with fused predictions
        """
        window_embeddings = []
        window_results = {}
        
        # Process each window
        for window in self.windows:
            if window.name in window_inputs:
                x = window_inputs[window.name]
                result = self.window_detectors[window.name](x, return_embeddings=True)
                window_embeddings.append(result['aggregated_embedding'])
                window_results[window.name] = result
            else:
                # Use zeros if window data not available
                batch_size = next(iter(window_inputs.values())).size(0)
                window_embeddings.append(
                    torch.zeros(batch_size, self.hidden_dim, device=next(self.parameters()).device)
                )
        
        # Stack embeddings: [batch, num_windows, hidden]
        stacked = torch.stack(window_embeddings, dim=1)
        
        # Cross-window attention
        attended, cross_attn = self.cross_window_attention(stacked, stacked, stacked)
        
        # Apply window importance weights
        importance = F.softmax(self.window_importance, dim=0)
        weighted = attended * importance.view(1, -1, 1)
        
        # Concatenate and fuse
        fused = weighted.view(weighted.size(0), -1)  # [batch, num_windows * hidden]
        fused = self.fusion(fused)  # [batch, hidden]
        
        # Final predictions
        logits = self.classifier(fused)
        confidence = self.confidence_head(fused)
        
        # Aggregate window anomaly scores
        window_anomalies = [
            window_results[w.name]['anomaly_score'] 
            for w in self.windows 
            if w.name in window_results
        ]
        if window_anomalies:
            combined_anomaly = torch.stack(window_anomalies, dim=1).mean(dim=1)
        else:
            combined_anomaly = torch.zeros(fused.size(0), 1, device=fused.device)
        
        result = {
            'logits': logits,
            'probs': F.softmax(logits, dim=-1),
            'confidence': confidence,
            'anomaly_score': combined_anomaly,
            'window_importance': importance,
            'cross_attention': cross_attn,
            'fused_embedding': fused
        }
        
        if return_window_results:
            result['window_results'] = window_results
        
        return result


class TemporalDataBuffer:
    """
    Circular buffer for storing temporal data at multiple resolutions.
    Thread-safe for real-time data collection.
    """
    
    def __init__(self, windows: List[TemporalWindow] = None):
        if windows is None:
            windows = [WINDOW_1MIN, WINDOW_15MIN, WINDOW_1HOUR]
        
        self.windows = windows
        self.buffers: Dict[str, deque] = {
            w.name: deque(maxlen=w.max_steps)
            for w in windows
        }
        
        self.current_step: Dict[str, Optional[TimeStepData]] = {
            w.name: None for w in windows
        }
        
        self.last_step_time: Dict[str, Optional[datetime]] = {
            w.name: None for w in windows
        }
        
        self.lock = threading.Lock()
        
        # Aggregation buffers for computing step data
        self._agg_buffers: Dict[str, Dict] = {
            w.name: self._create_agg_buffer() for w in windows
        }
    
    def _create_agg_buffer(self) -> Dict:
        """Create empty aggregation buffer."""
        return {
            'bytes': 0,
            'packets': 0,
            'connections': 0,
            'src_ips': set(),
            'dst_ips': set(),
            'dst_ports': set(),
            'protocols': defaultdict(int),
            'packet_sizes': [],
            'durations': [],
            'syn_count': 0,
            'rst_count': 0,
            'fin_count': 0,
            'failed': 0,
            'retrans': 0,
            'new_hosts': 0
        }
    
    def add_flow(
        self,
        timestamp: datetime,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        bytes_total: int,
        packets: int,
        duration: float,
        flags: Optional[Dict[str, bool]] = None,
        is_new_host: bool = False,
        is_failed: bool = False
    ):
        """Add a flow to the temporal buffer."""
        with self.lock:
            for window in self.windows:
                self._update_window(
                    window, timestamp, src_ip, dst_ip, dst_port,
                    protocol, bytes_total, packets, duration,
                    flags, is_new_host, is_failed
                )
    
    def _update_window(
        self,
        window: TemporalWindow,
        timestamp: datetime,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        bytes_total: int,
        packets: int,
        duration: float,
        flags: Optional[Dict[str, bool]],
        is_new_host: bool,
        is_failed: bool
    ):
        """Update a specific window's buffer."""
        name = window.name
        
        # Check if we need to finalize current step
        if self.last_step_time[name] is not None:
            elapsed = timestamp - self.last_step_time[name]
            if elapsed >= window.resolution:
                # Finalize current step and start new one
                self._finalize_step(window)
                self.last_step_time[name] = timestamp
        else:
            self.last_step_time[name] = timestamp
        
        # Update aggregation buffer
        agg = self._agg_buffers[name]
        agg['bytes'] += bytes_total
        agg['packets'] += packets
        agg['connections'] += 1
        agg['src_ips'].add(src_ip)
        agg['dst_ips'].add(dst_ip)
        agg['dst_ports'].add(dst_port)
        agg['protocols'][protocol] += 1
        agg['packet_sizes'].append(bytes_total / max(packets, 1))
        agg['durations'].append(duration)
        
        if flags:
            if flags.get('SYN'): agg['syn_count'] += 1
            if flags.get('RST'): agg['rst_count'] += 1
            if flags.get('FIN'): agg['fin_count'] += 1
        
        if is_failed: agg['failed'] += 1
        if is_new_host: agg['new_hosts'] += 1
    
    def _finalize_step(self, window: TemporalWindow):
        """Finalize current time step and add to buffer."""
        name = window.name
        agg = self._agg_buffers[name]
        
        if agg['connections'] == 0:
            # No data for this step
            step = TimeStepData(timestamp=self.last_step_time[name] or datetime.now())
        else:
            # Compute protocol ratios
            total_proto = sum(agg['protocols'].values())
            tcp_ratio = agg['protocols'].get('TCP', 0) / total_proto if total_proto > 0 else 0
            udp_ratio = agg['protocols'].get('UDP', 0) / total_proto if total_proto > 0 else 0
            icmp_ratio = agg['protocols'].get('ICMP', 0) / total_proto if total_proto > 0 else 0
            other_ratio = 1 - tcp_ratio - udp_ratio - icmp_ratio
            
            # Compute entropy
            def compute_entropy(items: set) -> float:
                if len(items) <= 1:
                    return 0.0
                counts = defaultdict(int)
                for item in items:
                    counts[item] += 1
                total = len(items)
                probs = [c / total for c in counts.values()]
                return -sum(p * np.log(p + 1e-10) for p in probs)
            
            step = TimeStepData(
                timestamp=self.last_step_time[name] or datetime.now(),
                total_bytes=agg['bytes'],
                total_packets=agg['packets'],
                connection_count=agg['connections'],
                unique_sources=len(agg['src_ips']),
                unique_destinations=len(agg['dst_ips']),
                unique_ports=len(agg['dst_ports']),
                tcp_ratio=tcp_ratio,
                udp_ratio=udp_ratio,
                icmp_ratio=icmp_ratio,
                other_ratio=other_ratio,
                avg_packet_size=np.mean(agg['packet_sizes']) if agg['packet_sizes'] else 0,
                avg_duration=np.mean(agg['durations']) if agg['durations'] else 0,
                syn_count=agg['syn_count'],
                rst_count=agg['rst_count'],
                fin_count=agg['fin_count'],
                failed_connections=agg['failed'],
                retransmissions=agg['retrans'],
                new_hosts=agg['new_hosts'],
                entropy_src_ip=compute_entropy(agg['src_ips']),
                entropy_dst_ip=compute_entropy(agg['dst_ips']),
                entropy_dst_port=compute_entropy({str(p) for p in agg['dst_ports']})
            )
        
        # Add to buffer
        self.buffers[name].append(step)
        
        # Reset aggregation buffer
        self._agg_buffers[name] = self._create_agg_buffer()
    
    def get_window_tensor(
        self,
        window_name: str,
        min_steps: int = 5
    ) -> Optional[torch.Tensor]:
        """
        Get tensor for a specific window.
        
        Args:
            window_name: Name of the window
            min_steps: Minimum steps required
            
        Returns:
            Tensor [1, seq_len, features] or None if insufficient data
        """
        with self.lock:
            buffer = self.buffers.get(window_name)
            if buffer is None or len(buffer) < min_steps:
                return None
            
            tensors = [step.to_tensor() for step in buffer]
            stacked = torch.stack(tensors, dim=0)
            return stacked.unsqueeze(0)  # Add batch dimension
    
    def get_all_window_tensors(
        self,
        min_steps: int = 5
    ) -> Dict[str, torch.Tensor]:
        """Get tensors for all windows with sufficient data."""
        result = {}
        for window in self.windows:
            tensor = self.get_window_tensor(window.name, min_steps)
            if tensor is not None:
                result[window.name] = tensor
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self.lock:
            return {
                window.name: {
                    'buffer_size': len(self.buffers[window.name]),
                    'max_size': window.max_steps,
                    'fill_ratio': len(self.buffers[window.name]) / window.max_steps
                }
                for window in self.windows
            }
    
    def clear(self):
        """Clear all buffers."""
        with self.lock:
            for name in self.buffers:
                self.buffers[name].clear()
                self._agg_buffers[name] = self._create_agg_buffer()
                self.current_step[name] = None
                self.last_step_time[name] = None


class TemporalAnomalyAnalyzer:
    """
    High-level analyzer for temporal anomaly detection.
    Orchestrates multi-window detection and provides interpretable results.
    """
    
    # Attack types for classification
    ATTACK_TYPES = {
        0: 'Normal',
        1: 'Flood/DoS',
        2: 'Scan/Probe',
        3: 'Lateral Movement',
        4: 'Data Exfiltration'
    }
    
    def __init__(
        self,
        model: Optional[MultiWindowTemporalDetector] = None,
        windows: List[TemporalWindow] = None,
        device: str = 'cpu',
        alert_threshold: float = 0.7
    ):
        if windows is None:
            windows = [WINDOW_1MIN, WINDOW_15MIN, WINDOW_1HOUR]
        
        self.windows = windows
        self.device = device
        self.alert_threshold = alert_threshold
        
        # Initialize model
        if model is None:
            self.model = MultiWindowTemporalDetector(
                windows=windows,
                input_dim=21,
                hidden_dim=64,
                num_classes=len(self.ATTACK_TYPES)
            ).to(device)
        else:
            self.model = model.to(device)
        
        # Data buffer
        self.buffer = TemporalDataBuffer(windows)
        
        # Alert history
        self.alert_history: List[Dict] = []
    
    def ingest_flow(
        self,
        timestamp: datetime,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        protocol: str,
        bytes_total: int,
        packets: int,
        duration: float,
        **kwargs
    ):
        """Ingest a network flow for analysis."""
        self.buffer.add_flow(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            dst_port=dst_port,
            protocol=protocol,
            bytes_total=bytes_total,
            packets=packets,
            duration=duration,
            **kwargs
        )
    
    @torch.no_grad()
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze current temporal patterns.
        
        Returns:
            Analysis results with predictions and explanations
        """
        self.model.eval()
        
        # Get window data
        window_tensors = self.buffer.get_all_window_tensors()
        
        if not window_tensors:
            return {
                'status': 'insufficient_data',
                'message': 'Not enough temporal data collected',
                'buffer_stats': self.buffer.get_statistics()
            }
        
        # Move to device
        window_tensors = {k: v.to(self.device) for k, v in window_tensors.items()}
        
        # Run detection
        output = self.model(window_tensors, return_window_results=True)
        
        # Parse results
        predicted_class = output['logits'].argmax(dim=-1).item()
        confidence = output['confidence'].item()
        anomaly_score = output['anomaly_score'].item()
        
        # Get per-window analysis
        window_analysis = {}
        for window in self.windows:
            if window.name in output.get('window_results', {}):
                wr = output['window_results'][window.name]
                window_analysis[window.name] = {
                    'anomaly_score': wr['anomaly_score'].item(),
                    'predicted_class': wr['logits'].argmax(dim=-1).item(),
                    'attention_peaks': self._find_attention_peaks(wr['attention_weights'])
                }
        
        result = {
            'status': 'analyzed',
            'timestamp': datetime.now().isoformat(),
            'prediction': {
                'class': predicted_class,
                'label': self.ATTACK_TYPES.get(predicted_class, 'Unknown'),
                'confidence': confidence,
                'anomaly_score': anomaly_score
            },
            'is_anomaly': anomaly_score > self.alert_threshold,
            'window_analysis': window_analysis,
            'window_importance': {
                w.name: output['window_importance'][i].item()
                for i, w in enumerate(self.windows)
            },
            'buffer_stats': self.buffer.get_statistics()
        }
        
        # Generate alert if anomalous
        if result['is_anomaly']:
            alert = self._generate_alert(result)
            self.alert_history.append(alert)
            result['alert'] = alert
        
        return result
    
    def _find_attention_peaks(
        self,
        attention: torch.Tensor,
        top_k: int = 3
    ) -> List[Dict]:
        """Find top attention peaks in sequence."""
        # attention shape: [batch, 1, seq_len]
        attn = attention.squeeze().cpu().numpy()
        
        if attn.ndim == 0:
            return []
        
        top_indices = np.argsort(attn)[-top_k:][::-1]
        
        return [
            {'position': int(idx), 'weight': float(attn[idx])}
            for idx in top_indices
        ]
    
    def _generate_alert(self, analysis: Dict) -> Dict:
        """Generate detailed alert from analysis."""
        prediction = analysis['prediction']
        
        # Determine severity
        if prediction['anomaly_score'] > 0.9:
            severity = 'CRITICAL'
        elif prediction['anomaly_score'] > 0.8:
            severity = 'HIGH'
        elif prediction['anomaly_score'] > 0.7:
            severity = 'MEDIUM'
        else:
            severity = 'LOW'
        
        # Identify contributing windows
        contributing_windows = [
            name for name, data in analysis['window_analysis'].items()
            if data['anomaly_score'] > 0.5
        ]
        
        # Generate description based on attack type
        descriptions = {
            'Flood/DoS': "High-volume traffic pattern detected suggesting potential denial of service attack",
            'Scan/Probe': "Reconnaissance activity detected with unusual port/host scanning patterns",
            'Lateral Movement': "Internal propagation pattern detected suggesting compromised host spreading",
            'Data Exfiltration': "Abnormal outbound data transfer detected suggesting potential data theft"
        }
        
        return {
            'id': f"TEMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'attack_type': prediction['label'],
            'confidence': prediction['confidence'],
            'anomaly_score': prediction['anomaly_score'],
            'description': descriptions.get(prediction['label'], "Anomalous temporal pattern detected"),
            'contributing_windows': contributing_windows,
            'window_details': analysis['window_analysis'],
            'recommended_action': self._get_recommended_action(prediction['label'], severity)
        }
    
    def _get_recommended_action(self, attack_type: str, severity: str) -> str:
        """Get recommended response action."""
        actions = {
            ('Flood/DoS', 'CRITICAL'): "IMMEDIATE: Enable DDoS mitigation, rate limit affected sources",
            ('Flood/DoS', 'HIGH'): "Rate limit suspicious sources, monitor for escalation",
            ('Scan/Probe', 'CRITICAL'): "Block scanning source IPs, review firewall rules",
            ('Scan/Probe', 'HIGH'): "Monitor scanning activity, prepare containment",
            ('Lateral Movement', 'CRITICAL'): "IMMEDIATE: Isolate affected hosts, initiate IR",
            ('Lateral Movement', 'HIGH'): "Enhanced monitoring of internal traffic, prepare isolation",
            ('Data Exfiltration', 'CRITICAL'): "IMMEDIATE: Block outbound traffic, preserve evidence",
            ('Data Exfiltration', 'HIGH'): "Monitor and log all outbound transfers, alert SOC"
        }
        
        return actions.get(
            (attack_type, severity),
            "Continue monitoring, log activity for analysis"
        )
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends from alert history."""
        if not self.alert_history:
            return {'status': 'no_data', 'message': 'No alerts recorded'}
        
        # Count by type
        type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        hourly_counts = defaultdict(int)
        
        for alert in self.alert_history:
            type_counts[alert['attack_type']] += 1
            severity_counts[alert['severity']] += 1
            
            ts = datetime.fromisoformat(alert['timestamp'])
            hourly_counts[ts.hour] += 1
        
        return {
            'total_alerts': len(self.alert_history),
            'by_attack_type': dict(type_counts),
            'by_severity': dict(severity_counts),
            'by_hour': dict(hourly_counts),
            'most_common_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'peak_hour': max(hourly_counts, key=hourly_counts.get) if hourly_counts else None
        }
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved temporal model to {path}")
    
    def load_model(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Loaded temporal model from {path}")


def create_temporal_detector(
    windows: List[TemporalWindow] = None,
    pretrained_path: Optional[str] = None,
    device: str = 'cpu'
) -> TemporalAnomalyAnalyzer:
    """
    Factory function to create a temporal anomaly analyzer.
    
    Args:
        windows: List of temporal windows to use
        pretrained_path: Path to pretrained model weights
        device: Device to run on
        
    Returns:
        Configured temporal analyzer
    """
    analyzer = TemporalAnomalyAnalyzer(windows=windows, device=device)
    
    if pretrained_path:
        analyzer.load_model(pretrained_path)
    
    return analyzer


if __name__ == "__main__":
    print("Multi-Window Temporal Inference Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = TemporalAnomalyAnalyzer(
        windows=[WINDOW_1MIN, WINDOW_15MIN, WINDOW_1HOUR]
    )
    
    print(f"Model parameters: {sum(p.numel() for p in analyzer.model.parameters()):,}")
    print(f"Windows: {[w.name for w in analyzer.windows]}")
    
    # Simulate network traffic
    from datetime import datetime
    now = datetime.now()
    
    print("\nSimulating network traffic...")
    for i in range(200):
        # Normal traffic
        timestamp = now + timedelta(seconds=i * 0.5)
        analyzer.ingest_flow(
            timestamp=timestamp,
            src_ip=f"192.168.1.{i % 50}",
            dst_ip=f"10.0.0.{i % 20}",
            dst_port=443 if i % 3 == 0 else 80,
            protocol="TCP",
            bytes_total=1000 + i * 10,
            packets=5 + i % 10,
            duration=0.5
        )
        
        # Occasional suspicious traffic (port scan simulation)
        if i % 30 == 0:
            for port in range(1, 100):
                analyzer.ingest_flow(
                    timestamp=timestamp + timedelta(milliseconds=port),
                    src_ip="192.168.1.100",
                    dst_ip="10.0.0.1",
                    dst_port=port,
                    protocol="TCP",
                    bytes_total=60,
                    packets=1,
                    duration=0.01
                )
    
    print(f"\nBuffer Statistics:")
    stats = analyzer.buffer.get_statistics()
    for window_name, data in stats.items():
        print(f"  {window_name}: {data['buffer_size']}/{data['max_size']} steps ({data['fill_ratio']:.1%} full)")
    
    # Analyze
    print("\nRunning temporal analysis...")
    result = analyzer.analyze()
    
    if result['status'] == 'analyzed':
        print(f"\nAnalysis Results:")
        print(f"  Prediction: {result['prediction']['label']}")
        print(f"  Confidence: {result['prediction']['confidence']:.2%}")
        print(f"  Anomaly Score: {result['prediction']['anomaly_score']:.4f}")
        print(f"  Is Anomaly: {result['is_anomaly']}")
        
        print(f"\nWindow Importance:")
        for window_name, importance in result['window_importance'].items():
            print(f"  {window_name}: {importance:.2%}")
        
        if 'alert' in result:
            alert = result['alert']
            print(f"\n⚠️ ALERT Generated:")
            print(f"  ID: {alert['id']}")
            print(f"  Severity: {alert['severity']}")
            print(f"  Description: {alert['description']}")
            print(f"  Recommended: {alert['recommended_action']}")
    else:
        print(f"\nStatus: {result['status']}")
        print(f"Message: {result['message']}")
    
    print("\n✅ Multi-window temporal inference ready!")
