"""
Drift Detector for AI-NIDS

This module detects when network behavior drifts from established baselines.
Critical for:
- Detecting gradual attacks (low-and-slow)
- Identifying compromised hosts
- Triggering model retraining
- Adaptive ensemble weight adjustment

Author: AI-NIDS Team
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

from .baseline_engine import BaselineEngine, BaselineMetrics

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of baseline drift."""
    VOLUME_INCREASE = "volume_increase"
    VOLUME_DECREASE = "volume_decrease"
    PATTERN_CHANGE = "pattern_change"
    NEW_BEHAVIOR = "new_behavior"
    TEMPORAL_SHIFT = "temporal_shift"
    PROTOCOL_ANOMALY = "protocol_anomaly"
    CONNECTION_SURGE = "connection_surge"
    DESTINATION_DRIFT = "destination_drift"


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""
    alert_id: str
    drift_type: DriftType
    entity: str  # IP, subnet, or protocol
    entity_type: str  # 'host', 'subnet', 'protocol'
    severity: float  # 0.0 - 1.0
    drift_score: float  # How much the behavior has drifted
    baseline_value: float
    current_value: float
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'drift_type': self.drift_type.value,
            'entity': self.entity,
            'entity_type': self.entity_type,
            'severity': self.severity,
            'drift_score': self.drift_score,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'description': self.description,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class SlidingWindowStats:
    """Statistics over a sliding time window."""
    
    def __init__(self, window_size_minutes: int = 15):
        self.window_size = timedelta(minutes=window_size_minutes)
        self._values: deque = deque()
        self._timestamps: deque = deque()
    
    def add(self, value: float, timestamp: Optional[datetime] = None):
        """Add a value to the window."""
        ts = timestamp or datetime.now()
        self._values.append(value)
        self._timestamps.append(ts)
        self._clean_old()
    
    def _clean_old(self):
        """Remove values outside the window."""
        cutoff = datetime.now() - self.window_size
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._values.popleft()
    
    @property
    def count(self) -> int:
        self._clean_old()
        return len(self._values)
    
    @property
    def mean(self) -> float:
        self._clean_old()
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)
    
    @property
    def std(self) -> float:
        self._clean_old()
        if len(self._values) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self._values) / len(self._values)
        return math.sqrt(variance)
    
    @property
    def sum(self) -> float:
        self._clean_old()
        return sum(self._values)


class DriftDetector:
    """
    Detects drift from behavioral baselines.
    
    Uses multiple detection methods:
    - Statistical drift (mean/std changes)
    - Volume drift (traffic increases/decreases)
    - Pattern drift (behavioral changes)
    - Temporal drift (time-of-day changes)
    """
    
    def __init__(
        self,
        baseline_engine: BaselineEngine,
        sensitivity: float = 0.7,  # 0-1, higher = more sensitive
        alert_cooldown_minutes: int = 5
    ):
        self.baseline_engine = baseline_engine
        self.sensitivity = sensitivity
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        
        # Sliding windows per entity
        self._windows: Dict[str, Dict[str, SlidingWindowStats]] = {}
        
        # Alert tracking
        self._recent_alerts: Dict[str, datetime] = {}
        self._alert_counter = 0
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        
        # Thresholds (adjusted by sensitivity)
        self._base_thresholds = {
            'z_score': 3.0,
            'volume_ratio': 2.0,
            'pattern_score': 0.5
        }
    
    def register_alert_callback(self, callback: Callable):
        """Register a callback for drift alerts."""
        self._alert_callbacks.append(callback)
    
    def _get_thresholds(self) -> Dict[str, float]:
        """Get thresholds adjusted for sensitivity."""
        # Higher sensitivity = lower thresholds
        factor = 1.5 - (self.sensitivity * 0.5)  # 1.0 to 1.5
        return {k: v * factor for k, v in self._base_thresholds.items()}
    
    def _get_window(self, entity: str, metric: str) -> SlidingWindowStats:
        """Get or create a sliding window for an entity/metric."""
        if entity not in self._windows:
            self._windows[entity] = {}
        if metric not in self._windows[entity]:
            self._windows[entity][metric] = SlidingWindowStats(window_size_minutes=15)
        return self._windows[entity][metric]
    
    def _should_alert(self, entity: str, drift_type: DriftType) -> bool:
        """Check if we should generate an alert (cooldown check)."""
        key = f"{entity}:{drift_type.value}"
        if key in self._recent_alerts:
            if datetime.now() - self._recent_alerts[key] < self.alert_cooldown:
                return False
        return True
    
    def _generate_alert_id(self) -> str:
        """Generate a unique alert ID."""
        self._alert_counter += 1
        return f"DRIFT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"
    
    async def _trigger_alert(self, alert: DriftAlert):
        """Trigger alert callbacks."""
        key = f"{alert.entity}:{alert.drift_type.value}"
        self._recent_alerts[key] = datetime.now()
        
        for callback in self._alert_callbacks:
            try:
                if hasattr(callback, '__await__'):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def check_host_drift(
        self,
        ip: str,
        current_bytes_in: int,
        current_bytes_out: int,
        current_connections: int,
        current_destinations: int
    ) -> List[DriftAlert]:
        """
        Check for drift in host behavior.
        
        Returns list of drift alerts.
        """
        alerts = []
        thresholds = self._get_thresholds()
        
        # Get baseline
        baseline = self.baseline_engine.get_host_baseline(ip)
        
        # Update sliding windows
        self._get_window(ip, 'bytes_in').add(current_bytes_in)
        self._get_window(ip, 'bytes_out').add(current_bytes_out)
        self._get_window(ip, 'connections').add(current_connections)
        self._get_window(ip, 'destinations').add(current_destinations)
        
        # Skip if baseline not established
        if baseline.bytes_in.count < 100:
            return alerts
        
        # Check volume drift - bytes in
        if baseline.bytes_in.mean > 0:
            window_mean = self._get_window(ip, 'bytes_in').mean
            ratio = window_mean / baseline.bytes_in.mean
            
            if ratio > thresholds['volume_ratio']:
                if self._should_alert(ip, DriftType.VOLUME_INCREASE):
                    alerts.append(DriftAlert(
                        alert_id=self._generate_alert_id(),
                        drift_type=DriftType.VOLUME_INCREASE,
                        entity=ip,
                        entity_type='host',
                        severity=min((ratio - 1) / 3, 1.0),
                        drift_score=ratio,
                        baseline_value=baseline.bytes_in.mean,
                        current_value=window_mean,
                        description=f"Inbound traffic {ratio:.1f}x higher than baseline",
                        timestamp=datetime.now(),
                        metadata={'metric': 'bytes_in', 'ratio': ratio}
                    ))
            elif ratio < 1 / thresholds['volume_ratio']:
                if self._should_alert(ip, DriftType.VOLUME_DECREASE):
                    alerts.append(DriftAlert(
                        alert_id=self._generate_alert_id(),
                        drift_type=DriftType.VOLUME_DECREASE,
                        entity=ip,
                        entity_type='host',
                        severity=min((1/ratio - 1) / 3, 1.0),
                        drift_score=ratio,
                        baseline_value=baseline.bytes_in.mean,
                        current_value=window_mean,
                        description=f"Inbound traffic {1/ratio:.1f}x lower than baseline",
                        timestamp=datetime.now(),
                        metadata={'metric': 'bytes_in', 'ratio': ratio}
                    ))
        
        # Check connection surge
        if baseline.connections_per_minute.mean > 0:
            window_conns = self._get_window(ip, 'connections').sum
            expected = baseline.connections_per_minute.mean * 15  # 15-min window
            
            if expected > 0:
                conn_ratio = window_conns / expected
                if conn_ratio > thresholds['volume_ratio']:
                    if self._should_alert(ip, DriftType.CONNECTION_SURGE):
                        alerts.append(DriftAlert(
                            alert_id=self._generate_alert_id(),
                            drift_type=DriftType.CONNECTION_SURGE,
                            entity=ip,
                            entity_type='host',
                            severity=min((conn_ratio - 1) / 5, 1.0),
                            drift_score=conn_ratio,
                            baseline_value=expected,
                            current_value=window_conns,
                            description=f"Connection rate {conn_ratio:.1f}x higher than baseline",
                            timestamp=datetime.now(),
                            metadata={'window_connections': window_conns}
                        ))
        
        # Check destination drift
        if len(baseline.common_destinations) > 10:
            # This would need actual new destination tracking
            pass
        
        return alerts
    
    def check_subnet_drift(
        self,
        subnet: str,
        current_bytes: int,
        current_connections: int,
        active_hosts: int
    ) -> List[DriftAlert]:
        """Check for drift in subnet behavior."""
        alerts = []
        thresholds = self._get_thresholds()
        
        baseline = self.baseline_engine.get_subnet_baseline(subnet)
        
        # Update windows
        self._get_window(subnet, 'bytes').add(current_bytes)
        self._get_window(subnet, 'connections').add(current_connections)
        self._get_window(subnet, 'hosts').add(active_hosts)
        
        if baseline.total_bytes.count < 100:
            return alerts
        
        # Check for subnet-wide traffic surge
        if baseline.total_bytes.mean > 0:
            window_mean = self._get_window(subnet, 'bytes').mean
            ratio = window_mean / baseline.total_bytes.mean
            
            if ratio > thresholds['volume_ratio']:
                if self._should_alert(subnet, DriftType.VOLUME_INCREASE):
                    alerts.append(DriftAlert(
                        alert_id=self._generate_alert_id(),
                        drift_type=DriftType.VOLUME_INCREASE,
                        entity=subnet,
                        entity_type='subnet',
                        severity=min((ratio - 1) / 3, 1.0),
                        drift_score=ratio,
                        baseline_value=baseline.total_bytes.mean,
                        current_value=window_mean,
                        description=f"Subnet traffic {ratio:.1f}x higher than baseline",
                        timestamp=datetime.now()
                    ))
        
        return alerts
    
    def check_protocol_drift(
        self,
        protocol: str,
        current_packet_size: int,
        current_inter_arrival: float
    ) -> List[DriftAlert]:
        """Check for drift in protocol behavior."""
        alerts = []
        thresholds = self._get_thresholds()
        
        baseline = self.baseline_engine.get_protocol_baseline(protocol)
        
        if baseline.packet_size.count < 100:
            return alerts
        
        # Check packet size drift
        is_anom, z_score = baseline.packet_size.is_anomalous(
            current_packet_size, threshold_std=thresholds['z_score']
        )
        
        if is_anom:
            if self._should_alert(protocol, DriftType.PROTOCOL_ANOMALY):
                alerts.append(DriftAlert(
                    alert_id=self._generate_alert_id(),
                    drift_type=DriftType.PROTOCOL_ANOMALY,
                    entity=protocol,
                    entity_type='protocol',
                    severity=min(z_score / 5, 1.0),
                    drift_score=z_score,
                    baseline_value=baseline.packet_size.mean,
                    current_value=current_packet_size,
                    description=f"Unusual {protocol} packet size (z={z_score:.1f})",
                    timestamp=datetime.now()
                ))
        
        return alerts
    
    def get_drift_score(self, entity: str) -> float:
        """
        Get overall drift score for an entity.
        
        Returns 0-1 score indicating how much behavior has drifted.
        """
        if entity not in self._windows:
            return 0.0
        
        baseline = self.baseline_engine.get_host_baseline(entity)
        if baseline.bytes_in.count < 100:
            return 0.0
        
        drift_scores = []
        
        # Calculate drift for each metric
        for metric in ['bytes_in', 'bytes_out', 'connections']:
            if metric in self._windows[entity]:
                window = self._windows[entity][metric]
                baseline_metric = getattr(baseline, metric, None)
                
                if baseline_metric and baseline_metric.mean > 0 and window.count > 0:
                    ratio = window.mean / baseline_metric.mean
                    # Convert ratio to 0-1 score
                    if ratio > 1:
                        score = min((ratio - 1) / 5, 1.0)
                    else:
                        score = min((1/ratio - 1) / 5, 1.0)
                    drift_scores.append(score)
        
        return max(drift_scores) if drift_scores else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            'sensitivity': self.sensitivity,
            'entities_tracked': len(self._windows),
            'recent_alerts': len(self._recent_alerts),
            'total_alerts_generated': self._alert_counter,
            'thresholds': self._get_thresholds()
        }


def create_drift_detector(
    baseline_engine: BaselineEngine,
    sensitivity: float = 0.7
) -> DriftDetector:
    """Create a drift detector."""
    return DriftDetector(
        baseline_engine=baseline_engine,
        sensitivity=sensitivity
    )
