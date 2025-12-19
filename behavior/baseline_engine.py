"""
Baseline Engine for AI-NIDS

This module provides behavioral baseline tracking:
- Per-host traffic baselines
- Per-subnet aggregate baselines
- Per-protocol baselines
- Statistical metrics (mean, std, percentiles)
- Time-of-day patterns

This is CRITICAL for detecting anomalies without ML:
- "192.168.1.24 normally does DNS every 5 mins, now doing 60 req/min"
- That's an intrusion WITHOUT any ML model.

Author: AI-NIDS Team
"""

import asyncio
import json
import logging
import math
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Statistical metrics for a baseline."""
    count: int = 0
    total: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    p50: float = 0.0  # Median
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    last_updated: Optional[datetime] = None
    
    # For incremental calculation
    _values: List[float] = field(default_factory=list)
    _m2: float = 0.0  # For Welford's algorithm
    
    def update(self, value: float):
        """Update metrics with a new value using Welford's algorithm."""
        self.count += 1
        self.total += value
        
        # Track min/max
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        # Welford's online algorithm for mean and variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2
        
        if self.count > 1:
            self.std = math.sqrt(self._m2 / (self.count - 1))
        
        # Store values for percentile calculation (with limit)
        self._values.append(value)
        if len(self._values) > 10000:
            # Keep last 10k values for percentiles
            self._values = self._values[-10000:]
        
        self.last_updated = datetime.now()
    
    def calculate_percentiles(self):
        """Calculate percentiles from stored values."""
        if not self._values:
            return
        
        sorted_vals = sorted(self._values)
        n = len(sorted_vals)
        
        self.p50 = sorted_vals[int(n * 0.50)]
        self.p90 = sorted_vals[int(n * 0.90)]
        self.p95 = sorted_vals[int(n * 0.95)]
        self.p99 = sorted_vals[min(int(n * 0.99), n - 1)]
    
    def is_anomalous(self, value: float, threshold_std: float = 3.0) -> Tuple[bool, float]:
        """
        Check if a value is anomalous based on standard deviations.
        
        Returns:
            (is_anomalous, z_score)
        """
        if self.count < 10 or self.std == 0:
            return False, 0.0
        
        z_score = abs(value - self.mean) / self.std
        return z_score > threshold_std, z_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'count': self.count,
            'total': self.total,
            'mean': self.mean,
            'std': self.std,
            'min': self.min_val if self.min_val != float('inf') else None,
            'max': self.max_val if self.max_val != float('-inf') else None,
            'p50': self.p50,
            'p90': self.p90,
            'p95': self.p95,
            'p99': self.p99,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineMetrics':
        """Create from dictionary."""
        metrics = cls()
        metrics.count = data.get('count', 0)
        metrics.total = data.get('total', 0.0)
        metrics.mean = data.get('mean', 0.0)
        metrics.std = data.get('std', 0.0)
        metrics.min_val = data.get('min', float('inf')) or float('inf')
        metrics.max_val = data.get('max', float('-inf')) or float('-inf')
        metrics.p50 = data.get('p50', 0.0)
        metrics.p90 = data.get('p90', 0.0)
        metrics.p95 = data.get('p95', 0.0)
        metrics.p99 = data.get('p99', 0.0)
        if data.get('last_updated'):
            metrics.last_updated = datetime.fromisoformat(data['last_updated'])
        return metrics


@dataclass
class TimeOfDayPattern:
    """Traffic pattern by hour of day."""
    hourly_metrics: Dict[int, BaselineMetrics] = field(
        default_factory=lambda: {h: BaselineMetrics() for h in range(24)}
    )
    
    def update(self, value: float, hour: Optional[int] = None):
        """Update the pattern for a specific hour."""
        if hour is None:
            hour = datetime.now().hour
        self.hourly_metrics[hour].update(value)
    
    def is_anomalous(self, value: float, hour: Optional[int] = None) -> Tuple[bool, float]:
        """Check if value is anomalous for the given hour."""
        if hour is None:
            hour = datetime.now().hour
        return self.hourly_metrics[hour].is_anomalous(value)
    
    def to_dict(self) -> Dict[int, Dict[str, Any]]:
        """Convert to dictionary."""
        return {h: m.to_dict() for h, m in self.hourly_metrics.items()}


@dataclass
class HostBaseline:
    """Baseline for a single host (IP address)."""
    ip_address: str
    first_seen: datetime
    last_seen: datetime
    
    # Traffic volume metrics
    bytes_in: BaselineMetrics = field(default_factory=BaselineMetrics)
    bytes_out: BaselineMetrics = field(default_factory=BaselineMetrics)
    packets_in: BaselineMetrics = field(default_factory=BaselineMetrics)
    packets_out: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Connection metrics
    connections_per_minute: BaselineMetrics = field(default_factory=BaselineMetrics)
    unique_destinations: BaselineMetrics = field(default_factory=BaselineMetrics)
    unique_ports: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Protocol-specific
    dns_queries_per_minute: BaselineMetrics = field(default_factory=BaselineMetrics)
    http_requests_per_minute: BaselineMetrics = field(default_factory=BaselineMetrics)
    https_requests_per_minute: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Time-of-day patterns
    traffic_by_hour: TimeOfDayPattern = field(default_factory=TimeOfDayPattern)
    
    # Common destinations and ports
    common_destinations: Set[str] = field(default_factory=set)
    common_ports: Set[int] = field(default_factory=set)
    common_protocols: Set[str] = field(default_factory=set)
    
    def update_traffic(
        self,
        bytes_in: int,
        bytes_out: int,
        packets_in: int,
        packets_out: int,
        destination: str,
        port: int,
        protocol: str
    ):
        """Update traffic metrics."""
        self.last_seen = datetime.now()
        
        self.bytes_in.update(bytes_in)
        self.bytes_out.update(bytes_out)
        self.packets_in.update(packets_in)
        self.packets_out.update(packets_out)
        
        # Track time-of-day pattern
        total_bytes = bytes_in + bytes_out
        self.traffic_by_hour.update(total_bytes)
        
        # Track common patterns
        self.common_destinations.add(destination)
        self.common_ports.add(port)
        self.common_protocols.add(protocol)
        
        # Limit set sizes
        if len(self.common_destinations) > 1000:
            self.common_destinations = set(list(self.common_destinations)[-1000:])
        if len(self.common_ports) > 100:
            self.common_ports = set(list(self.common_ports)[-100:])
    
    def check_anomalies(
        self,
        bytes_in: int,
        bytes_out: int,
        destination: str,
        port: int
    ) -> List[Dict[str, Any]]:
        """Check for anomalies in the given traffic."""
        anomalies = []
        
        # Check traffic volume
        is_anom, z_score = self.bytes_in.is_anomalous(bytes_in)
        if is_anom:
            anomalies.append({
                'type': 'bytes_in',
                'value': bytes_in,
                'expected_mean': self.bytes_in.mean,
                'z_score': z_score,
                'severity': min(z_score / 3.0, 1.0)
            })
        
        is_anom, z_score = self.bytes_out.is_anomalous(bytes_out)
        if is_anom:
            anomalies.append({
                'type': 'bytes_out',
                'value': bytes_out,
                'expected_mean': self.bytes_out.mean,
                'z_score': z_score,
                'severity': min(z_score / 3.0, 1.0)
            })
        
        # Check for new destination
        if destination not in self.common_destinations and len(self.common_destinations) > 10:
            anomalies.append({
                'type': 'new_destination',
                'value': destination,
                'severity': 0.5
            })
        
        # Check for new port
        if port not in self.common_ports and len(self.common_ports) > 5:
            anomalies.append({
                'type': 'new_port',
                'value': port,
                'severity': 0.3
            })
        
        return anomalies
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ip_address': self.ip_address,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'bytes_in': self.bytes_in.to_dict(),
            'bytes_out': self.bytes_out.to_dict(),
            'packets_in': self.packets_in.to_dict(),
            'packets_out': self.packets_out.to_dict(),
            'connections_per_minute': self.connections_per_minute.to_dict(),
            'dns_queries_per_minute': self.dns_queries_per_minute.to_dict(),
            'common_destinations_count': len(self.common_destinations),
            'common_ports': list(self.common_ports)[:20],
            'common_protocols': list(self.common_protocols)
        }


@dataclass
class SubnetBaseline:
    """Baseline for a subnet (e.g., 192.168.1.0/24)."""
    subnet: str
    first_seen: datetime
    last_seen: datetime
    
    # Aggregate metrics
    total_bytes: BaselineMetrics = field(default_factory=BaselineMetrics)
    total_connections: BaselineMetrics = field(default_factory=BaselineMetrics)
    active_hosts: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Protocol distribution
    tcp_ratio: BaselineMetrics = field(default_factory=BaselineMetrics)
    udp_ratio: BaselineMetrics = field(default_factory=BaselineMetrics)
    icmp_ratio: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Hosts in subnet
    known_hosts: Set[str] = field(default_factory=set)
    
    def update(
        self,
        host_ip: str,
        bytes_total: int,
        connections: int,
        protocol: str
    ):
        """Update subnet metrics."""
        self.last_seen = datetime.now()
        self.known_hosts.add(host_ip)
        
        self.total_bytes.update(bytes_total)
        self.total_connections.update(connections)
        self.active_hosts.update(len(self.known_hosts))
        
        # Track protocol ratios (simplified)
        if protocol.lower() == 'tcp':
            self.tcp_ratio.update(1.0)
            self.udp_ratio.update(0.0)
        elif protocol.lower() == 'udp':
            self.tcp_ratio.update(0.0)
            self.udp_ratio.update(1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'subnet': self.subnet,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'total_bytes': self.total_bytes.to_dict(),
            'total_connections': self.total_connections.to_dict(),
            'known_hosts_count': len(self.known_hosts)
        }


@dataclass
class ProtocolBaseline:
    """Baseline for a specific protocol."""
    protocol: str
    first_seen: datetime
    last_seen: datetime
    
    # Protocol-specific metrics
    packet_size: BaselineMetrics = field(default_factory=BaselineMetrics)
    inter_arrival_time: BaselineMetrics = field(default_factory=BaselineMetrics)
    session_duration: BaselineMetrics = field(default_factory=BaselineMetrics)
    
    # Port usage
    common_ports: Set[int] = field(default_factory=set)
    
    def update(
        self,
        packet_size: int,
        inter_arrival_ms: float,
        port: int
    ):
        """Update protocol metrics."""
        self.last_seen = datetime.now()
        
        self.packet_size.update(packet_size)
        self.inter_arrival_time.update(inter_arrival_ms)
        self.common_ports.add(port)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'protocol': self.protocol,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'packet_size': self.packet_size.to_dict(),
            'inter_arrival_time': self.inter_arrival_time.to_dict(),
            'common_ports': list(self.common_ports)[:20]
        }


class BaselineEngine:
    """
    Central engine for managing all baselines.
    
    This is the heart of behavioral detection:
    - Tracks normal behavior for every entity
    - Detects deviations without ML
    - Provides context for ML models
    """
    
    def __init__(
        self,
        db_path: str = "data/baselines.db",
        learning_period_hours: int = 24,
        min_samples: int = 100
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.learning_period = timedelta(hours=learning_period_hours)
        self.min_samples = min_samples
        
        # In-memory baselines
        self._host_baselines: Dict[str, HostBaseline] = {}
        self._subnet_baselines: Dict[str, SubnetBaseline] = {}
        self._protocol_baselines: Dict[str, ProtocolBaseline] = {}
        
        # Statistics
        self.stats = {
            'flows_processed': 0,
            'anomalies_detected': 0,
            'hosts_tracked': 0,
            'subnets_tracked': 0
        }
        
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS host_baselines (
                    ip_address TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subnet_baselines (
                    subnet TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS protocol_baselines (
                    protocol TEXT PRIMARY KEY,
                    data TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            conn.commit()
    
    def get_host_baseline(self, ip: str) -> HostBaseline:
        """Get or create a host baseline."""
        if ip not in self._host_baselines:
            now = datetime.now()
            self._host_baselines[ip] = HostBaseline(
                ip_address=ip,
                first_seen=now,
                last_seen=now
            )
            self.stats['hosts_tracked'] = len(self._host_baselines)
        return self._host_baselines[ip]
    
    def get_subnet_baseline(self, subnet: str) -> SubnetBaseline:
        """Get or create a subnet baseline."""
        if subnet not in self._subnet_baselines:
            now = datetime.now()
            self._subnet_baselines[subnet] = SubnetBaseline(
                subnet=subnet,
                first_seen=now,
                last_seen=now
            )
            self.stats['subnets_tracked'] = len(self._subnet_baselines)
        return self._subnet_baselines[subnet]
    
    def get_protocol_baseline(self, protocol: str) -> ProtocolBaseline:
        """Get or create a protocol baseline."""
        if protocol not in self._protocol_baselines:
            now = datetime.now()
            self._protocol_baselines[protocol] = ProtocolBaseline(
                protocol=protocol,
                first_seen=now,
                last_seen=now
            )
        return self._protocol_baselines[protocol]
    
    def _ip_to_subnet(self, ip: str) -> str:
        """Convert IP to /24 subnet."""
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        return ip
    
    def process_flow(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        bytes_in: int,
        bytes_out: int,
        packets_in: int,
        packets_out: int,
        duration_ms: float = 0.0
    ) -> Dict[str, Any]:
        """
        Process a network flow and update baselines.
        
        Returns anomaly information if detected.
        """
        self.stats['flows_processed'] += 1
        
        result = {
            'anomalies': [],
            'baseline_drift': 0.0,
            'is_learning': False
        }
        
        # Update source host baseline
        src_baseline = self.get_host_baseline(src_ip)
        src_baseline.update_traffic(
            bytes_in=bytes_out,  # From src perspective
            bytes_out=bytes_in,
            packets_in=packets_out,
            packets_out=packets_in,
            destination=dst_ip,
            port=dst_port,
            protocol=protocol
        )
        
        # Update destination host baseline
        dst_baseline = self.get_host_baseline(dst_ip)
        dst_baseline.update_traffic(
            bytes_in=bytes_in,
            bytes_out=bytes_out,
            packets_in=packets_in,
            packets_out=packets_out,
            destination=src_ip,
            port=src_port,
            protocol=protocol
        )
        
        # Update subnet baselines
        src_subnet = self._ip_to_subnet(src_ip)
        dst_subnet = self._ip_to_subnet(dst_ip)
        
        self.get_subnet_baseline(src_subnet).update(
            src_ip, bytes_in + bytes_out, 1, protocol
        )
        if src_subnet != dst_subnet:
            self.get_subnet_baseline(dst_subnet).update(
                dst_ip, bytes_in + bytes_out, 1, protocol
            )
        
        # Update protocol baseline
        proto_baseline = self.get_protocol_baseline(protocol)
        proto_baseline.update(
            packet_size=(bytes_in + bytes_out) // max(packets_in + packets_out, 1),
            inter_arrival_ms=duration_ms / max(packets_in + packets_out, 1),
            port=dst_port
        )
        
        # Check if still in learning period
        if datetime.now() - src_baseline.first_seen < self.learning_period:
            result['is_learning'] = True
            return result
        
        if src_baseline.bytes_in.count < self.min_samples:
            result['is_learning'] = True
            return result
        
        # Check for anomalies
        anomalies = src_baseline.check_anomalies(
            bytes_in=bytes_out,
            bytes_out=bytes_in,
            destination=dst_ip,
            port=dst_port
        )
        
        if anomalies:
            result['anomalies'] = anomalies
            result['baseline_drift'] = max(a.get('severity', 0) for a in anomalies)
            self.stats['anomalies_detected'] += len(anomalies)
        
        return result
    
    def get_host_profile(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get the behavioral profile for a host."""
        if ip not in self._host_baselines:
            return None
        
        baseline = self._host_baselines[ip]
        
        # Calculate percentiles for key metrics
        baseline.bytes_in.calculate_percentiles()
        baseline.bytes_out.calculate_percentiles()
        
        return {
            'ip': ip,
            'baseline': baseline.to_dict(),
            'normal_behavior': {
                'avg_bytes_in': baseline.bytes_in.mean,
                'avg_bytes_out': baseline.bytes_out.mean,
                'typical_ports': list(baseline.common_ports)[:10],
                'typical_destinations_count': len(baseline.common_destinations),
                'protocols': list(baseline.common_protocols)
            },
            'is_established': baseline.bytes_in.count >= self.min_samples
        }
    
    def get_network_overview(self) -> Dict[str, Any]:
        """Get an overview of the network's baseline."""
        return {
            'hosts_tracked': len(self._host_baselines),
            'subnets_tracked': len(self._subnet_baselines),
            'protocols_tracked': len(self._protocol_baselines),
            'stats': self.stats,
            'subnets': [s.to_dict() for s in self._subnet_baselines.values()],
            'protocols': [p.to_dict() for p in self._protocol_baselines.values()]
        }
    
    def save_baselines(self):
        """Persist baselines to database."""
        with sqlite3.connect(self.db_path) as conn:
            now = datetime.now().isoformat()
            
            for ip, baseline in self._host_baselines.items():
                conn.execute("""
                    INSERT OR REPLACE INTO host_baselines
                    (ip_address, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    ip,
                    json.dumps(baseline.to_dict()),
                    baseline.first_seen.isoformat(),
                    now
                ))
            
            for subnet, baseline in self._subnet_baselines.items():
                conn.execute("""
                    INSERT OR REPLACE INTO subnet_baselines
                    (subnet, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    subnet,
                    json.dumps(baseline.to_dict()),
                    baseline.first_seen.isoformat(),
                    now
                ))
            
            for protocol, baseline in self._protocol_baselines.items():
                conn.execute("""
                    INSERT OR REPLACE INTO protocol_baselines
                    (protocol, data, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    protocol,
                    json.dumps(baseline.to_dict()),
                    baseline.first_seen.isoformat(),
                    now
                ))
            
            conn.commit()
        
        logger.info(f"Saved {len(self._host_baselines)} host baselines")
    
    def load_baselines(self):
        """Load baselines from database."""
        # For now, just log - full deserialization would require more complex logic
        with sqlite3.connect(self.db_path) as conn:
            host_count = conn.execute(
                "SELECT COUNT(*) FROM host_baselines"
            ).fetchone()[0]
            subnet_count = conn.execute(
                "SELECT COUNT(*) FROM subnet_baselines"
            ).fetchone()[0]
            
            logger.info(f"Database has {host_count} hosts, {subnet_count} subnets")


def create_baseline_engine(
    db_path: str = "data/baselines.db",
    learning_period_hours: int = 24
) -> BaselineEngine:
    """Create a baseline engine."""
    return BaselineEngine(
        db_path=db_path,
        learning_period_hours=learning_period_hours
    )
