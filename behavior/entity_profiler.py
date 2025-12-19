"""
Entity Profiler for AI-NIDS

This module provides comprehensive entity profiling:
- Device classification (server, workstation, IoT, etc.)
- Behavioral pattern recognition
- Role inference from traffic patterns
- Risk scoring per entity

Author: AI-NIDS Team
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .baseline_engine import BaselineEngine, HostBaseline

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of network entities."""
    SERVER = "server"
    WORKSTATION = "workstation"
    IOT_DEVICE = "iot_device"
    MOBILE = "mobile"
    NETWORK_DEVICE = "network_device"
    PRINTER = "printer"
    UNKNOWN = "unknown"


class BehaviorPattern(Enum):
    """Identified behavioral patterns."""
    WEB_BROWSING = "web_browsing"
    FILE_TRANSFER = "file_transfer"
    EMAIL_CLIENT = "email_client"
    STREAMING = "streaming"
    GAMING = "gaming"
    VPN_USAGE = "vpn_usage"
    P2P = "peer_to_peer"
    DATABASE = "database_access"
    API_CONSUMER = "api_consumer"
    API_PROVIDER = "api_provider"
    DNS_HEAVY = "dns_heavy"
    BACKUP = "backup"
    IOT_TELEMETRY = "iot_telemetry"
    C2_LIKE = "c2_like"
    SCANNING = "scanning"
    BEACONING = "beaconing"
    DATA_EXFIL = "data_exfiltration"


@dataclass
class EntityProfile:
    """Comprehensive profile of a network entity."""
    entity_id: str  # Usually IP address
    entity_type: EntityType
    confidence: float  # Confidence in classification
    
    # Behavioral patterns
    patterns: List[BehaviorPattern] = field(default_factory=list)
    pattern_scores: Dict[str, float] = field(default_factory=dict)
    
    # Traffic characteristics
    is_mostly_inbound: bool = False
    is_mostly_outbound: bool = False
    typical_ports: Set[int] = field(default_factory=set)
    typical_protocols: Set[str] = field(default_factory=set)
    
    # Temporal patterns
    active_hours: Set[int] = field(default_factory=set)  # Hours 0-23
    is_24x7: bool = False
    is_business_hours_only: bool = False
    
    # Risk indicators
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    
    # Relationships
    frequent_peers: Set[str] = field(default_factory=set)
    internal_only: bool = False
    external_heavy: bool = False
    
    # Metadata
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    total_flows: int = 0
    total_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'entity_id': self.entity_id,
            'entity_type': self.entity_type.value,
            'confidence': self.confidence,
            'patterns': [p.value for p in self.patterns],
            'pattern_scores': self.pattern_scores,
            'is_mostly_inbound': self.is_mostly_inbound,
            'is_mostly_outbound': self.is_mostly_outbound,
            'typical_ports': list(self.typical_ports)[:20],
            'typical_protocols': list(self.typical_protocols),
            'active_hours': list(self.active_hours),
            'is_24x7': self.is_24x7,
            'is_business_hours_only': self.is_business_hours_only,
            'risk_score': self.risk_score,
            'risk_factors': self.risk_factors,
            'frequent_peers_count': len(self.frequent_peers),
            'internal_only': self.internal_only,
            'external_heavy': self.external_heavy,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None,
            'total_flows': self.total_flows,
            'total_bytes': self.total_bytes
        }


class EntityProfiler:
    """
    Profiles network entities based on behavioral analysis.
    
    Uses traffic patterns to:
    - Classify device types
    - Identify behavioral patterns
    - Detect suspicious behaviors
    - Calculate risk scores
    """
    
    # Port-based classification hints
    SERVER_PORTS = {80, 443, 22, 21, 25, 53, 110, 143, 993, 995, 3306, 5432, 1433, 8080, 8443}
    IOT_PORTS = {1883, 8883, 5683, 5684}  # MQTT, CoAP
    
    # Business hours (adjust as needed)
    BUSINESS_HOURS = set(range(8, 18))  # 8 AM to 6 PM
    
    def __init__(
        self,
        baseline_engine: BaselineEngine,
        min_flows_for_classification: int = 100
    ):
        self.baseline_engine = baseline_engine
        self.min_flows = min_flows_for_classification
        
        # Entity profiles
        self._profiles: Dict[str, EntityProfile] = {}
        
        # Traffic tracking
        self._connection_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._port_usage: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._hour_activity: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._direction_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {'in': 0, 'out': 0})
    
    def process_flow(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        bytes_in: int,
        bytes_out: int,
        timestamp: Optional[datetime] = None
    ):
        """Process a flow and update entity profiles."""
        ts = timestamp or datetime.now()
        hour = ts.hour
        
        # Track connections
        self._connection_counts[src_ip][dst_ip] += 1
        self._connection_counts[dst_ip][src_ip] += 1
        
        # Track port usage
        self._port_usage[src_ip][dst_port] += 1
        self._port_usage[dst_ip][src_port] += 1
        
        # Track hourly activity
        self._hour_activity[src_ip][hour] += 1
        self._hour_activity[dst_ip][hour] += 1
        
        # Track direction
        self._direction_stats[src_ip]['out'] += bytes_out
        self._direction_stats[src_ip]['in'] += bytes_in
        self._direction_stats[dst_ip]['in'] += bytes_out
        self._direction_stats[dst_ip]['out'] += bytes_in
        
        # Update profiles
        self._update_profile(src_ip, dst_port, protocol, ts)
        self._update_profile(dst_ip, src_port, protocol, ts)
    
    def _update_profile(
        self,
        ip: str,
        port: int,
        protocol: str,
        timestamp: datetime
    ):
        """Update an entity profile with new observation."""
        if ip not in self._profiles:
            self._profiles[ip] = EntityProfile(
                entity_id=ip,
                entity_type=EntityType.UNKNOWN,
                confidence=0.0,
                first_seen=timestamp
            )
        
        profile = self._profiles[ip]
        profile.last_seen = timestamp
        profile.total_flows += 1
        profile.typical_ports.add(port)
        profile.typical_protocols.add(protocol)
        profile.active_hours.add(timestamp.hour)
        
        # Limit set sizes
        if len(profile.typical_ports) > 100:
            profile.typical_ports = set(list(profile.typical_ports)[-100:])
    
    def classify_entity(self, ip: str) -> EntityProfile:
        """
        Classify an entity based on observed behavior.
        
        Returns updated profile with classification.
        """
        if ip not in self._profiles:
            return EntityProfile(
                entity_id=ip,
                entity_type=EntityType.UNKNOWN,
                confidence=0.0
            )
        
        profile = self._profiles[ip]
        baseline = self.baseline_engine.get_host_baseline(ip)
        
        # Check if we have enough data
        if profile.total_flows < self.min_flows:
            return profile
        
        # Classify entity type
        entity_type, type_confidence = self._classify_entity_type(ip, profile, baseline)
        profile.entity_type = entity_type
        profile.confidence = type_confidence
        
        # Identify behavioral patterns
        patterns = self._identify_patterns(ip, profile, baseline)
        profile.patterns = patterns
        
        # Analyze temporal patterns
        self._analyze_temporal(ip, profile)
        
        # Analyze traffic direction
        self._analyze_direction(ip, profile)
        
        # Calculate risk
        self._calculate_risk(ip, profile, baseline)
        
        return profile
    
    def _classify_entity_type(
        self,
        ip: str,
        profile: EntityProfile,
        baseline: HostBaseline
    ) -> Tuple[EntityType, float]:
        """Classify the entity type based on behavior."""
        scores = {
            EntityType.SERVER: 0.0,
            EntityType.WORKSTATION: 0.0,
            EntityType.IOT_DEVICE: 0.0,
            EntityType.NETWORK_DEVICE: 0.0,
            EntityType.UNKNOWN: 0.1
        }
        
        # Server indicators
        server_ports = profile.typical_ports & self.SERVER_PORTS
        if server_ports:
            scores[EntityType.SERVER] += 0.3 * len(server_ports) / len(self.SERVER_PORTS)
        
        # High inbound ratio suggests server
        if ip in self._direction_stats:
            stats = self._direction_stats[ip]
            total = stats['in'] + stats['out']
            if total > 0:
                in_ratio = stats['in'] / total
                if in_ratio > 0.7:
                    scores[EntityType.SERVER] += 0.3
                elif in_ratio < 0.3:
                    scores[EntityType.WORKSTATION] += 0.2
        
        # 24x7 activity suggests server or IoT
        if len(profile.active_hours) > 20:
            scores[EntityType.SERVER] += 0.2
            scores[EntityType.IOT_DEVICE] += 0.1
        
        # Business hours only suggests workstation
        if profile.active_hours.issubset(self.BUSINESS_HOURS):
            scores[EntityType.WORKSTATION] += 0.3
        
        # IoT indicators
        iot_ports = profile.typical_ports & self.IOT_PORTS
        if iot_ports:
            scores[EntityType.IOT_DEVICE] += 0.4
        
        # Limited port diversity suggests IoT or printer
        if len(profile.typical_ports) < 5:
            scores[EntityType.IOT_DEVICE] += 0.2
        
        # Many unique peers suggests workstation
        if ip in self._connection_counts:
            peer_count = len(self._connection_counts[ip])
            if peer_count > 50:
                scores[EntityType.WORKSTATION] += 0.2
        
        # Find highest score
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]
        
        return best_type, min(confidence, 1.0)
    
    def _identify_patterns(
        self,
        ip: str,
        profile: EntityProfile,
        baseline: HostBaseline
    ) -> List[BehaviorPattern]:
        """Identify behavioral patterns for an entity."""
        patterns = []
        pattern_scores = {}
        
        # Web browsing pattern
        web_ports = {80, 443, 8080, 8443}
        if profile.typical_ports & web_ports:
            score = len(profile.typical_ports & web_ports) / 4
            if score > 0.3:
                patterns.append(BehaviorPattern.WEB_BROWSING)
                pattern_scores[BehaviorPattern.WEB_BROWSING.value] = score
        
        # DNS heavy
        if 53 in profile.typical_ports:
            # Would need actual DNS query count from baseline
            if baseline.dns_queries_per_minute.mean > 1:
                patterns.append(BehaviorPattern.DNS_HEAVY)
                pattern_scores[BehaviorPattern.DNS_HEAVY.value] = 0.6
        
        # API consumer/provider
        if 443 in profile.typical_ports or 8443 in profile.typical_ports:
            if ip in self._direction_stats:
                stats = self._direction_stats[ip]
                total = stats['in'] + stats['out']
                if total > 0:
                    if stats['out'] > stats['in'] * 2:
                        patterns.append(BehaviorPattern.API_CONSUMER)
                        pattern_scores[BehaviorPattern.API_CONSUMER.value] = 0.5
                    elif stats['in'] > stats['out'] * 2:
                        patterns.append(BehaviorPattern.API_PROVIDER)
                        pattern_scores[BehaviorPattern.API_PROVIDER.value] = 0.5
        
        # Beaconing detection (regular intervals)
        # Would need inter-arrival time analysis
        
        # Scanning detection
        if ip in self._port_usage:
            unique_ports = len(self._port_usage[ip])
            if unique_ports > 100:
                patterns.append(BehaviorPattern.SCANNING)
                pattern_scores[BehaviorPattern.SCANNING.value] = min(unique_ports / 500, 1.0)
        
        # P2P pattern
        if ip in self._connection_counts:
            peer_count = len(self._connection_counts[ip])
            if peer_count > 100:
                patterns.append(BehaviorPattern.P2P)
                pattern_scores[BehaviorPattern.P2P.value] = min(peer_count / 200, 1.0)
        
        profile.pattern_scores = pattern_scores
        return patterns
    
    def _analyze_temporal(self, ip: str, profile: EntityProfile):
        """Analyze temporal patterns."""
        if ip not in self._hour_activity:
            return
        
        activity = self._hour_activity[ip]
        active_hours = set(h for h, count in activity.items() if count > 0)
        
        profile.active_hours = active_hours
        profile.is_24x7 = len(active_hours) >= 20
        profile.is_business_hours_only = active_hours.issubset(self.BUSINESS_HOURS)
    
    def _analyze_direction(self, ip: str, profile: EntityProfile):
        """Analyze traffic direction patterns."""
        if ip not in self._direction_stats:
            return
        
        stats = self._direction_stats[ip]
        total = stats['in'] + stats['out']
        
        if total > 0:
            in_ratio = stats['in'] / total
            profile.is_mostly_inbound = in_ratio > 0.7
            profile.is_mostly_outbound = in_ratio < 0.3
    
    def _calculate_risk(
        self,
        ip: str,
        profile: EntityProfile,
        baseline: HostBaseline
    ):
        """Calculate risk score for an entity."""
        risk_factors = []
        risk_score = 0.0
        
        # Suspicious patterns
        suspicious_patterns = {
            BehaviorPattern.C2_LIKE,
            BehaviorPattern.SCANNING,
            BehaviorPattern.BEACONING,
            BehaviorPattern.DATA_EXFIL
        }
        
        found_suspicious = set(profile.patterns) & suspicious_patterns
        if found_suspicious:
            for pattern in found_suspicious:
                risk_factors.append(f"Suspicious pattern: {pattern.value}")
                risk_score += 0.3
        
        # Unusual hours for workstation
        if profile.entity_type == EntityType.WORKSTATION:
            night_hours = set(range(0, 6)) | set(range(22, 24))
            if profile.active_hours & night_hours:
                risk_factors.append("Activity during unusual hours")
                risk_score += 0.2
        
        # Too many unique ports
        if len(profile.typical_ports) > 200:
            risk_factors.append("Excessive port usage")
            risk_score += 0.3
        
        # Check for known bad ports
        risky_ports = {4444, 5555, 6666, 31337}  # Common malware ports
        if profile.typical_ports & risky_ports:
            risk_factors.append("Known suspicious ports detected")
            risk_score += 0.4
        
        profile.risk_score = min(risk_score, 1.0)
        profile.risk_factors = risk_factors
    
    def get_profile(self, ip: str) -> Optional[EntityProfile]:
        """Get the profile for an entity."""
        if ip in self._profiles:
            return self.classify_entity(ip)
        return None
    
    def get_all_profiles(self) -> List[EntityProfile]:
        """Get all entity profiles."""
        return [self.classify_entity(ip) for ip in self._profiles.keys()]
    
    def get_high_risk_entities(self, threshold: float = 0.5) -> List[EntityProfile]:
        """Get entities with risk score above threshold."""
        profiles = self.get_all_profiles()
        return [p for p in profiles if p.risk_score >= threshold]
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[EntityProfile]:
        """Get entities of a specific type."""
        profiles = self.get_all_profiles()
        return [p for p in profiles if p.entity_type == entity_type]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiler statistics."""
        profiles = self.get_all_profiles()
        
        type_counts = defaultdict(int)
        for p in profiles:
            type_counts[p.entity_type.value] += 1
        
        pattern_counts = defaultdict(int)
        for p in profiles:
            for pattern in p.patterns:
                pattern_counts[pattern.value] += 1
        
        high_risk = sum(1 for p in profiles if p.risk_score > 0.5)
        
        return {
            'total_entities': len(profiles),
            'by_type': dict(type_counts),
            'by_pattern': dict(pattern_counts),
            'high_risk_count': high_risk,
            'classified_count': sum(1 for p in profiles if p.entity_type != EntityType.UNKNOWN)
        }


def create_entity_profiler(
    baseline_engine: BaselineEngine,
    min_flows: int = 100
) -> EntityProfiler:
    """Create an entity profiler."""
    return EntityProfiler(
        baseline_engine=baseline_engine,
        min_flows_for_classification=min_flows
    )
