"""
Intelligence Aggregator for AI-NIDS

This module provides:
- Multi-source intelligence correlation
- Threat pattern detection
- Campaign tracking
- Attack attribution hints

Author: AI-NIDS Team
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import hashlib
import json

from .ioc_feeds import IOCEntry, IndicatorType, ThreatCategory
from .threat_intel_manager import ThreatIndicator, ThreatLevel, ThreatIntelManager

logger = logging.getLogger(__name__)


@dataclass
class AggregatedThreat:
    """
    Represents a correlated threat with multiple indicators.
    Used for campaign tracking and attack pattern detection.
    """
    threat_id: str
    name: str
    description: str
    threat_level: ThreatLevel
    confidence: float
    indicators: List[ThreatIndicator]
    categories: Set[ThreatCategory]
    tactics: List[str]  # MITRE ATT&CK tactics
    techniques: List[str]  # MITRE ATT&CK techniques
    first_seen: datetime
    last_seen: datetime
    related_campaigns: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'threat_id': self.threat_id,
            'name': self.name,
            'description': self.description,
            'threat_level': self.threat_level.value,
            'confidence': self.confidence,
            'indicator_count': len(self.indicators),
            'categories': [c.value for c in self.categories],
            'tactics': self.tactics,
            'techniques': self.techniques,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'related_campaigns': self.related_campaigns,
            'metadata': self.metadata
        }


@dataclass
class CorrelationRule:
    """Rule for correlating indicators."""
    rule_id: str
    name: str
    description: str
    conditions: Dict[str, Any]
    severity_boost: float = 0.0
    ttl_minutes: int = 60
    
    def matches(self, indicators: List[ThreatIndicator]) -> bool:
        """Check if indicators match this rule."""
        # Check minimum indicator count
        min_count = self.conditions.get('min_indicators', 1)
        if len(indicators) < min_count:
            return False
        
        # Check required categories
        required_categories = self.conditions.get('required_categories', [])
        if required_categories:
            all_categories = set()
            for ind in indicators:
                all_categories.update(ind.categories)
            if not all(ThreatCategory(c) in all_categories for c in required_categories):
                return False
        
        # Check time window
        time_window = self.conditions.get('time_window_minutes')
        if time_window:
            times = [ind.last_seen for ind in indicators]
            if (max(times) - min(times)).total_seconds() / 60 > time_window:
                return False
        
        # Check minimum confidence
        min_confidence = self.conditions.get('min_confidence', 0.0)
        if not all(ind.confidence >= min_confidence for ind in indicators):
            return False
        
        return True


class CorrelationEngine:
    """
    Engine for correlating threat indicators and detecting patterns.
    
    Features:
    - Multi-indicator correlation
    - Time-based pattern detection
    - Campaign identification
    - MITRE ATT&CK mapping
    """
    
    # Default correlation rules
    DEFAULT_RULES = [
        CorrelationRule(
            rule_id="c2_comm",
            name="C2 Communication Pattern",
            description="Multiple C2-related indicators in short timeframe",
            conditions={
                'min_indicators': 2,
                'required_categories': ['command_and_control'],
                'time_window_minutes': 30,
                'min_confidence': 0.6
            },
            severity_boost=0.2
        ),
        CorrelationRule(
            rule_id="multi_stage",
            name="Multi-Stage Attack",
            description="Indicators spanning multiple attack phases",
            conditions={
                'min_indicators': 3,
                'required_categories': ['exploit', 'malware'],
                'time_window_minutes': 120
            },
            severity_boost=0.3
        ),
        CorrelationRule(
            rule_id="apt_pattern",
            name="APT Activity Pattern",
            description="Advanced persistent threat indicators",
            conditions={
                'min_indicators': 2,
                'required_categories': ['apt'],
                'min_confidence': 0.7
            },
            severity_boost=0.4
        ),
        CorrelationRule(
            rule_id="ransomware_chain",
            name="Ransomware Kill Chain",
            description="Indicators matching ransomware attack pattern",
            conditions={
                'min_indicators': 2,
                'required_categories': ['ransomware'],
                'min_confidence': 0.5
            },
            severity_boost=0.5
        ),
        CorrelationRule(
            rule_id="botnet_activity",
            name="Botnet Activity",
            description="Multiple botnet-related indicators",
            conditions={
                'min_indicators': 3,
                'required_categories': ['botnet'],
                'time_window_minutes': 60
            },
            severity_boost=0.2
        )
    ]
    
    # MITRE ATT&CK mapping
    CATEGORY_TO_TACTICS = {
        ThreatCategory.C2: ["Command and Control"],
        ThreatCategory.MALWARE: ["Execution", "Persistence"],
        ThreatCategory.EXPLOIT: ["Initial Access", "Exploitation"],
        ThreatCategory.RANSOMWARE: ["Impact", "Execution"],
        ThreatCategory.BOTNET: ["Command and Control", "Persistence"],
        ThreatCategory.PHISHING: ["Initial Access"],
        ThreatCategory.BRUTE_FORCE: ["Credential Access"],
        ThreatCategory.SCANNER: ["Reconnaissance", "Discovery"],
        ThreatCategory.APT: ["All Tactics"],
    }
    
    def __init__(self, rules: Optional[List[CorrelationRule]] = None):
        self.rules = rules or self.DEFAULT_RULES
        self._correlation_cache: Dict[str, AggregatedThreat] = {}
        self._indicator_groups: Dict[str, List[ThreatIndicator]] = defaultdict(list)
        
    def add_rule(self, rule: CorrelationRule):
        """Add a correlation rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a correlation rule."""
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
    
    def correlate(
        self,
        indicators: List[ThreatIndicator],
        context: Optional[Dict[str, Any]] = None
    ) -> List[AggregatedThreat]:
        """
        Correlate a list of indicators to find patterns.
        
        Args:
            indicators: List of threat indicators
            context: Optional context (e.g., network info, time range)
        
        Returns:
            List of aggregated threats representing correlated patterns
        """
        aggregated_threats = []
        
        # Group indicators by various dimensions
        by_source_ip = defaultdict(list)
        by_dest_ip = defaultdict(list)
        by_category = defaultdict(list)
        by_time_window = defaultdict(list)
        
        for ind in indicators:
            # Group by categories
            for cat in ind.categories:
                by_category[cat].append(ind)
            
            # Group by time windows (1-hour buckets)
            time_bucket = ind.last_seen.replace(minute=0, second=0, microsecond=0)
            by_time_window[time_bucket].append(ind)
        
        # Apply correlation rules
        for rule in self.rules:
            matched_indicators = []
            
            # Check category-based groupings
            for cat_str in rule.conditions.get('required_categories', []):
                cat = ThreatCategory(cat_str)
                if cat in by_category:
                    matched_indicators.extend(by_category[cat])
            
            # Remove duplicates
            seen = set()
            unique_indicators = []
            for ind in matched_indicators:
                if ind.indicator not in seen:
                    seen.add(ind.indicator)
                    unique_indicators.append(ind)
            
            if rule.matches(unique_indicators):
                threat = self._create_aggregated_threat(
                    rule, unique_indicators, context
                )
                aggregated_threats.append(threat)
        
        # Detect novel patterns not covered by rules
        novel_patterns = self._detect_novel_patterns(indicators)
        aggregated_threats.extend(novel_patterns)
        
        return aggregated_threats
    
    def _create_aggregated_threat(
        self,
        rule: CorrelationRule,
        indicators: List[ThreatIndicator],
        context: Optional[Dict[str, Any]]
    ) -> AggregatedThreat:
        """Create an aggregated threat from matched indicators."""
        # Collect all categories
        categories = set()
        for ind in indicators:
            categories.update(ind.categories)
        
        # Map to MITRE tactics
        tactics = set()
        for cat in categories:
            if cat in self.CATEGORY_TO_TACTICS:
                tactics.update(self.CATEGORY_TO_TACTICS[cat])
        
        # Calculate aggregated confidence
        avg_confidence = sum(ind.confidence for ind in indicators) / len(indicators)
        boosted_confidence = min(avg_confidence + rule.severity_boost, 1.0)
        
        # Determine threat level
        max_risk = max(ind.risk_score for ind in indicators)
        boosted_risk = min(max_risk * (1 + rule.severity_boost), 100)
        
        if boosted_risk >= 90:
            threat_level = ThreatLevel.CRITICAL
        elif boosted_risk >= 70:
            threat_level = ThreatLevel.HIGH
        elif boosted_risk >= 50:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW
        
        # Generate threat ID
        indicator_str = ''.join(sorted(ind.indicator for ind in indicators))
        threat_id = hashlib.sha256(
            f"{rule.rule_id}:{indicator_str}".encode()
        ).hexdigest()[:16]
        
        # Time range
        first_seen = min(ind.first_seen for ind in indicators)
        last_seen = max(ind.last_seen for ind in indicators)
        
        return AggregatedThreat(
            threat_id=threat_id,
            name=rule.name,
            description=rule.description,
            threat_level=threat_level,
            confidence=boosted_confidence,
            indicators=indicators,
            categories=categories,
            tactics=list(tactics),
            techniques=[],  # Could be expanded with technique mapping
            first_seen=first_seen,
            last_seen=last_seen,
            metadata={
                'rule_id': rule.rule_id,
                'severity_boost': rule.severity_boost,
                'context': context or {}
            }
        )
    
    def _detect_novel_patterns(
        self,
        indicators: List[ThreatIndicator]
    ) -> List[AggregatedThreat]:
        """Detect patterns not covered by explicit rules."""
        novel_threats = []
        
        # High-volume pattern: Many indicators in short time
        if len(indicators) >= 5:
            times = [ind.last_seen for ind in indicators]
            time_range = (max(times) - min(times)).total_seconds() / 60
            
            if time_range <= 30:  # 5+ indicators in 30 minutes
                threat_id = hashlib.sha256(
                    f"volume:{datetime.now().isoformat()}".encode()
                ).hexdigest()[:16]
                
                categories = set()
                for ind in indicators:
                    categories.update(ind.categories)
                
                novel_threats.append(AggregatedThreat(
                    threat_id=threat_id,
                    name="High-Volume Threat Activity",
                    description=f"{len(indicators)} threat indicators detected in {time_range:.0f} minutes",
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.7,
                    indicators=indicators,
                    categories=categories,
                    tactics=["Multiple"],
                    techniques=[],
                    first_seen=min(times),
                    last_seen=max(times),
                    metadata={'pattern_type': 'high_volume'}
                ))
        
        # Multi-category pattern: Diverse attack types
        all_categories = set()
        for ind in indicators:
            all_categories.update(ind.categories)
        
        if len(all_categories) >= 3:
            threat_id = hashlib.sha256(
                f"diverse:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]
            
            novel_threats.append(AggregatedThreat(
                threat_id=threat_id,
                name="Multi-Vector Attack Pattern",
                description=f"Attack spanning {len(all_categories)} different categories",
                threat_level=ThreatLevel.HIGH,
                confidence=0.65,
                indicators=indicators,
                categories=all_categories,
                tactics=["Multiple"],
                techniques=[],
                first_seen=min(ind.first_seen for ind in indicators),
                last_seen=max(ind.last_seen for ind in indicators),
                metadata={'pattern_type': 'multi_vector'}
            ))
        
        return novel_threats


class IntelAggregator:
    """
    High-level aggregator combining threat intel with correlation.
    
    Features:
    - Real-time indicator tracking
    - Sliding window correlation
    - Alert generation for correlated threats
    - Campaign tracking over time
    """
    
    def __init__(
        self,
        intel_manager: Optional[ThreatIntelManager] = None,
        correlation_engine: Optional[CorrelationEngine] = None,
        window_size_minutes: int = 60
    ):
        self.intel_manager = intel_manager
        self.correlation_engine = correlation_engine or CorrelationEngine()
        self.window_size = timedelta(minutes=window_size_minutes)
        
        # Sliding window of recent indicators
        self._recent_indicators: List[Tuple[datetime, ThreatIndicator]] = []
        
        # Tracked campaigns
        self._campaigns: Dict[str, AggregatedThreat] = {}
        
        # Alert callbacks
        self._alert_callbacks: List[callable] = []
    
    def register_alert_callback(self, callback: callable):
        """Register a callback for correlated threat alerts."""
        self._alert_callbacks.append(callback)
    
    async def process_indicator(
        self,
        indicator: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[ThreatIndicator]:
        """
        Process a single indicator through the aggregation pipeline.
        
        Returns the threat indicator if found, and triggers correlation.
        """
        if self.intel_manager is None:
            return None
        
        # Look up the indicator
        threat = await self.intel_manager.lookup(indicator)
        
        if threat:
            # Add to sliding window
            now = datetime.now()
            self._recent_indicators.append((now, threat))
            
            # Clean old entries
            self._clean_window()
            
            # Run correlation
            await self._run_correlation(context)
        
        return threat
    
    async def process_batch(
        self,
        indicators: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ThreatIndicator]:
        """Process multiple indicators."""
        if self.intel_manager is None:
            return []
        
        results = await self.intel_manager.bulk_lookup(indicators)
        threats = [t for t in results.values() if t is not None]
        
        # Add to sliding window
        now = datetime.now()
        for threat in threats:
            self._recent_indicators.append((now, threat))
        
        # Clean and correlate
        self._clean_window()
        await self._run_correlation(context)
        
        return threats
    
    def _clean_window(self):
        """Remove indicators outside the sliding window."""
        cutoff = datetime.now() - self.window_size
        self._recent_indicators = [
            (ts, ind) for ts, ind in self._recent_indicators
            if ts > cutoff
        ]
    
    async def _run_correlation(self, context: Optional[Dict[str, Any]] = None):
        """Run correlation on recent indicators."""
        if len(self._recent_indicators) < 2:
            return
        
        indicators = [ind for _, ind in self._recent_indicators]
        aggregated = self.correlation_engine.correlate(indicators, context)
        
        # Check for new or updated campaigns
        for threat in aggregated:
            if threat.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.HIGH):
                # Trigger alerts
                await self._trigger_alerts(threat)
            
            # Track campaigns
            self._campaigns[threat.threat_id] = threat
    
    async def _trigger_alerts(self, threat: AggregatedThreat):
        """Trigger alert callbacks for a correlated threat."""
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(threat)
                else:
                    callback(threat)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def get_active_campaigns(self) -> List[AggregatedThreat]:
        """Get currently tracked campaigns."""
        return list(self._campaigns.values())
    
    def get_recent_indicators(self) -> List[ThreatIndicator]:
        """Get indicators in the current window."""
        return [ind for _, ind in self._recent_indicators]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics."""
        return {
            'window_size_minutes': self.window_size.total_seconds() / 60,
            'indicators_in_window': len(self._recent_indicators),
            'active_campaigns': len(self._campaigns),
            'correlation_rules': len(self.correlation_engine.rules),
            'alert_callbacks': len(self._alert_callbacks)
        }


def create_aggregator(
    intel_manager: Optional[ThreatIntelManager] = None,
    window_size_minutes: int = 60
) -> IntelAggregator:
    """Create an intelligence aggregator."""
    return IntelAggregator(
        intel_manager=intel_manager,
        correlation_engine=CorrelationEngine(),
        window_size_minutes=window_size_minutes
    )
