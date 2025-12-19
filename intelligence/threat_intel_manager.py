"""
Threat Intelligence Manager for AI-NIDS

This module provides:
- Centralized threat intelligence management
- Indicator correlation and enrichment
- Threat scoring and classification
- Real-time lookups with caching

Author: AI-NIDS Team
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json

from .ioc_feeds import (
    IOCEntry, IOCCache, FeedManager, IndicatorType, ThreatCategory,
    create_feed_manager
)

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


@dataclass
class ThreatIndicator:
    """Enriched threat indicator with aggregated intelligence."""
    indicator: str
    indicator_type: IndicatorType
    threat_level: ThreatLevel
    confidence: float
    risk_score: float  # 0.0 - 100.0
    categories: Set[ThreatCategory]
    sources: Set[str]
    first_seen: datetime
    last_seen: datetime
    ioc_entries: List[IOCEntry]
    enrichment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'indicator': self.indicator,
            'indicator_type': self.indicator_type.value,
            'threat_level': self.threat_level.value,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'categories': [c.value for c in self.categories],
            'sources': list(self.sources),
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'enrichment': self.enrichment,
            'ioc_count': len(self.ioc_entries)
        }


class ThreatIntelManager:
    """
    Central manager for threat intelligence operations.
    
    Features:
    - Multi-source IOC aggregation
    - Intelligent caching with TTL
    - Threat scoring and classification
    - Bulk lookups and batch processing
    - Watchlist management
    """
    
    def __init__(
        self,
        feed_manager: Optional[FeedManager] = None,
        cache_ttl: int = 3600,  # 1 hour
        auto_update: bool = True
    ):
        self.feed_manager = feed_manager
        self.cache_ttl = cache_ttl
        self.auto_update = auto_update
        
        # Internal caches
        self._indicator_cache: Dict[str, Tuple[ThreatIndicator, datetime]] = {}
        self._watchlist: Set[str] = set()
        self._whitelist: Set[str] = set()
        
        # Statistics
        self.stats = {
            'lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'threats_detected': 0
        }
    
    async def initialize(
        self,
        otx_api_key: Optional[str] = None,
        abuseipdb_api_key: Optional[str] = None,
        virustotal_api_key: Optional[str] = None
    ):
        """Initialize with API keys."""
        if self.feed_manager is None:
            self.feed_manager = create_feed_manager(
                otx_api_key=otx_api_key,
                abuseipdb_api_key=abuseipdb_api_key,
                virustotal_api_key=virustotal_api_key
            )
        
        # Initial feed update
        if self.auto_update:
            await self.update_feeds()
    
    async def update_feeds(self) -> Dict[str, int]:
        """Update all intelligence feeds."""
        if self.feed_manager is None:
            return {}
        return await self.feed_manager.update_all()
    
    async def lookup(self, indicator: str, force_refresh: bool = False) -> Optional[ThreatIndicator]:
        """
        Look up a single indicator.
        
        Args:
            indicator: IP, domain, hash, or URL to look up
            force_refresh: Bypass cache and query feeds directly
        
        Returns:
            ThreatIndicator if found, None otherwise
        """
        self.stats['lookups'] += 1
        
        # Check whitelist
        if indicator in self._whitelist:
            return None
        
        # Check cache
        if not force_refresh and indicator in self._indicator_cache:
            cached, timestamp = self._indicator_cache[indicator]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                self.stats['cache_hits'] += 1
                return cached
        
        self.stats['cache_misses'] += 1
        
        # Query feeds
        if self.feed_manager is None:
            return None
        
        entries = await self.feed_manager.lookup(indicator)
        if not entries:
            return None
        
        # Aggregate results
        threat_indicator = self._aggregate_entries(indicator, entries)
        
        # Cache result
        self._indicator_cache[indicator] = (threat_indicator, datetime.now())
        
        if threat_indicator.risk_score >= 50:
            self.stats['threats_detected'] += 1
        
        return threat_indicator
    
    async def bulk_lookup(
        self,
        indicators: List[str],
        concurrency: int = 10
    ) -> Dict[str, Optional[ThreatIndicator]]:
        """
        Look up multiple indicators concurrently.
        
        Args:
            indicators: List of indicators to look up
            concurrency: Maximum concurrent lookups
        
        Returns:
            Dictionary mapping indicators to results
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def lookup_with_limit(indicator: str):
            async with semaphore:
                return indicator, await self.lookup(indicator)
        
        tasks = [lookup_with_limit(ind) for ind in indicators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            ind: result if not isinstance(result, Exception) else None
            for ind, result in results
        }
    
    async def check_flow(
        self,
        src_ip: str,
        dst_ip: str,
        src_port: int,
        dst_port: int,
        protocol: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check a network flow against threat intelligence.
        
        Returns enriched flow analysis with threat indicators.
        """
        results = {
            'is_malicious': False,
            'risk_score': 0.0,
            'threat_level': ThreatLevel.INFO.value,
            'indicators_found': [],
            'recommendations': []
        }
        
        # Check source IP
        src_threat = await self.lookup(src_ip)
        if src_threat:
            results['indicators_found'].append({
                'type': 'source_ip',
                'indicator': src_ip,
                'threat': src_threat.to_dict()
            })
            results['risk_score'] = max(results['risk_score'], src_threat.risk_score)
        
        # Check destination IP
        dst_threat = await self.lookup(dst_ip)
        if dst_threat:
            results['indicators_found'].append({
                'type': 'destination_ip',
                'indicator': dst_ip,
                'threat': dst_threat.to_dict()
            })
            results['risk_score'] = max(results['risk_score'], dst_threat.risk_score)
        
        # Check domain if provided
        if domain:
            domain_threat = await self.lookup(domain)
            if domain_threat:
                results['indicators_found'].append({
                    'type': 'domain',
                    'indicator': domain,
                    'threat': domain_threat.to_dict()
                })
                results['risk_score'] = max(results['risk_score'], domain_threat.risk_score)
        
        # Determine overall threat level
        if results['risk_score'] >= 90:
            results['threat_level'] = ThreatLevel.CRITICAL.value
            results['is_malicious'] = True
            results['recommendations'].append("BLOCK immediately - Critical threat detected")
        elif results['risk_score'] >= 70:
            results['threat_level'] = ThreatLevel.HIGH.value
            results['is_malicious'] = True
            results['recommendations'].append("INVESTIGATE and consider blocking")
        elif results['risk_score'] >= 50:
            results['threat_level'] = ThreatLevel.MEDIUM.value
            results['recommendations'].append("MONITOR closely for suspicious activity")
        elif results['risk_score'] >= 25:
            results['threat_level'] = ThreatLevel.LOW.value
            results['recommendations'].append("Log for review")
        
        return results
    
    def _aggregate_entries(self, indicator: str, entries: List[IOCEntry]) -> ThreatIndicator:
        """Aggregate multiple IOC entries into a single threat indicator."""
        if not entries:
            raise ValueError("No entries to aggregate")
        
        # Collect all data
        categories = set()
        sources = set()
        confidences = []
        severities = []
        first_seen = datetime.now()
        last_seen = datetime.now()
        
        for entry in entries:
            categories.add(entry.category)
            sources.add(entry.source)
            confidences.append(entry.confidence)
            severities.append(entry.severity)
            if entry.first_seen < first_seen:
                first_seen = entry.first_seen
            if entry.last_seen > last_seen:
                last_seen = entry.last_seen
        
        # Calculate aggregated confidence and severity
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        avg_severity = sum(severities) / len(severities)
        max_severity = max(severities)
        
        # Multi-source confirmation boosts confidence
        source_boost = min(len(sources) * 0.1, 0.3)
        final_confidence = min(avg_confidence + source_boost, 1.0)
        
        # Calculate risk score (0-100)
        risk_score = (
            (max_severity * 0.5 + avg_severity * 0.5) * 0.6 +
            final_confidence * 0.4
        ) * 100
        
        # Determine threat level
        threat_level = self._calculate_threat_level(risk_score, categories)
        
        # Enrichment data
        enrichment = {
            'source_count': len(sources),
            'category_count': len(categories),
            'confidence_stats': {
                'avg': avg_confidence,
                'max': max_confidence
            },
            'severity_stats': {
                'avg': avg_severity,
                'max': max_severity
            }
        }
        
        return ThreatIndicator(
            indicator=indicator,
            indicator_type=entries[0].indicator_type,
            threat_level=threat_level,
            confidence=final_confidence,
            risk_score=risk_score,
            categories=categories,
            sources=sources,
            first_seen=first_seen,
            last_seen=last_seen,
            ioc_entries=entries,
            enrichment=enrichment
        )
    
    def _calculate_threat_level(
        self,
        risk_score: float,
        categories: Set[ThreatCategory]
    ) -> ThreatLevel:
        """Calculate threat level from risk score and categories."""
        # Critical categories always elevate
        critical_categories = {
            ThreatCategory.RANSOMWARE,
            ThreatCategory.APT,
            ThreatCategory.C2
        }
        
        has_critical = bool(categories & critical_categories)
        
        if risk_score >= 90 or (has_critical and risk_score >= 70):
            return ThreatLevel.CRITICAL
        elif risk_score >= 70:
            return ThreatLevel.HIGH
        elif risk_score >= 50:
            return ThreatLevel.MEDIUM
        elif risk_score >= 25:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.INFO
    
    # Watchlist management
    def add_to_watchlist(self, indicator: str):
        """Add indicator to watchlist for enhanced monitoring."""
        self._watchlist.add(indicator)
    
    def remove_from_watchlist(self, indicator: str):
        """Remove indicator from watchlist."""
        self._watchlist.discard(indicator)
    
    def is_on_watchlist(self, indicator: str) -> bool:
        """Check if indicator is on watchlist."""
        return indicator in self._watchlist
    
    def get_watchlist(self) -> Set[str]:
        """Get all watchlist indicators."""
        return self._watchlist.copy()
    
    # Whitelist management
    def add_to_whitelist(self, indicator: str):
        """Add indicator to whitelist (skip threat checks)."""
        self._whitelist.add(indicator)
    
    def remove_from_whitelist(self, indicator: str):
        """Remove indicator from whitelist."""
        self._whitelist.discard(indicator)
    
    def is_whitelisted(self, indicator: str) -> bool:
        """Check if indicator is whitelisted."""
        return indicator in self._whitelist
    
    def get_whitelist(self) -> Set[str]:
        """Get all whitelisted indicators."""
        return self._whitelist.copy()
    
    # Statistics and management
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self.stats,
            'cache_size': len(self._indicator_cache),
            'watchlist_size': len(self._watchlist),
            'whitelist_size': len(self._whitelist),
            'feed_stats': self.feed_manager.get_stats() if self.feed_manager else {}
        }
    
    def clear_cache(self):
        """Clear the indicator cache."""
        self._indicator_cache.clear()
    
    async def close(self):
        """Clean up resources."""
        if self.feed_manager:
            await self.feed_manager.close_all()


def create_threat_intel_manager(
    otx_api_key: Optional[str] = None,
    abuseipdb_api_key: Optional[str] = None,
    virustotal_api_key: Optional[str] = None,
    cache_ttl: int = 3600
) -> ThreatIntelManager:
    """Create and configure a threat intel manager."""
    feed_manager = create_feed_manager(
        otx_api_key=otx_api_key,
        abuseipdb_api_key=abuseipdb_api_key,
        virustotal_api_key=virustotal_api_key
    )
    
    return ThreatIntelManager(
        feed_manager=feed_manager,
        cache_ttl=cache_ttl
    )
