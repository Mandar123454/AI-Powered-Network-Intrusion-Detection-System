"""
Threat Intelligence Package for AI-NIDS

This package provides comprehensive threat intelligence capabilities:
- IOC (Indicators of Compromise) feeds from multiple sources
- Threat intelligence aggregation and correlation
- Real-time feed updates and caching
- IP/Domain/Hash reputation scoring

Supported Intelligence Sources:
- AlienVault OTX (Open Threat Exchange)
- AbuseIPDB
- VirusTotal
- FireHOL IP Lists
- Emerging Threats
- Spamhaus
- CISA Known Exploited Vulnerabilities
- Cisco Talos

Author: AI-NIDS Team
Version: 2.0.0 (Defense Mode)
"""

from .ioc_feeds import (
    IOCFeed,
    AlienVaultOTX,
    AbuseIPDB,
    VirusTotal,
    FireHOL,
    EmergingThreats,
    Spamhaus,
    CISAFeed,
    create_feed_manager
)

from .threat_intel_manager import (
    ThreatIntelManager,
    ThreatIndicator,
    IndicatorType,
    ThreatLevel,
    create_threat_intel_manager
)

from .aggregator import (
    IntelAggregator,
    AggregatedThreat,
    CorrelationEngine,
    create_aggregator
)

from .updater import (
    FeedUpdater,
    UpdateScheduler,
    UpdateResult,
    create_updater
)

__all__ = [
    # IOC Feeds
    'IOCFeed',
    'AlienVaultOTX',
    'AbuseIPDB',
    'VirusTotal',
    'FireHOL',
    'EmergingThreats',
    'Spamhaus',
    'CISAFeed',
    'create_feed_manager',
    
    # Threat Intel Manager
    'ThreatIntelManager',
    'ThreatIndicator',
    'IndicatorType',
    'ThreatLevel',
    'create_threat_intel_manager',
    
    # Aggregator
    'IntelAggregator',
    'AggregatedThreat',
    'CorrelationEngine',
    'create_aggregator',
    
    # Updater
    'FeedUpdater',
    'UpdateScheduler',
    'UpdateResult',
    'create_updater'
]

__version__ = '2.0.0'
__author__ = 'AI-NIDS Team'
