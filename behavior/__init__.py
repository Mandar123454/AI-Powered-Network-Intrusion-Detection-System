"""
Behavioral Analysis Package for AI-NIDS

This package provides network behavior modeling:
- Per-host baseline tracking
- Per-subnet baseline tracking
- Per-protocol baseline tracking
- Baseline drift detection
- Anomaly scoring without ML

Author: AI-NIDS Team
Version: 2.0.0 (Defense Mode)
"""

from .baseline_engine import (
    BaselineEngine,
    HostBaseline,
    SubnetBaseline,
    ProtocolBaseline,
    BaselineMetrics,
    create_baseline_engine
)

from .drift_detector import (
    DriftDetector,
    DriftAlert,
    DriftType,
    create_drift_detector
)

from .entity_profiler import (
    EntityProfiler,
    EntityProfile,
    EntityType,
    BehaviorPattern,
    create_entity_profiler
)

__all__ = [
    # Baseline Engine
    'BaselineEngine',
    'HostBaseline',
    'SubnetBaseline',
    'ProtocolBaseline',
    'BaselineMetrics',
    'create_baseline_engine',
    
    # Drift Detector
    'DriftDetector',
    'DriftAlert',
    'DriftType',
    'create_drift_detector',
    
    # Entity Profiler
    'EntityProfiler',
    'EntityProfile',
    'EntityType',
    'BehaviorPattern',
    'create_entity_profiler'
]

__version__ = '2.0.0'
__author__ = 'AI-NIDS Team'
