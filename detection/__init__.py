"""
Detection Package for AI-NIDS
"""

from .detector import DetectionEngine, DetectionResult, ThreatSeverity, create_detection_engine
from .alert_manager import AlertManager, Alert, AlertPriority, AlertStatus, create_alert_manager

__all__ = [
    'DetectionEngine', 'DetectionResult', 'ThreatSeverity', 'create_detection_engine',
    'AlertManager', 'Alert', 'AlertPriority', 'AlertStatus', 'create_alert_manager'
]
