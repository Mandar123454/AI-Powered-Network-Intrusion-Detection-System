"""
Alert Manager for AI-NIDS
Handles alert creation, storage, and notification
"""

from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    INFO = 5


class AlertStatus(Enum):
    """Alert status values."""
    NEW = 'new'
    ACKNOWLEDGED = 'acknowledged'
    INVESTIGATING = 'investigating'
    RESOLVED = 'resolved'
    FALSE_POSITIVE = 'false_positive'


@dataclass
class Alert:
    """Represents a security alert."""
    id: str
    attack_type: str
    severity: str
    confidence: float
    source_ip: str
    destination_ip: str
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: Optional[str]
    timestamp: datetime
    status: AlertStatus = AlertStatus.NEW
    priority: AlertPriority = AlertPriority.MEDIUM
    model_used: str = 'ensemble'
    shap_explanation: Optional[Dict] = None
    notes: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'attack_type': self.attack_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'priority': self.priority.name,
            'model_used': self.model_used,
            'shap_explanation': self.shap_explanation,
            'notes': self.notes,
            'assigned_to': self.assigned_to,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes,
            'metadata': self.metadata
        }


class AlertManager:
    """
    Manages security alerts for the AI-NIDS system.
    Handles alert creation, deduplication, escalation, and notification.
    """
    
    def __init__(
        self,
        db_session=None,
        config: Optional[Dict] = None
    ):
        """
        Initialize AlertManager.
        
        Args:
            db_session: SQLAlchemy database session
            config: Configuration dictionary
        """
        self.db_session = db_session
        self.config = config or {}
        
        # Alert settings
        self.dedup_window = config.get('dedup_window_minutes', 5)
        self.escalation_threshold = config.get('escalation_threshold', 10)
        self.auto_acknowledge_low = config.get('auto_acknowledge_low', False)
        
        # Notification callbacks
        self.notification_handlers: List[Callable] = []
        
        # In-memory alert cache for deduplication
        self._alert_cache: Dict[str, datetime] = {}
        self._cache_lock = threading.Lock()
        
        # Alert counters for rate limiting
        self._alert_counts: Dict[str, int] = {}
        self._counter_reset_time = datetime.utcnow()
        
        logger.info("AlertManager initialized")
    
    def create_alert(self, detection_result) -> Optional[Alert]:
        """
        Create an alert from a detection result.
        
        Args:
            detection_result: DetectionResult from detector
            
        Returns:
            Alert instance or None if deduplicated
        """
        if not detection_result.is_attack:
            return None
        
        # Generate alert key for deduplication
        alert_key = self._generate_alert_key(detection_result)
        
        # Check for duplicate
        if self._is_duplicate(alert_key):
            logger.debug(f"Duplicate alert suppressed: {alert_key}")
            return None
        
        # Generate unique ID
        alert_id = self._generate_alert_id()
        
        # Determine priority based on severity
        priority = self._severity_to_priority(detection_result.severity.name)
        
        # Create alert
        alert = Alert(
            id=alert_id,
            attack_type=detection_result.attack_type,
            severity=detection_result.severity.name,
            confidence=detection_result.confidence,
            source_ip=detection_result.source_ip or 'Unknown',
            destination_ip=detection_result.destination_ip or 'Unknown',
            source_port=detection_result.source_port,
            destination_port=detection_result.destination_port,
            protocol=detection_result.protocol,
            timestamp=detection_result.timestamp,
            priority=priority,
            model_used=detection_result.model_used,
            shap_explanation=detection_result.shap_explanation,
            metadata=detection_result.metadata
        )
        
        # Auto-acknowledge low priority if configured
        if self.auto_acknowledge_low and priority == AlertPriority.LOW:
            alert.status = AlertStatus.ACKNOWLEDGED
        
        # Save to database
        if self.db_session:
            self._save_to_db(alert)
        
        # Update cache
        self._update_cache(alert_key)
        
        # Send notifications
        self._notify(alert)
        
        # Check for escalation
        self._check_escalation(alert)
        
        logger.info(f"Created alert: {alert_id} - {alert.attack_type} from {alert.source_ip}")
        
        return alert
    
    def _generate_alert_key(self, detection_result) -> str:
        """Generate a key for deduplication."""
        return f"{detection_result.source_ip}:{detection_result.destination_ip}:{detection_result.attack_type}"
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        import uuid
        return f"ALERT-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    def _is_duplicate(self, alert_key: str) -> bool:
        """Check if alert is a duplicate within the dedup window."""
        with self._cache_lock:
            if alert_key in self._alert_cache:
                last_seen = self._alert_cache[alert_key]
                if datetime.utcnow() - last_seen < timedelta(minutes=self.dedup_window):
                    return True
            return False
    
    def _update_cache(self, alert_key: str) -> None:
        """Update the dedup cache."""
        with self._cache_lock:
            self._alert_cache[alert_key] = datetime.utcnow()
            
            # Clean old entries
            cutoff = datetime.utcnow() - timedelta(minutes=self.dedup_window * 2)
            self._alert_cache = {
                k: v for k, v in self._alert_cache.items()
                if v > cutoff
            }
    
    def _severity_to_priority(self, severity: str) -> AlertPriority:
        """Map severity to priority."""
        mapping = {
            'CRITICAL': AlertPriority.CRITICAL,
            'HIGH': AlertPriority.HIGH,
            'MEDIUM': AlertPriority.MEDIUM,
            'LOW': AlertPriority.LOW,
            'INFO': AlertPriority.INFO
        }
        return mapping.get(severity, AlertPriority.MEDIUM)
    
    def _save_to_db(self, alert: Alert) -> None:
        """Save alert to database."""
        try:
            from app.models.database import Alert as DBAlert
            
            db_alert = DBAlert(
                attack_type=alert.attack_type,
                severity=alert.severity.lower(),
                confidence=alert.confidence,
                source_ip=alert.source_ip,
                destination_ip=alert.destination_ip,
                source_port=alert.source_port,
                destination_port=alert.destination_port,
                protocol=alert.protocol,
                timestamp=alert.timestamp,
                status=alert.status.value,
                model_version=alert.model_used,
                shap_values=alert.shap_explanation
            )
            
            self.db_session.add(db_alert)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Failed to save alert to database: {e}")
            if self.db_session:
                self.db_session.rollback()
    
    def _notify(self, alert: Alert) -> None:
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    def _check_escalation(self, alert: Alert) -> None:
        """Check if alert rate requires escalation."""
        # Reset counters hourly
        if datetime.utcnow() - self._counter_reset_time > timedelta(hours=1):
            self._alert_counts = {}
            self._counter_reset_time = datetime.utcnow()
        
        # Update counter
        source_key = alert.source_ip
        self._alert_counts[source_key] = self._alert_counts.get(source_key, 0) + 1
        
        # Check threshold
        if self._alert_counts[source_key] >= self.escalation_threshold:
            logger.warning(f"ESCALATION: High alert rate from {source_key}")
            
            # Create escalation alert
            escalation = Alert(
                id=self._generate_alert_id(),
                attack_type='High Alert Rate Detected',
                severity='CRITICAL',
                confidence=1.0,
                source_ip=source_key,
                destination_ip='Multiple',
                source_port=None,
                destination_port=None,
                protocol=None,
                timestamp=datetime.utcnow(),
                priority=AlertPriority.CRITICAL,
                metadata={'alert_count': self._alert_counts[source_key]}
            )
            
            self._notify(escalation)
    
    def register_notification_handler(self, handler: Callable) -> None:
        """Register a notification callback."""
        self.notification_handlers.append(handler)
    
    def acknowledge_alert(self, alert_id: str, user: Optional[str] = None) -> bool:
        """Acknowledge an alert."""
        if not self.db_session:
            return False
        
        try:
            from app.models.database import Alert as DBAlert
            
            alert = self.db_session.query(DBAlert).filter_by(id=alert_id).first()
            if alert:
                alert.status = 'acknowledged'
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.utcnow()
                self.db_session.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: str = '',
        user: Optional[str] = None
    ) -> bool:
        """Resolve an alert."""
        if not self.db_session:
            return False
        
        try:
            from app.models.database import Alert as DBAlert
            
            alert = self.db_session.query(DBAlert).filter_by(id=alert_id).first()
            if alert:
                alert.status = 'resolved'
                alert.resolved_by = user
                alert.resolved_at = datetime.utcnow()
                self.db_session.commit()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    def get_active_alerts(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        attack_type: Optional[str] = None
    ) -> List[Dict]:
        """Get active (unresolved) alerts."""
        if not self.db_session:
            return []
        
        try:
            from app.models.database import Alert as DBAlert
            
            query = self.db_session.query(DBAlert).filter(
                DBAlert.status.in_(['new', 'acknowledged', 'investigating'])
            )
            
            if severity:
                query = query.filter(DBAlert.severity == severity.lower())
            
            if attack_type:
                query = query.filter(DBAlert.attack_type == attack_type)
            
            alerts = query.order_by(DBAlert.timestamp.desc()).limit(limit).all()
            
            return [a.to_dict() for a in alerts]
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    def get_alert_stats(self, hours: int = 24) -> Dict:
        """Get alert statistics for the past N hours."""
        if not self.db_session:
            return {}
        
        try:
            from app.models.database import Alert as DBAlert
            from sqlalchemy import func
            
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            # Total alerts
            total = self.db_session.query(func.count(DBAlert.id)).filter(
                DBAlert.timestamp >= cutoff
            ).scalar()
            
            # By severity
            by_severity = dict(self.db_session.query(
                DBAlert.severity, func.count(DBAlert.id)
            ).filter(DBAlert.timestamp >= cutoff).group_by(DBAlert.severity).all())
            
            # By attack type
            by_type = dict(self.db_session.query(
                DBAlert.attack_type, func.count(DBAlert.id)
            ).filter(DBAlert.timestamp >= cutoff).group_by(DBAlert.attack_type).all())
            
            # By status
            by_status = dict(self.db_session.query(
                DBAlert.status, func.count(DBAlert.id)
            ).filter(DBAlert.timestamp >= cutoff).group_by(DBAlert.status).all())
            
            return {
                'total': total,
                'by_severity': by_severity,
                'by_attack_type': by_type,
                'by_status': by_status,
                'hours': hours
            }
            
        except Exception as e:
            logger.error(f"Failed to get alert stats: {e}")
            return {}


def create_alert_manager(db_session=None, config: Optional[Dict] = None) -> AlertManager:
    """Factory function to create AlertManager."""
    return AlertManager(db_session=db_session, config=config)
