"""
Database Models
===============
SQLAlchemy ORM models for AI-NIDS application.
"""

from datetime import datetime
import secrets
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from app import db


class User(UserMixin, db.Model):
    """User model for authentication."""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='analyst')  # admin, analyst, viewer
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    api_keys = db.relationship('APIKey', backref='user', lazy='dynamic')
    
    def set_password(self, password):
        """Hash and set password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash."""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class Alert(db.Model):
    """Alert model for detected threats."""
    
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Network Information
    source_ip = db.Column(db.String(45), index=True)  # IPv6 compatible
    destination_ip = db.Column(db.String(45), index=True)
    source_port = db.Column(db.Integer)
    destination_port = db.Column(db.Integer)
    protocol = db.Column(db.String(20))
    
    # Detection Information
    attack_type = db.Column(db.String(100), index=True)
    severity = db.Column(db.String(20), index=True)  # critical, high, medium, low, info
    confidence = db.Column(db.Float)  # 0.0 to 1.0
    risk_score = db.Column(db.Float)  # Ensemble score
    
    # Description
    description = db.Column(db.Text)
    
    # Model Information
    model_used = db.Column(db.String(50))  # xgboost, autoencoder, lstm, ensemble
    shap_values = db.Column(db.Text)  # JSON string of SHAP explanations
    
    # Status
    acknowledged = db.Column(db.Boolean, default=False)
    acknowledged_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    acknowledged_at = db.Column(db.DateTime)
    
    resolved = db.Column(db.Boolean, default=False)
    resolved_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    resolved_at = db.Column(db.DateTime)
    resolution_notes = db.Column(db.Text)
    
    # Raw Data
    raw_data = db.Column(db.Text)  # JSON string of original flow data
    
    # Indexes for performance
    __table_args__ = (
        db.Index('idx_alert_severity_timestamp', 'severity', 'timestamp'),
        db.Index('idx_alert_type_timestamp', 'attack_type', 'timestamp'),
    )
    
    def to_dict(self, include_explanation=False):
        """Convert to dictionary."""
        data = {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'attack_type': self.attack_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'description': self.description,
            'model_used': self.model_used,
            'acknowledged': self.acknowledged,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes
        }
        
        if include_explanation and self.shap_values:
            try:
                data['explanation'] = json.loads(self.shap_values)
            except:
                data['explanation'] = None
        
        return data
    
    @property
    def severity_color(self):
        """Get Bootstrap color class for severity."""
        colors = {
            'critical': 'danger',
            'high': 'warning',
            'medium': 'info',
            'low': 'secondary',
            'info': 'light'
        }
        return colors.get(self.severity, 'secondary')
    
    @property
    def severity_icon(self):
        """Get icon for severity."""
        icons = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢',
            'info': '🔵'
        }
        return icons.get(self.severity, '⚪')
    
    def __repr__(self):
        return f'<Alert {self.id} - {self.attack_type} ({self.severity})>'


class NetworkFlow(db.Model):
    """Network flow model for traffic data."""
    
    __tablename__ = 'network_flows'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Network Information
    source_ip = db.Column(db.String(45), index=True)
    destination_ip = db.Column(db.String(45), index=True)
    source_port = db.Column(db.Integer)
    destination_port = db.Column(db.Integer, index=True)
    protocol = db.Column(db.String(20))
    
    # Flow Statistics
    duration = db.Column(db.Float)
    total_bytes = db.Column(db.BigInteger)
    packets_sent = db.Column(db.Integer)
    packets_recv = db.Column(db.Integer)
    bytes_sent = db.Column(db.BigInteger)
    bytes_recv = db.Column(db.BigInteger)
    
    # Flags
    syn_count = db.Column(db.Integer, default=0)
    ack_count = db.Column(db.Integer, default=0)
    fin_count = db.Column(db.Integer, default=0)
    rst_count = db.Column(db.Integer, default=0)
    
    # Classification
    label = db.Column(db.String(50))  # BENIGN, DoS, DDoS, etc.
    predicted_label = db.Column(db.String(50))
    is_anomaly = db.Column(db.Boolean, default=False)
    
    # Raw Data
    raw_data = db.Column(db.Text)  # JSON string
    
    # Indexes for performance
    __table_args__ = (
        db.Index('idx_flow_timestamp', 'timestamp'),
        db.Index('idx_flow_src_dst', 'source_ip', 'destination_ip'),
    )
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol,
            'duration': self.duration,
            'total_bytes': self.total_bytes,
            'packets_sent': self.packets_sent,
            'packets_recv': self.packets_recv,
            'label': self.label,
            'predicted_label': self.predicted_label
        }
    
    def __repr__(self):
        return f'<NetworkFlow {self.source_ip}:{self.source_port} -> {self.destination_ip}:{self.destination_port}>'


class APIKey(db.Model):
    """API Key model for external integrations."""
    
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(64), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    expires_at = db.Column(db.DateTime)
    
    @classmethod
    def generate_key(cls, user_id, name='API Key', expires_days=None):
        """Generate a new API key."""
        key = secrets.token_hex(32)
        
        api_key = cls(
            key=key,
            name=name,
            user_id=user_id
        )
        
        if expires_days:
            from datetime import timedelta
            api_key.expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        return api_key
    
    def is_valid(self):
        """Check if key is valid."""
        if not self.is_active:
            return False
        
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        
        return True
    
    def to_dict(self):
        """Convert to dictionary (without full key)."""
        return {
            'id': self.id,
            'name': self.name,
            'key_prefix': self.key[:8] + '...' if self.key else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }
    
    def __repr__(self):
        return f'<APIKey {self.name}>'


class SystemMetrics(db.Model):
    """System metrics for monitoring."""
    
    __tablename__ = 'system_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    # Processing Metrics
    flows_processed = db.Column(db.Integer, default=0)
    alerts_generated = db.Column(db.Integer, default=0)
    processing_time_ms = db.Column(db.Float)
    
    # System Metrics
    cpu_usage = db.Column(db.Float)
    memory_usage = db.Column(db.Float)
    disk_usage = db.Column(db.Float)
    
    # Model Metrics
    model_inference_time_ms = db.Column(db.Float)
    model_accuracy = db.Column(db.Float)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'flows_processed': self.flows_processed,
            'alerts_generated': self.alerts_generated,
            'processing_time_ms': self.processing_time_ms,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'model_inference_time_ms': self.model_inference_time_ms
        }
    
    def __repr__(self):
        return f'<SystemMetrics {self.timestamp}>'


class ThreatIntelligence(db.Model):
    """Threat intelligence data."""
    
    __tablename__ = 'threat_intelligence'
    
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(45), unique=True, index=True)
    threat_type = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    source = db.Column(db.String(50))  # AbuseIPDB, VirusTotal, etc.
    first_seen = db.Column(db.DateTime)
    last_seen = db.Column(db.DateTime)
    is_blocked = db.Column(db.Boolean, default=False)
    notes = db.Column(db.Text)
    raw_data = db.Column(db.Text)  # JSON from source
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'id': self.id,
            'ip_address': self.ip_address,
            'threat_type': self.threat_type,
            'confidence': self.confidence,
            'source': self.source,
            'is_blocked': self.is_blocked,
            'first_seen': self.first_seen.isoformat() if self.first_seen else None,
            'last_seen': self.last_seen.isoformat() if self.last_seen else None
        }
    
    def __repr__(self):
        return f'<ThreatIntelligence {self.ip_address}>'
