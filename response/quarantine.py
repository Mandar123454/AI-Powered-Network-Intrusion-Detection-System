"""
Quarantine Manager for AI-NIDS

This module manages device quarantine operations:
- Isolate compromised devices
- Maintain quarantine state
- Auto-release after timeout
- Integration with firewall manager

Author: AI-NIDS Team
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .firewall_manager import FirewallManager, FirewallRule, RuleDirection

logger = logging.getLogger(__name__)


class QuarantineReason(Enum):
    """Reasons for quarantine."""
    MALWARE_DETECTED = "malware_detected"
    C2_COMMUNICATION = "c2_communication"
    DATA_EXFILTRATION = "data_exfiltration"
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_BEHAVIOR = "suspicious_behavior"
    POLICY_VIOLATION = "policy_violation"
    MANUAL = "manual"
    THREAT_INTEL_MATCH = "threat_intel_match"


class QuarantineStatus(Enum):
    """Quarantine status."""
    PENDING = "pending"
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class QuarantineEntry:
    """Represents a quarantined device."""
    entry_id: str
    ip_address: str
    hostname: Optional[str]
    reason: QuarantineReason
    status: QuarantineStatus
    severity: float
    created_at: datetime
    expires_at: Optional[datetime]
    released_at: Optional[datetime] = None
    notes: str = ""
    firewall_rules: List[str] = field(default_factory=list)
    allowed_destinations: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_id': self.entry_id,
            'ip_address': self.ip_address,
            'hostname': self.hostname,
            'reason': self.reason.value,
            'status': self.status.value,
            'severity': self.severity,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'released_at': self.released_at.isoformat() if self.released_at else None,
            'notes': self.notes,
            'firewall_rules': self.firewall_rules,
            'allowed_destinations': list(self.allowed_destinations),
            'metadata': self.metadata
        }
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class QuarantineManager:
    """
    Manages device quarantine operations.
    
    Features:
    - Full isolation (block all traffic)
    - Partial isolation (allow specific destinations)
    - Automatic expiration
    - Manual release with approval
    """
    
    # Default allowed destinations during quarantine
    DEFAULT_ALLOWED = {
        "management_server",  # Placeholder - configure with actual IPs
    }
    
    def __init__(
        self,
        firewall_manager: FirewallManager,
        db_path: str = "data/quarantine.db",
        default_duration_hours: int = 24
    ):
        self.firewall = firewall_manager
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_duration = timedelta(hours=default_duration_hours)
        
        # In-memory state
        self._entries: Dict[str, QuarantineEntry] = {}
        self._entry_counter = 0
        
        self._init_db()
        self._load_entries()
    
    def _init_db(self):
        """Initialize the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quarantine_entries (
                    entry_id TEXT PRIMARY KEY,
                    ip_address TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TEXT,
                    expires_at TEXT,
                    status TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ip ON quarantine_entries(ip_address)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON quarantine_entries(status)
            """)
            conn.commit()
    
    def _load_entries(self):
        """Load active entries from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT entry_id, data FROM quarantine_entries
                WHERE status = 'active' OR status = 'pending'
            """)
            for row in cursor:
                data = json.loads(row[1])
                entry = QuarantineEntry(
                    entry_id=data['entry_id'],
                    ip_address=data['ip_address'],
                    hostname=data.get('hostname'),
                    reason=QuarantineReason(data['reason']),
                    status=QuarantineStatus(data['status']),
                    severity=data['severity'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
                    notes=data.get('notes', ''),
                    firewall_rules=data.get('firewall_rules', []),
                    allowed_destinations=set(data.get('allowed_destinations', []))
                )
                self._entries[entry.entry_id] = entry
    
    def _save_entry(self, entry: QuarantineEntry):
        """Save entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO quarantine_entries
                (entry_id, ip_address, data, created_at, expires_at, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entry.entry_id,
                entry.ip_address,
                json.dumps(entry.to_dict()),
                entry.created_at.isoformat(),
                entry.expires_at.isoformat() if entry.expires_at else None,
                entry.status.value
            ))
            conn.commit()
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID."""
        self._entry_counter += 1
        return f"Q{datetime.now().strftime('%Y%m%d%H%M%S')}{self._entry_counter:04d}"
    
    async def quarantine(
        self,
        ip_address: str,
        reason: QuarantineReason,
        severity: float = 0.8,
        duration_hours: Optional[int] = None,
        hostname: Optional[str] = None,
        notes: str = "",
        allowed_destinations: Optional[Set[str]] = None
    ) -> QuarantineEntry:
        """
        Quarantine a device.
        
        Args:
            ip_address: IP to quarantine
            reason: Reason for quarantine
            severity: Threat severity
            duration_hours: How long to quarantine (None = use default)
            hostname: Optional hostname
            notes: Additional notes
            allowed_destinations: IPs that are still allowed
        
        Returns:
            QuarantineEntry
        """
        # Check if already quarantined
        existing = self.get_entry_by_ip(ip_address)
        if existing and existing.status == QuarantineStatus.ACTIVE:
            logger.warning(f"{ip_address} is already quarantined")
            return existing
        
        # Create entry
        duration = timedelta(hours=duration_hours) if duration_hours else self.default_duration
        
        entry = QuarantineEntry(
            entry_id=self._generate_entry_id(),
            ip_address=ip_address,
            hostname=hostname,
            reason=reason,
            status=QuarantineStatus.PENDING,
            severity=severity,
            created_at=datetime.now(),
            expires_at=datetime.now() + duration,
            notes=notes,
            allowed_destinations=allowed_destinations or self.DEFAULT_ALLOWED.copy()
        )
        
        # Apply firewall rules
        success = await self._apply_quarantine_rules(entry)
        
        if success:
            entry.status = QuarantineStatus.ACTIVE
            logger.info(f"Quarantined {ip_address}: {reason.value}")
        else:
            entry.status = QuarantineStatus.FAILED
            logger.error(f"Failed to quarantine {ip_address}")
        
        # Store
        self._entries[entry.entry_id] = entry
        self._save_entry(entry)
        
        return entry
    
    async def _apply_quarantine_rules(self, entry: QuarantineEntry) -> bool:
        """Apply firewall rules for quarantine."""
        rules_created = []
        
        try:
            # Block all inbound
            in_rule = await self.firewall.block_ip(
                ip=entry.ip_address,
                reason=f"Quarantine: {entry.reason.value}",
                duration_hours=int((entry.expires_at - datetime.now()).total_seconds() / 3600) if entry.expires_at else None,
                direction=RuleDirection.INBOUND
            )
            if in_rule:
                rules_created.append(in_rule.rule_id)
            
            # Block all outbound
            out_rule = await self.firewall.block_ip(
                ip=entry.ip_address,
                reason=f"Quarantine: {entry.reason.value}",
                duration_hours=int((entry.expires_at - datetime.now()).total_seconds() / 3600) if entry.expires_at else None,
                direction=RuleDirection.OUTBOUND
            )
            if out_rule:
                rules_created.append(out_rule.rule_id)
            
            # Note: In production, you'd also add ALLOW rules for allowed_destinations
            # This requires more sophisticated firewall management
            
            entry.firewall_rules = rules_created
            return len(rules_created) > 0
            
        except Exception as e:
            logger.error(f"Error applying quarantine rules: {e}")
            # Rollback any created rules
            for rule_id in rules_created:
                await self.firewall.remove_rule(rule_id)
            return False
    
    async def release(
        self,
        entry_id: str,
        released_by: str = "system",
        notes: str = ""
    ) -> bool:
        """
        Release a device from quarantine.
        
        Args:
            entry_id: Entry ID to release
            released_by: Who released it
            notes: Release notes
        
        Returns:
            Success status
        """
        if entry_id not in self._entries:
            return False
        
        entry = self._entries[entry_id]
        
        if entry.status != QuarantineStatus.ACTIVE:
            logger.warning(f"Entry {entry_id} is not active")
            return False
        
        # Remove firewall rules
        for rule_id in entry.firewall_rules:
            await self.firewall.remove_rule(rule_id)
        
        # Update entry
        entry.status = QuarantineStatus.RELEASED
        entry.released_at = datetime.now()
        entry.notes += f"\nReleased by {released_by}: {notes}"
        
        self._save_entry(entry)
        logger.info(f"Released {entry.ip_address} from quarantine")
        
        return True
    
    async def release_by_ip(self, ip_address: str) -> bool:
        """Release a device by IP address."""
        entry = self.get_entry_by_ip(ip_address)
        if entry:
            return await self.release(entry.entry_id)
        return False
    
    async def extend_quarantine(
        self,
        entry_id: str,
        additional_hours: int
    ) -> bool:
        """Extend quarantine duration."""
        if entry_id not in self._entries:
            return False
        
        entry = self._entries[entry_id]
        
        if entry.expires_at:
            entry.expires_at += timedelta(hours=additional_hours)
        else:
            entry.expires_at = datetime.now() + timedelta(hours=additional_hours)
        
        self._save_entry(entry)
        logger.info(f"Extended quarantine for {entry.ip_address} by {additional_hours} hours")
        
        return True
    
    async def check_expirations(self) -> List[QuarantineEntry]:
        """Check for and process expired quarantines."""
        expired = []
        
        for entry in list(self._entries.values()):
            if entry.status == QuarantineStatus.ACTIVE and entry.is_expired():
                await self.release(entry.entry_id, released_by="auto-expiration")
                entry.status = QuarantineStatus.EXPIRED
                expired.append(entry)
        
        return expired
    
    def get_entry(self, entry_id: str) -> Optional[QuarantineEntry]:
        """Get entry by ID."""
        return self._entries.get(entry_id)
    
    def get_entry_by_ip(self, ip_address: str) -> Optional[QuarantineEntry]:
        """Get active entry for an IP."""
        for entry in self._entries.values():
            if entry.ip_address == ip_address and entry.status == QuarantineStatus.ACTIVE:
                return entry
        return None
    
    def is_quarantined(self, ip_address: str) -> bool:
        """Check if IP is currently quarantined."""
        return self.get_entry_by_ip(ip_address) is not None
    
    def get_active_entries(self) -> List[QuarantineEntry]:
        """Get all active quarantine entries."""
        return [e for e in self._entries.values() if e.status == QuarantineStatus.ACTIVE]
    
    def get_all_entries(self, include_released: bool = False) -> List[QuarantineEntry]:
        """Get all entries."""
        if include_released:
            return list(self._entries.values())
        return [e for e in self._entries.values() 
                if e.status in (QuarantineStatus.ACTIVE, QuarantineStatus.PENDING)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quarantine statistics."""
        by_status = {}
        by_reason = {}
        
        for entry in self._entries.values():
            status = entry.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            reason = entry.reason.value
            by_reason[reason] = by_reason.get(reason, 0) + 1
        
        active = self.get_active_entries()
        
        return {
            'total_entries': len(self._entries),
            'active_count': len(active),
            'by_status': by_status,
            'by_reason': by_reason,
            'expiring_soon': sum(
                1 for e in active 
                if e.expires_at and e.expires_at - datetime.now() < timedelta(hours=1)
            )
        }


def create_quarantine_manager(
    firewall_manager: FirewallManager,
    db_path: str = "data/quarantine.db"
) -> QuarantineManager:
    """Create a quarantine manager."""
    return QuarantineManager(
        firewall_manager=firewall_manager,
        db_path=db_path
    )
