"""
Firewall Manager for AI-NIDS

This module provides cross-platform firewall management:
- Windows: netsh advfirewall
- Linux: iptables/nftables/ufw
- Cloud: Azure NSG, AWS Security Groups

Author: AI-NIDS Team

⚠️ SECURITY WARNING:
This module executes system commands with elevated privileges.
Improper use can lock you out of systems or disrupt network connectivity.
Always test in a controlled environment first.
"""

import asyncio
import ipaddress
import json
import logging
import os
import platform
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class FirewallAction(Enum):
    """Firewall rule actions."""
    BLOCK = "block"
    ALLOW = "allow"
    LOG = "log"
    RATE_LIMIT = "rate_limit"


class RuleDirection(Enum):
    """Traffic direction for rules."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BOTH = "both"


class Protocol(Enum):
    """Network protocols."""
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"
    ANY = "any"


@dataclass
class FirewallRule:
    """Represents a firewall rule."""
    rule_id: str
    name: str
    action: FirewallAction
    direction: RuleDirection
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: Protocol = Protocol.ANY
    enabled: bool = True
    priority: int = 100
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    reason: str = ""
    auto_generated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'rule_id': self.rule_id,
            'name': self.name,
            'action': self.action.value,
            'direction': self.direction.value,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'source_port': self.source_port,
            'destination_port': self.destination_port,
            'protocol': self.protocol.value,
            'enabled': self.enabled,
            'priority': self.priority,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'reason': self.reason,
            'auto_generated': self.auto_generated
        }
    
    def is_expired(self) -> bool:
        """Check if the rule has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class FirewallBackend(ABC):
    """Abstract base class for firewall backends."""
    
    @abstractmethod
    async def add_rule(self, rule: FirewallRule) -> bool:
        """Add a firewall rule."""
        pass
    
    @abstractmethod
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a firewall rule."""
        pass
    
    @abstractmethod
    async def list_rules(self) -> List[FirewallRule]:
        """List all managed rules."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this backend is available."""
        pass


class WindowsFirewall(FirewallBackend):
    """Windows Firewall backend using netsh."""
    
    RULE_PREFIX = "AINIDS_"
    
    async def is_available(self) -> bool:
        """Check if Windows Firewall is available."""
        if platform.system() != 'Windows':
            return False
        try:
            result = await asyncio.create_subprocess_exec(
                'netsh', 'advfirewall', 'show', 'currentprofile',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def add_rule(self, rule: FirewallRule) -> bool:
        """Add a Windows Firewall rule."""
        rule_name = f"{self.RULE_PREFIX}{rule.rule_id}"
        
        cmd = [
            'netsh', 'advfirewall', 'firewall', 'add', 'rule',
            f'name={rule_name}',
            f'dir={"in" if rule.direction == RuleDirection.INBOUND else "out"}',
            f'action={"block" if rule.action == FirewallAction.BLOCK else "allow"}',
        ]
        
        # Add protocol
        if rule.protocol != Protocol.ANY:
            cmd.append(f'protocol={rule.protocol.value}')
        
        # Add remote IP
        if rule.source_ip and rule.direction == RuleDirection.INBOUND:
            cmd.append(f'remoteip={rule.source_ip}')
        elif rule.destination_ip and rule.direction == RuleDirection.OUTBOUND:
            cmd.append(f'remoteip={rule.destination_ip}')
        
        # Add port
        if rule.destination_port:
            cmd.append(f'remoteport={rule.destination_port}')
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"Added Windows Firewall rule: {rule_name}")
                return True
            else:
                logger.error(f"Failed to add rule: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Error adding Windows Firewall rule: {e}")
            return False
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a Windows Firewall rule."""
        rule_name = f"{self.RULE_PREFIX}{rule_id}"
        
        cmd = [
            'netsh', 'advfirewall', 'firewall', 'delete', 'rule',
            f'name={rule_name}'
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            
            if result.returncode == 0:
                logger.info(f"Removed Windows Firewall rule: {rule_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing Windows Firewall rule: {e}")
            return False
    
    async def list_rules(self) -> List[FirewallRule]:
        """List all AI-NIDS managed rules."""
        # netsh output parsing is complex; return empty for now
        # In production, you'd parse: netsh advfirewall firewall show rule name=all
        return []


class LinuxFirewall(FirewallBackend):
    """Linux Firewall backend using iptables."""
    
    CHAIN_NAME = "AINIDS"
    
    async def is_available(self) -> bool:
        """Check if iptables is available."""
        if platform.system() != 'Linux':
            return False
        try:
            result = await asyncio.create_subprocess_exec(
                'iptables', '--version',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception:
            return False
    
    async def _ensure_chain(self):
        """Ensure our custom chain exists."""
        # Create chain if not exists
        create_cmd = ['iptables', '-N', self.CHAIN_NAME]
        await asyncio.create_subprocess_exec(
            *create_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        
        # Add jump rule if not exists
        check_cmd = ['iptables', '-C', 'INPUT', '-j', self.CHAIN_NAME]
        result = await asyncio.create_subprocess_exec(
            *check_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await result.wait()
        
        if result.returncode != 0:
            add_cmd = ['iptables', '-A', 'INPUT', '-j', self.CHAIN_NAME]
            await asyncio.create_subprocess_exec(*add_cmd)
    
    async def add_rule(self, rule: FirewallRule) -> bool:
        """Add an iptables rule."""
        await self._ensure_chain()
        
        cmd = ['iptables', '-A', self.CHAIN_NAME]
        
        # Source IP
        if rule.source_ip:
            cmd.extend(['-s', rule.source_ip])
        
        # Destination IP
        if rule.destination_ip:
            cmd.extend(['-d', rule.destination_ip])
        
        # Protocol
        if rule.protocol != Protocol.ANY:
            cmd.extend(['-p', rule.protocol.value])
            
            # Port (only for TCP/UDP)
            if rule.destination_port and rule.protocol in (Protocol.TCP, Protocol.UDP):
                cmd.extend(['--dport', str(rule.destination_port)])
        
        # Action
        action = 'DROP' if rule.action == FirewallAction.BLOCK else 'ACCEPT'
        cmd.extend(['-j', action])
        
        # Add comment with rule ID
        cmd.extend(['-m', 'comment', '--comment', f'AINIDS_{rule.rule_id}'])
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"Added iptables rule: {rule.rule_id}")
                return True
            else:
                logger.error(f"Failed to add rule: {stderr.decode()}")
                return False
        except Exception as e:
            logger.error(f"Error adding iptables rule: {e}")
            return False
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove an iptables rule by comment."""
        # Find and remove rule with matching comment
        cmd = [
            'iptables', '-D', self.CHAIN_NAME,
            '-m', 'comment', '--comment', f'AINIDS_{rule_id}'
        ]
        
        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error removing iptables rule: {e}")
            return False
    
    async def list_rules(self) -> List[FirewallRule]:
        """List AI-NIDS managed rules."""
        return []


class AzureNSGFirewall(FirewallBackend):
    """Azure Network Security Group backend."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        nsg_name: str,
        credential: Optional[Any] = None
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.nsg_name = nsg_name
        self.credential = credential
        self._client = None
    
    async def is_available(self) -> bool:
        """Check if Azure SDK is available."""
        try:
            from azure.mgmt.network import NetworkManagementClient
            from azure.identity import DefaultAzureCredential
            
            if self.credential is None:
                self.credential = DefaultAzureCredential()
            
            self._client = NetworkManagementClient(
                self.credential,
                self.subscription_id
            )
            return True
        except ImportError:
            logger.warning("Azure SDK not installed")
            return False
        except Exception as e:
            logger.error(f"Azure authentication error: {e}")
            return False
    
    async def add_rule(self, rule: FirewallRule) -> bool:
        """Add an Azure NSG rule."""
        if self._client is None:
            return False
        
        try:
            from azure.mgmt.network.models import SecurityRule
            
            # Get current NSG
            nsg = self._client.network_security_groups.get(
                self.resource_group,
                self.nsg_name
            )
            
            # Create new rule
            nsg_rule = SecurityRule(
                name=f"AINIDS_{rule.rule_id}",
                protocol=rule.protocol.value if rule.protocol != Protocol.ANY else '*',
                source_address_prefix=rule.source_ip or '*',
                destination_address_prefix=rule.destination_ip or '*',
                source_port_range='*',
                destination_port_range=str(rule.destination_port) if rule.destination_port else '*',
                access='Deny' if rule.action == FirewallAction.BLOCK else 'Allow',
                priority=rule.priority,
                direction='Inbound' if rule.direction == RuleDirection.INBOUND else 'Outbound',
                description=rule.reason or f"AI-NIDS auto-generated rule"
            )
            
            # Add to NSG
            nsg.security_rules.append(nsg_rule)
            
            # Update NSG
            self._client.network_security_groups.begin_create_or_update(
                self.resource_group,
                self.nsg_name,
                nsg
            )
            
            logger.info(f"Added Azure NSG rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding Azure NSG rule: {e}")
            return False
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove an Azure NSG rule."""
        if self._client is None:
            return False
        
        try:
            nsg = self._client.network_security_groups.get(
                self.resource_group,
                self.nsg_name
            )
            
            rule_name = f"AINIDS_{rule_id}"
            nsg.security_rules = [
                r for r in nsg.security_rules
                if r.name != rule_name
            ]
            
            self._client.network_security_groups.begin_create_or_update(
                self.resource_group,
                self.nsg_name,
                nsg
            )
            
            logger.info(f"Removed Azure NSG rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing Azure NSG rule: {e}")
            return False
    
    async def list_rules(self) -> List[FirewallRule]:
        """List AI-NIDS managed NSG rules."""
        return []


class FirewallManager:
    """
    Cross-platform firewall manager.
    
    Automatically selects the appropriate backend based on the platform
    and provides a unified interface for firewall management.
    """
    
    def __init__(
        self,
        db_path: str = "data/firewall_rules.db",
        dry_run: bool = False
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        
        self._backend: Optional[FirewallBackend] = None
        self._rules: Dict[str, FirewallRule] = {}
        self._rule_counter = 0
        
        self._init_db()
    
    def _init_db(self):
        """Initialize the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS firewall_rules (
                    rule_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT,
                    expires_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rule_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_id TEXT,
                    action TEXT,
                    timestamp TEXT,
                    success INTEGER,
                    details TEXT
                )
            """)
            conn.commit()
    
    async def initialize(self):
        """Initialize the firewall manager and select backend."""
        # Try backends in order of preference
        backends = [
            WindowsFirewall(),
            LinuxFirewall(),
        ]
        
        for backend in backends:
            if await backend.is_available():
                self._backend = backend
                logger.info(f"Using firewall backend: {backend.__class__.__name__}")
                break
        
        if self._backend is None:
            logger.warning("No firewall backend available - running in log-only mode")
        
        # Load existing rules from database
        self._load_rules()
    
    def _load_rules(self):
        """Load rules from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT rule_id, data FROM firewall_rules")
            for row in cursor:
                data = json.loads(row[1])
                rule = FirewallRule(
                    rule_id=data['rule_id'],
                    name=data['name'],
                    action=FirewallAction(data['action']),
                    direction=RuleDirection(data['direction']),
                    source_ip=data.get('source_ip'),
                    destination_ip=data.get('destination_ip'),
                    source_port=data.get('source_port'),
                    destination_port=data.get('destination_port'),
                    protocol=Protocol(data.get('protocol', 'any')),
                    enabled=data.get('enabled', True),
                    priority=data.get('priority', 100),
                    reason=data.get('reason', ''),
                    auto_generated=data.get('auto_generated', False)
                )
                if data.get('expires_at'):
                    rule.expires_at = datetime.fromisoformat(data['expires_at'])
                if data.get('created_at'):
                    rule.created_at = datetime.fromisoformat(data['created_at'])
                self._rules[rule.rule_id] = rule
    
    def _save_rule(self, rule: FirewallRule):
        """Save a rule to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO firewall_rules (rule_id, data, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            """, (
                rule.rule_id,
                json.dumps(rule.to_dict()),
                rule.created_at.isoformat(),
                rule.expires_at.isoformat() if rule.expires_at else None
            ))
            conn.commit()
    
    def _log_action(self, rule_id: str, action: str, success: bool, details: str = ""):
        """Log a firewall action."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO rule_history (rule_id, action, timestamp, success, details)
                VALUES (?, ?, ?, ?, ?)
            """, (rule_id, action, datetime.now().isoformat(), int(success), details))
            conn.commit()
    
    def _generate_rule_id(self) -> str:
        """Generate a unique rule ID."""
        self._rule_counter += 1
        return f"R{datetime.now().strftime('%Y%m%d%H%M%S')}{self._rule_counter:04d}"
    
    async def block_ip(
        self,
        ip: str,
        reason: str = "",
        duration_hours: Optional[int] = None,
        direction: RuleDirection = RuleDirection.BOTH
    ) -> Optional[FirewallRule]:
        """
        Block an IP address.
        
        Args:
            ip: IP address to block
            reason: Reason for blocking
            duration_hours: How long to block (None = permanent)
            direction: Traffic direction to block
        
        Returns:
            The created firewall rule, or None on failure
        """
        # Validate IP
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            logger.error(f"Invalid IP address: {ip}")
            return None
        
        rule = FirewallRule(
            rule_id=self._generate_rule_id(),
            name=f"Block {ip}",
            action=FirewallAction.BLOCK,
            direction=direction,
            source_ip=ip if direction in (RuleDirection.INBOUND, RuleDirection.BOTH) else None,
            destination_ip=ip if direction in (RuleDirection.OUTBOUND, RuleDirection.BOTH) else None,
            reason=reason,
            auto_generated=True
        )
        
        if duration_hours:
            rule.expires_at = datetime.now() + timedelta(hours=duration_hours)
        
        success = await self._apply_rule(rule)
        
        if success:
            self._rules[rule.rule_id] = rule
            self._save_rule(rule)
            self._log_action(rule.rule_id, 'block_ip', True, f"Blocked {ip}")
            return rule
        
        self._log_action(rule.rule_id, 'block_ip', False, f"Failed to block {ip}")
        return None
    
    async def unblock_ip(self, ip: str) -> bool:
        """Unblock an IP address."""
        # Find rules for this IP
        rules_to_remove = [
            r for r in self._rules.values()
            if r.source_ip == ip or r.destination_ip == ip
        ]
        
        success = True
        for rule in rules_to_remove:
            if not await self.remove_rule(rule.rule_id):
                success = False
        
        return success
    
    async def block_port(
        self,
        port: int,
        protocol: Protocol = Protocol.TCP,
        reason: str = "",
        direction: RuleDirection = RuleDirection.INBOUND
    ) -> Optional[FirewallRule]:
        """Block a specific port."""
        rule = FirewallRule(
            rule_id=self._generate_rule_id(),
            name=f"Block port {port}/{protocol.value}",
            action=FirewallAction.BLOCK,
            direction=direction,
            destination_port=port,
            protocol=protocol,
            reason=reason,
            auto_generated=True
        )
        
        success = await self._apply_rule(rule)
        
        if success:
            self._rules[rule.rule_id] = rule
            self._save_rule(rule)
            return rule
        
        return None
    
    async def _apply_rule(self, rule: FirewallRule) -> bool:
        """Apply a rule using the backend."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would apply rule: {rule.to_dict()}")
            return True
        
        if self._backend is None:
            logger.warning(f"No backend - logging rule only: {rule.to_dict()}")
            return True
        
        # Handle BOTH direction by creating two rules
        if rule.direction == RuleDirection.BOTH:
            in_rule = FirewallRule(
                rule_id=f"{rule.rule_id}_IN",
                name=f"{rule.name} (Inbound)",
                action=rule.action,
                direction=RuleDirection.INBOUND,
                source_ip=rule.source_ip or rule.destination_ip,
                destination_port=rule.destination_port,
                protocol=rule.protocol,
                reason=rule.reason,
                auto_generated=rule.auto_generated
            )
            out_rule = FirewallRule(
                rule_id=f"{rule.rule_id}_OUT",
                name=f"{rule.name} (Outbound)",
                action=rule.action,
                direction=RuleDirection.OUTBOUND,
                destination_ip=rule.source_ip or rule.destination_ip,
                destination_port=rule.destination_port,
                protocol=rule.protocol,
                reason=rule.reason,
                auto_generated=rule.auto_generated
            )
            
            in_success = await self._backend.add_rule(in_rule)
            out_success = await self._backend.add_rule(out_rule)
            return in_success and out_success
        
        return await self._backend.add_rule(rule)
    
    async def remove_rule(self, rule_id: str) -> bool:
        """Remove a firewall rule."""
        if rule_id not in self._rules:
            return False
        
        rule = self._rules[rule_id]
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would remove rule: {rule_id}")
            success = True
        elif self._backend:
            success = await self._backend.remove_rule(rule_id)
            
            # Handle BOTH direction
            if rule.direction == RuleDirection.BOTH:
                await self._backend.remove_rule(f"{rule_id}_IN")
                await self._backend.remove_rule(f"{rule_id}_OUT")
        else:
            success = True
        
        if success:
            del self._rules[rule_id]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM firewall_rules WHERE rule_id = ?", (rule_id,))
                conn.commit()
            self._log_action(rule_id, 'remove', True)
        
        return success
    
    async def cleanup_expired(self) -> int:
        """Remove expired rules."""
        expired = [r for r in self._rules.values() if r.is_expired()]
        removed = 0
        
        for rule in expired:
            if await self.remove_rule(rule.rule_id):
                removed += 1
                logger.info(f"Removed expired rule: {rule.rule_id}")
        
        return removed
    
    def get_rules(self) -> List[FirewallRule]:
        """Get all managed rules."""
        return list(self._rules.values())
    
    def get_rule(self, rule_id: str) -> Optional[FirewallRule]:
        """Get a specific rule."""
        return self._rules.get(rule_id)
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if an IP is currently blocked."""
        for rule in self._rules.values():
            if rule.action == FirewallAction.BLOCK:
                if rule.source_ip == ip or rule.destination_ip == ip:
                    if not rule.is_expired():
                        return True
        return False
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get rule action history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM rule_history ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get firewall statistics."""
        return {
            'total_rules': len(self._rules),
            'block_rules': sum(1 for r in self._rules.values() if r.action == FirewallAction.BLOCK),
            'allow_rules': sum(1 for r in self._rules.values() if r.action == FirewallAction.ALLOW),
            'auto_generated': sum(1 for r in self._rules.values() if r.auto_generated),
            'expiring_rules': sum(1 for r in self._rules.values() if r.expires_at is not None),
            'backend': self._backend.__class__.__name__ if self._backend else 'None',
            'dry_run': self.dry_run
        }


def create_firewall_manager(
    db_path: str = "data/firewall_rules.db",
    dry_run: bool = True  # Default to dry run for safety
) -> FirewallManager:
    """Create a firewall manager."""
    return FirewallManager(db_path=db_path, dry_run=dry_run)
