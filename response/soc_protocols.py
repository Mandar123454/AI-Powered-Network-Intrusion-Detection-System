"""
SOC Protocols for AI-NIDS

This module provides SOC (Security Operations Center) integration:
- Incident ticketing
- Escalation procedures
- Playbook execution
- Response automation

Author: AI-NIDS Team
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation levels for incidents."""
    L1_ANALYST = "l1_analyst"
    L2_ANALYST = "l2_analyst"
    L3_ENGINEER = "l3_engineer"
    INCIDENT_MANAGER = "incident_manager"
    CISO = "ciso"


class IncidentStatus(Enum):
    """Incident ticket status."""
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    RESOLVED = "resolved"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class IncidentPriority(Enum):
    """Incident priority levels."""
    P1_CRITICAL = "p1_critical"
    P2_HIGH = "p2_high"
    P3_MEDIUM = "p3_medium"
    P4_LOW = "p4_low"
    P5_INFO = "p5_info"


@dataclass
class IncidentTicket:
    """Represents a SOC incident ticket."""
    ticket_id: str
    title: str
    description: str
    priority: IncidentPriority
    status: IncidentStatus
    escalation_level: EscalationLevel
    
    # Affected entities
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    affected_assets: List[str] = field(default_factory=list)
    
    # Detection info
    detection_source: str = "AI-NIDS"
    threat_type: str = ""
    severity_score: float = 0.0
    confidence_score: float = 0.0
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Assignment
    assigned_to: Optional[str] = None
    
    # Evidence and notes
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    
    # Related tickets
    related_tickets: List[str] = field(default_factory=list)
    parent_ticket: Optional[str] = None
    
    # SLA tracking
    sla_breach_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ticket_id': self.ticket_id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'status': self.status.value,
            'escalation_level': self.escalation_level.value,
            'source_ip': self.source_ip,
            'destination_ip': self.destination_ip,
            'affected_assets': self.affected_assets,
            'detection_source': self.detection_source,
            'threat_type': self.threat_type,
            'severity_score': self.severity_score,
            'confidence_score': self.confidence_score,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'assigned_to': self.assigned_to,
            'evidence_count': len(self.evidence),
            'notes_count': len(self.notes),
            'actions_taken': self.actions_taken
        }
    
    def add_note(self, note: str, author: str = "system"):
        """Add a note to the ticket."""
        self.notes.append(f"[{datetime.now().isoformat()}] {author}: {note}")
        self.updated_at = datetime.now()
    
    def add_evidence(self, evidence_type: str, data: Any, description: str = ""):
        """Add evidence to the ticket."""
        self.evidence.append({
            'type': evidence_type,
            'data': data,
            'description': description,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def log_action(self, action: str, result: str, actor: str = "system"):
        """Log an action taken."""
        self.actions_taken.append({
            'action': action,
            'result': result,
            'actor': actor,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()


@dataclass
class SOCProtocol:
    """Defines a SOC response protocol."""
    protocol_id: str
    name: str
    description: str
    threat_types: List[str]
    min_severity: float
    max_severity: float
    steps: List[Dict[str, Any]]
    escalation_policy: Dict[str, int]  # Level -> minutes before escalation
    auto_actions: List[str]
    requires_approval: bool = False


class PlaybookStep(ABC):
    """Abstract base class for playbook steps."""
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step and return results."""
        pass
    
    @abstractmethod
    def rollback(self, context: Dict[str, Any]) -> bool:
        """Rollback the step if needed."""
        pass


class PlaybookExecutor:
    """
    Executes SOC playbooks for incident response.
    
    Features:
    - Step-by-step playbook execution
    - Conditional branching
    - Rollback support
    - Audit logging
    """
    
    # Default playbooks
    DEFAULT_PROTOCOLS = {
        'malware_detection': SOCProtocol(
            protocol_id='malware_detection',
            name='Malware Detection Response',
            description='Response procedure for detected malware',
            threat_types=['malware', 'ransomware', 'trojan'],
            min_severity=0.7,
            max_severity=1.0,
            steps=[
                {'action': 'isolate_host', 'required': True},
                {'action': 'capture_memory', 'required': False},
                {'action': 'collect_logs', 'required': True},
                {'action': 'scan_related_hosts', 'required': True},
                {'action': 'notify_user', 'required': False}
            ],
            escalation_policy={
                'l1_analyst': 0,
                'l2_analyst': 30,
                'incident_manager': 60
            },
            auto_actions=['isolate_host', 'collect_logs'],
            requires_approval=False
        ),
        'c2_communication': SOCProtocol(
            protocol_id='c2_communication',
            name='C2 Communication Response',
            description='Response to command and control traffic',
            threat_types=['c2', 'botnet', 'apt'],
            min_severity=0.8,
            max_severity=1.0,
            steps=[
                {'action': 'block_destination', 'required': True},
                {'action': 'isolate_host', 'required': True},
                {'action': 'capture_traffic', 'required': True},
                {'action': 'threat_hunt', 'required': True},
                {'action': 'notify_soc_manager', 'required': True}
            ],
            escalation_policy={
                'l2_analyst': 0,
                'l3_engineer': 15,
                'incident_manager': 30
            },
            auto_actions=['block_destination'],
            requires_approval=True
        ),
        'data_exfiltration': SOCProtocol(
            protocol_id='data_exfiltration',
            name='Data Exfiltration Response',
            description='Response to suspected data exfiltration',
            threat_types=['exfiltration', 'data_theft'],
            min_severity=0.9,
            max_severity=1.0,
            steps=[
                {'action': 'block_all_external', 'required': True},
                {'action': 'capture_traffic', 'required': True},
                {'action': 'notify_legal', 'required': True},
                {'action': 'preserve_evidence', 'required': True}
            ],
            escalation_policy={
                'incident_manager': 0,
                'ciso': 30
            },
            auto_actions=['block_all_external'],
            requires_approval=True
        ),
        'brute_force': SOCProtocol(
            protocol_id='brute_force',
            name='Brute Force Attack Response',
            description='Response to brute force attempts',
            threat_types=['brute_force', 'credential_stuffing'],
            min_severity=0.5,
            max_severity=0.8,
            steps=[
                {'action': 'rate_limit_source', 'required': True},
                {'action': 'check_successful_logins', 'required': True},
                {'action': 'notify_account_owners', 'required': False}
            ],
            escalation_policy={
                'l1_analyst': 0,
                'l2_analyst': 60
            },
            auto_actions=['rate_limit_source'],
            requires_approval=False
        )
    }
    
    def __init__(
        self,
        db_path: str = "data/soc_tickets.db",
        protocols: Optional[Dict[str, SOCProtocol]] = None
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.protocols = protocols or self.DEFAULT_PROTOCOLS
        
        # Ticket storage
        self._tickets: Dict[str, IncidentTicket] = {}
        self._ticket_counter = 0
        
        # Execution state
        self._running_playbooks: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks
        self._action_handlers: Dict[str, Callable] = {}
        self._escalation_callbacks: List[Callable] = []
        
        self._init_db()
    
    def _init_db(self):
        """Initialize database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at TEXT,
                    status TEXT,
                    priority TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS playbook_executions (
                    execution_id TEXT PRIMARY KEY,
                    ticket_id TEXT,
                    protocol_id TEXT,
                    status TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    steps_completed TEXT
                )
            """)
            conn.commit()
    
    def register_action_handler(self, action: str, handler: Callable):
        """Register a handler for a playbook action."""
        self._action_handlers[action] = handler
    
    def register_escalation_callback(self, callback: Callable):
        """Register callback for escalations."""
        self._escalation_callbacks.append(callback)
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID."""
        self._ticket_counter += 1
        return f"INC{datetime.now().strftime('%Y%m%d')}{self._ticket_counter:06d}"
    
    async def create_incident(
        self,
        title: str,
        description: str,
        threat_type: str,
        severity: float,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        evidence: Optional[List[Dict[str, Any]]] = None
    ) -> IncidentTicket:
        """
        Create a new incident ticket.
        
        Args:
            title: Incident title
            description: Detailed description
            threat_type: Type of threat
            severity: Severity score (0-1)
            source_ip: Source IP if applicable
            destination_ip: Destination IP if applicable
            evidence: Initial evidence
        
        Returns:
            Created IncidentTicket
        """
        # Determine priority from severity
        if severity >= 0.9:
            priority = IncidentPriority.P1_CRITICAL
        elif severity >= 0.7:
            priority = IncidentPriority.P2_HIGH
        elif severity >= 0.5:
            priority = IncidentPriority.P3_MEDIUM
        elif severity >= 0.3:
            priority = IncidentPriority.P4_LOW
        else:
            priority = IncidentPriority.P5_INFO
        
        # Determine initial escalation level
        if priority == IncidentPriority.P1_CRITICAL:
            escalation = EscalationLevel.L2_ANALYST
        else:
            escalation = EscalationLevel.L1_ANALYST
        
        ticket = IncidentTicket(
            ticket_id=self._generate_ticket_id(),
            title=title,
            description=description,
            priority=priority,
            status=IncidentStatus.NEW,
            escalation_level=escalation,
            source_ip=source_ip,
            destination_ip=destination_ip,
            threat_type=threat_type,
            severity_score=severity,
            evidence=evidence or []
        )
        
        # Calculate SLA breach time
        sla_hours = {
            IncidentPriority.P1_CRITICAL: 1,
            IncidentPriority.P2_HIGH: 4,
            IncidentPriority.P3_MEDIUM: 8,
            IncidentPriority.P4_LOW: 24,
            IncidentPriority.P5_INFO: 72
        }
        ticket.sla_breach_at = datetime.now() + timedelta(hours=sla_hours[priority])
        
        # Store ticket
        self._tickets[ticket.ticket_id] = ticket
        self._save_ticket(ticket)
        
        logger.info(f"Created incident {ticket.ticket_id}: {title}")
        
        # Auto-start playbook if applicable
        protocol = self._find_protocol(threat_type, severity)
        if protocol:
            asyncio.create_task(self.execute_playbook(ticket.ticket_id, protocol.protocol_id))
        
        return ticket
    
    def _find_protocol(self, threat_type: str, severity: float) -> Optional[SOCProtocol]:
        """Find applicable protocol for a threat."""
        for protocol in self.protocols.values():
            if (threat_type.lower() in [t.lower() for t in protocol.threat_types] and
                protocol.min_severity <= severity <= protocol.max_severity):
                return protocol
        return None
    
    async def execute_playbook(
        self,
        ticket_id: str,
        protocol_id: str,
        approved: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a playbook for an incident.
        
        Args:
            ticket_id: Ticket to execute for
            protocol_id: Protocol/playbook to execute
            approved: Whether manual approval was given
        
        Returns:
            Execution results
        """
        if protocol_id not in self.protocols:
            return {'error': f'Protocol {protocol_id} not found'}
        
        if ticket_id not in self._tickets:
            return {'error': f'Ticket {ticket_id} not found'}
        
        protocol = self.protocols[protocol_id]
        ticket = self._tickets[ticket_id]
        
        # Check if approval required
        if protocol.requires_approval and not approved:
            ticket.add_note(f"Playbook {protocol_id} requires approval before execution")
            return {'status': 'pending_approval', 'protocol': protocol_id}
        
        execution_id = f"EX{datetime.now().strftime('%Y%m%d%H%M%S')}"
        results = {
            'execution_id': execution_id,
            'protocol_id': protocol_id,
            'ticket_id': ticket_id,
            'steps': [],
            'status': 'running'
        }
        
        self._running_playbooks[execution_id] = results
        
        ticket.status = IncidentStatus.IN_PROGRESS
        ticket.add_note(f"Started playbook execution: {protocol.name}")
        
        # Execute steps
        for step in protocol.steps:
            action = step['action']
            required = step.get('required', True)
            
            # Check if auto-execution is allowed
            if action not in protocol.auto_actions and not approved:
                ticket.add_note(f"Skipping non-auto action: {action}")
                results['steps'].append({
                    'action': action,
                    'status': 'skipped',
                    'reason': 'requires_approval'
                })
                continue
            
            try:
                # Execute action
                step_result = await self._execute_action(action, ticket)
                results['steps'].append({
                    'action': action,
                    'status': 'success',
                    'result': step_result
                })
                ticket.log_action(action, str(step_result))
                
            except Exception as e:
                logger.error(f"Playbook step failed: {action} - {e}")
                results['steps'].append({
                    'action': action,
                    'status': 'failed',
                    'error': str(e)
                })
                
                if required:
                    results['status'] = 'failed'
                    ticket.add_note(f"Playbook failed at step: {action}")
                    break
        
        if results['status'] != 'failed':
            results['status'] = 'completed'
            ticket.add_note(f"Playbook completed: {protocol.name}")
        
        self._save_ticket(ticket)
        return results
    
    async def _execute_action(
        self,
        action: str,
        ticket: IncidentTicket
    ) -> Dict[str, Any]:
        """Execute a single playbook action."""
        if action in self._action_handlers:
            handler = self._action_handlers[action]
            if asyncio.iscoroutinefunction(handler):
                return await handler(ticket)
            return handler(ticket)
        
        # Default implementations
        if action == 'isolate_host':
            logger.info(f"[PLAYBOOK] Would isolate host: {ticket.source_ip}")
            return {'isolated': ticket.source_ip}
        
        elif action == 'block_destination':
            logger.info(f"[PLAYBOOK] Would block destination: {ticket.destination_ip}")
            return {'blocked': ticket.destination_ip}
        
        elif action == 'collect_logs':
            logger.info(f"[PLAYBOOK] Would collect logs for: {ticket.source_ip}")
            return {'logs_collected': True}
        
        elif action == 'capture_traffic':
            logger.info(f"[PLAYBOOK] Would capture traffic")
            return {'capture_started': True}
        
        elif action == 'notify_soc_manager':
            logger.info(f"[PLAYBOOK] Would notify SOC manager")
            return {'notified': True}
        
        else:
            logger.warning(f"Unknown action: {action}")
            return {'action': action, 'status': 'no_handler'}
    
    async def escalate(
        self,
        ticket_id: str,
        reason: str = ""
    ) -> bool:
        """Escalate an incident to the next level."""
        if ticket_id not in self._tickets:
            return False
        
        ticket = self._tickets[ticket_id]
        
        # Determine next level
        levels = list(EscalationLevel)
        current_idx = levels.index(ticket.escalation_level)
        
        if current_idx >= len(levels) - 1:
            logger.warning(f"Cannot escalate {ticket_id} - already at highest level")
            return False
        
        new_level = levels[current_idx + 1]
        ticket.escalation_level = new_level
        ticket.add_note(f"Escalated to {new_level.value}: {reason}")
        
        # Trigger callbacks
        for callback in self._escalation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ticket, new_level)
                else:
                    callback(ticket, new_level)
            except Exception as e:
                logger.error(f"Escalation callback error: {e}")
        
        self._save_ticket(ticket)
        logger.info(f"Escalated {ticket_id} to {new_level.value}")
        
        return True
    
    def resolve(
        self,
        ticket_id: str,
        resolution: str,
        resolved_by: str = "system"
    ) -> bool:
        """Resolve an incident."""
        if ticket_id not in self._tickets:
            return False
        
        ticket = self._tickets[ticket_id]
        ticket.status = IncidentStatus.RESOLVED
        ticket.resolved_at = datetime.now()
        ticket.add_note(f"Resolved by {resolved_by}: {resolution}")
        
        self._save_ticket(ticket)
        return True
    
    def mark_false_positive(
        self,
        ticket_id: str,
        reason: str
    ) -> bool:
        """Mark incident as false positive."""
        if ticket_id not in self._tickets:
            return False
        
        ticket = self._tickets[ticket_id]
        ticket.status = IncidentStatus.FALSE_POSITIVE
        ticket.add_note(f"Marked as false positive: {reason}")
        
        self._save_ticket(ticket)
        return True
    
    def _save_ticket(self, ticket: IncidentTicket):
        """Save ticket to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tickets
                (ticket_id, data, created_at, status, priority)
                VALUES (?, ?, ?, ?, ?)
            """, (
                ticket.ticket_id,
                json.dumps(ticket.to_dict()),
                ticket.created_at.isoformat(),
                ticket.status.value,
                ticket.priority.value
            ))
            conn.commit()
    
    def get_ticket(self, ticket_id: str) -> Optional[IncidentTicket]:
        """Get a ticket by ID."""
        return self._tickets.get(ticket_id)
    
    def get_tickets(
        self,
        status: Optional[IncidentStatus] = None,
        priority: Optional[IncidentPriority] = None,
        limit: int = 100
    ) -> List[IncidentTicket]:
        """Get tickets with optional filters."""
        tickets = list(self._tickets.values())
        
        if status:
            tickets = [t for t in tickets if t.status == status]
        
        if priority:
            tickets = [t for t in tickets if t.priority == priority]
        
        # Sort by created_at desc
        tickets.sort(key=lambda t: t.created_at, reverse=True)
        
        return tickets[:limit]
    
    def get_open_tickets(self) -> List[IncidentTicket]:
        """Get all open tickets."""
        open_statuses = {
            IncidentStatus.NEW,
            IncidentStatus.ASSIGNED,
            IncidentStatus.IN_PROGRESS,
            IncidentStatus.PENDING
        }
        return [t for t in self._tickets.values() if t.status in open_statuses]
    
    def get_sla_breaching(self) -> List[IncidentTicket]:
        """Get tickets at risk of SLA breach."""
        now = datetime.now()
        return [
            t for t in self.get_open_tickets()
            if t.sla_breach_at and t.sla_breach_at <= now + timedelta(hours=1)
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SOC statistics."""
        tickets = list(self._tickets.values())
        
        by_status = {}
        by_priority = {}
        
        for ticket in tickets:
            status = ticket.status.value
            by_status[status] = by_status.get(status, 0) + 1
            
            priority = ticket.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
        
        open_tickets = self.get_open_tickets()
        
        return {
            'total_tickets': len(tickets),
            'open_tickets': len(open_tickets),
            'by_status': by_status,
            'by_priority': by_priority,
            'sla_at_risk': len(self.get_sla_breaching()),
            'protocols_available': len(self.protocols),
            'running_playbooks': len(self._running_playbooks)
        }


def create_playbook_executor(
    db_path: str = "data/soc_tickets.db"
) -> PlaybookExecutor:
    """Create a playbook executor."""
    return PlaybookExecutor(db_path=db_path)
