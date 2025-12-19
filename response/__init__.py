"""
Autonomous Response Package for AI-NIDS

This package provides automated threat response capabilities:
- Firewall integration (Windows netsh, Linux iptables, Cloud NSGs)
- Device quarantine
- Watchlist management
- SOC response protocols

Author: AI-NIDS Team
Version: 2.0.0 (Defense Mode)

⚠️ WARNING: Autonomous response actions can disrupt network connectivity.
Always test in a controlled environment before production deployment.
"""

from .firewall_manager import (
    FirewallManager,
    FirewallRule,
    FirewallAction,
    create_firewall_manager
)

from .response_engine import (
    ResponseEngine,
    ResponseAction,
    ResponseLevel,
    ResponseResult,
    create_response_engine
)

from .quarantine import (
    QuarantineManager,
    QuarantineEntry,
    QuarantineReason,
    create_quarantine_manager
)

from .soc_protocols import (
    SOCProtocol,
    IncidentTicket,
    EscalationLevel,
    PlaybookExecutor,
    create_playbook_executor
)

__all__ = [
    # Firewall Manager
    'FirewallManager',
    'FirewallRule',
    'FirewallAction',
    'create_firewall_manager',
    
    # Response Engine
    'ResponseEngine',
    'ResponseAction',
    'ResponseLevel',
    'ResponseResult',
    'create_response_engine',
    
    # Quarantine
    'QuarantineManager',
    'QuarantineEntry',
    'QuarantineReason',
    'create_quarantine_manager',
    
    # SOC Protocols
    'SOCProtocol',
    'IncidentTicket',
    'EscalationLevel',
    'PlaybookExecutor',
    'create_playbook_executor'
]

__version__ = '2.0.0'
__author__ = 'AI-NIDS Team'
