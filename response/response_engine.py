"""
Response Engine for AI-NIDS

This module provides automated threat response:
- Severity-based response selection
- Response action execution
- Response history tracking
- Rollback capabilities

Author: AI-NIDS Team
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .firewall_manager import FirewallManager, FirewallRule, RuleDirection

logger = logging.getLogger(__name__)


class ResponseLevel(Enum):
    """Response severity levels."""
    NONE = "none"
    LOG_ONLY = "log_only"
    MONITOR = "monitor"
    RATE_LIMIT = "rate_limit"
    QUARANTINE = "quarantine"
    BLOCK = "block"
    ISOLATE = "isolate"


class ResponseAction(Enum):
    """Types of response actions."""
    LOG = "log"
    ALERT = "alert"
    WATCHLIST_ADD = "watchlist_add"
    RATE_LIMIT = "rate_limit"
    BLOCK_IP = "block_ip"
    BLOCK_PORT = "block_port"
    QUARANTINE_HOST = "quarantine_host"
    ISOLATE_SUBNET = "isolate_subnet"
    NOTIFY_SOC = "notify_soc"
    CREATE_TICKET = "create_ticket"
    CAPTURE_TRAFFIC = "capture_traffic"
    ROLLBACK = "rollback"


@dataclass
class ResponseResult:
    """Result of a response action."""
    action: ResponseAction
    success: bool
    target: str
    timestamp: datetime
    details: str = ""
    can_rollback: bool = False
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.value,
            'success': self.success,
            'target': self.target,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'can_rollback': self.can_rollback
        }


@dataclass
class ResponsePolicy:
    """Policy for automated responses."""
    name: str
    enabled: bool = True
    
    # Severity thresholds
    log_threshold: float = 0.0
    monitor_threshold: float = 0.3
    rate_limit_threshold: float = 0.5
    block_threshold: float = 0.7
    isolate_threshold: float = 0.9
    
    # Timing
    block_duration_hours: int = 24
    rate_limit_duration_hours: int = 1
    
    # Limits
    max_blocks_per_hour: int = 100
    require_confirmation_above: float = 0.95


class ResponseEngine:
    """
    Automated threat response engine.
    
    Maps threat severity to appropriate responses:
    - CRITICAL (0.9-1.0): Block IP immediately
    - HIGH (0.7-0.9): Quarantine + investigate
    - MEDIUM (0.5-0.7): Add to watchlist + rate limit
    - LOW (0.3-0.5): Monitor closely
    - INFO (0.0-0.3): Log only
    """
    
    # Default severity-to-response mapping
    DEFAULT_RESPONSE_MAP = {
        (0.9, 1.0): ResponseLevel.BLOCK,
        (0.7, 0.9): ResponseLevel.QUARANTINE,
        (0.5, 0.7): ResponseLevel.RATE_LIMIT,
        (0.3, 0.5): ResponseLevel.MONITOR,
        (0.0, 0.3): ResponseLevel.LOG_ONLY
    }
    
    def __init__(
        self,
        firewall_manager: FirewallManager,
        policy: Optional[ResponsePolicy] = None,
        auto_respond: bool = False  # Default to manual confirmation
    ):
        self.firewall = firewall_manager
        self.policy = policy or ResponsePolicy(name="default")
        self.auto_respond = auto_respond
        
        # Response history
        self._history: List[ResponseResult] = []
        self._max_history = 10000
        
        # Rate limiting
        self._blocks_this_hour: int = 0
        self._hour_start: datetime = datetime.now()
        
        # Watchlist
        self._watchlist: Dict[str, datetime] = {}
        
        # Callbacks
        self._pre_response_callbacks: List[Callable] = []
        self._post_response_callbacks: List[Callable] = []
    
    def register_pre_response(self, callback: Callable):
        """Register callback before response execution."""
        self._pre_response_callbacks.append(callback)
    
    def register_post_response(self, callback: Callable):
        """Register callback after response execution."""
        self._post_response_callbacks.append(callback)
    
    def determine_response_level(self, severity: float) -> ResponseLevel:
        """Determine appropriate response level for a severity score."""
        if not self.policy.enabled:
            return ResponseLevel.LOG_ONLY
        
        if severity >= self.policy.isolate_threshold:
            return ResponseLevel.ISOLATE
        elif severity >= self.policy.block_threshold:
            return ResponseLevel.BLOCK
        elif severity >= self.policy.rate_limit_threshold:
            return ResponseLevel.RATE_LIMIT
        elif severity >= self.policy.monitor_threshold:
            return ResponseLevel.MONITOR
        else:
            return ResponseLevel.LOG_ONLY
    
    async def respond_to_threat(
        self,
        target_ip: str,
        severity: float,
        threat_type: str,
        reason: str = "",
        force: bool = False
    ) -> List[ResponseResult]:
        """
        Execute automated response to a threat.
        
        Args:
            target_ip: IP address to respond to
            severity: Threat severity (0.0 - 1.0)
            threat_type: Type of threat detected
            reason: Reason for the response
            force: Force response even if auto_respond is False
        
        Returns:
            List of response results
        """
        results = []
        response_level = self.determine_response_level(severity)
        
        logger.info(f"Responding to threat: {target_ip}, severity={severity:.2f}, level={response_level.value}")
        
        # Check rate limiting
        self._check_rate_limit()
        if self._blocks_this_hour >= self.policy.max_blocks_per_hour:
            logger.warning("Block rate limit reached")
            results.append(ResponseResult(
                action=ResponseAction.LOG,
                success=True,
                target=target_ip,
                timestamp=datetime.now(),
                details="Rate limit reached - logging only"
            ))
            return results
        
        # Execute callbacks
        for callback in self._pre_response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(target_ip, severity, response_level)
                else:
                    callback(target_ip, severity, response_level)
            except Exception as e:
                logger.error(f"Pre-response callback error: {e}")
        
        # Check if confirmation required
        if severity >= self.policy.require_confirmation_above and not force:
            if not self.auto_respond:
                logger.info(f"High severity threat requires confirmation: {target_ip}")
                results.append(ResponseResult(
                    action=ResponseAction.ALERT,
                    success=True,
                    target=target_ip,
                    timestamp=datetime.now(),
                    details=f"Confirmation required for {response_level.value} action"
                ))
                return results
        
        # Execute response based on level
        if response_level == ResponseLevel.LOG_ONLY:
            results.append(await self._log_threat(target_ip, severity, threat_type))
        
        elif response_level == ResponseLevel.MONITOR:
            results.append(await self._log_threat(target_ip, severity, threat_type))
            results.append(await self._add_to_watchlist(target_ip))
        
        elif response_level == ResponseLevel.RATE_LIMIT:
            results.append(await self._log_threat(target_ip, severity, threat_type))
            results.append(await self._add_to_watchlist(target_ip))
            # Rate limiting would be implemented with iptables -m limit or similar
        
        elif response_level == ResponseLevel.QUARANTINE:
            results.append(await self._log_threat(target_ip, severity, threat_type))
            results.append(await self._add_to_watchlist(target_ip))
            results.append(await self._quarantine_host(target_ip, reason, severity))
        
        elif response_level == ResponseLevel.BLOCK:
            results.append(await self._log_threat(target_ip, severity, threat_type))
            results.append(await self._block_ip(target_ip, reason, severity))
            self._blocks_this_hour += 1
        
        elif response_level == ResponseLevel.ISOLATE:
            results.append(await self._log_threat(target_ip, severity, threat_type))
            results.append(await self._block_ip(target_ip, reason, severity))
            results.append(await self._notify_soc(target_ip, severity, threat_type))
            self._blocks_this_hour += 1
        
        # Store history
        for result in results:
            self._add_to_history(result)
        
        # Execute post callbacks
        for callback in self._post_response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(target_ip, results)
                else:
                    callback(target_ip, results)
            except Exception as e:
                logger.error(f"Post-response callback error: {e}")
        
        return results
    
    async def _log_threat(
        self,
        target_ip: str,
        severity: float,
        threat_type: str
    ) -> ResponseResult:
        """Log a threat without taking action."""
        logger.warning(f"THREAT: {target_ip} - {threat_type} (severity: {severity:.2f})")
        return ResponseResult(
            action=ResponseAction.LOG,
            success=True,
            target=target_ip,
            timestamp=datetime.now(),
            details=f"{threat_type} with severity {severity:.2f}"
        )
    
    async def _add_to_watchlist(self, target_ip: str) -> ResponseResult:
        """Add IP to watchlist."""
        self._watchlist[target_ip] = datetime.now()
        logger.info(f"Added to watchlist: {target_ip}")
        return ResponseResult(
            action=ResponseAction.WATCHLIST_ADD,
            success=True,
            target=target_ip,
            timestamp=datetime.now(),
            details="Added to watchlist"
        )
    
    async def _block_ip(
        self,
        target_ip: str,
        reason: str,
        severity: float
    ) -> ResponseResult:
        """Block an IP address."""
        duration = self.policy.block_duration_hours
        
        # Higher severity = longer block
        if severity > 0.9:
            duration = duration * 2
        
        rule = await self.firewall.block_ip(
            ip=target_ip,
            reason=reason or f"Auto-blocked: severity {severity:.2f}",
            duration_hours=duration,
            direction=RuleDirection.BOTH
        )
        
        if rule:
            return ResponseResult(
                action=ResponseAction.BLOCK_IP,
                success=True,
                target=target_ip,
                timestamp=datetime.now(),
                details=f"Blocked for {duration} hours",
                can_rollback=True,
                rollback_data={'rule_id': rule.rule_id}
            )
        else:
            return ResponseResult(
                action=ResponseAction.BLOCK_IP,
                success=False,
                target=target_ip,
                timestamp=datetime.now(),
                details="Failed to create block rule"
            )
    
    async def _quarantine_host(
        self,
        target_ip: str,
        reason: str,
        severity: float
    ) -> ResponseResult:
        """Quarantine a host (block all except management traffic)."""
        # Block all traffic
        block_result = await self._block_ip(target_ip, reason, severity)
        
        if block_result.success:
            # In a real implementation, we'd also:
            # 1. Allow only management ports (SSH, RDP)
            # 2. Notify the host owner
            # 3. Create a ticket
            
            return ResponseResult(
                action=ResponseAction.QUARANTINE_HOST,
                success=True,
                target=target_ip,
                timestamp=datetime.now(),
                details=f"Host quarantined: {reason}",
                can_rollback=True,
                rollback_data=block_result.rollback_data
            )
        
        return ResponseResult(
            action=ResponseAction.QUARANTINE_HOST,
            success=False,
            target=target_ip,
            timestamp=datetime.now(),
            details="Failed to quarantine host"
        )
    
    async def _notify_soc(
        self,
        target_ip: str,
        severity: float,
        threat_type: str
    ) -> ResponseResult:
        """Notify SOC team of critical threat."""
        # In production, this would:
        # 1. Send email/SMS to on-call
        # 2. Create PagerDuty incident
        # 3. Post to Slack/Teams
        # 4. Create SIEM alert
        
        logger.critical(f"SOC NOTIFICATION: Critical threat from {target_ip} - {threat_type}")
        
        return ResponseResult(
            action=ResponseAction.NOTIFY_SOC,
            success=True,
            target=target_ip,
            timestamp=datetime.now(),
            details=f"SOC notified of {threat_type}"
        )
    
    async def rollback(self, result: ResponseResult) -> bool:
        """Rollback a response action."""
        if not result.can_rollback:
            return False
        
        if result.action == ResponseAction.BLOCK_IP:
            rule_id = result.rollback_data.get('rule_id')
            if rule_id:
                success = await self.firewall.remove_rule(rule_id)
                if success:
                    logger.info(f"Rolled back block for {result.target}")
                return success
        
        elif result.action == ResponseAction.QUARANTINE_HOST:
            return await self.rollback(ResponseResult(
                action=ResponseAction.BLOCK_IP,
                success=True,
                target=result.target,
                timestamp=datetime.now(),
                can_rollback=True,
                rollback_data=result.rollback_data
            ))
        
        return False
    
    def _check_rate_limit(self):
        """Reset rate limit counter if hour has passed."""
        now = datetime.now()
        if now - self._hour_start > timedelta(hours=1):
            self._blocks_this_hour = 0
            self._hour_start = now
    
    def _add_to_history(self, result: ResponseResult):
        """Add result to history."""
        self._history.append(result)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
    
    def is_on_watchlist(self, ip: str) -> bool:
        """Check if IP is on watchlist."""
        return ip in self._watchlist
    
    def get_watchlist(self) -> Dict[str, datetime]:
        """Get the current watchlist."""
        return self._watchlist.copy()
    
    def remove_from_watchlist(self, ip: str) -> bool:
        """Remove IP from watchlist."""
        if ip in self._watchlist:
            del self._watchlist[ip]
            return True
        return False
    
    def get_history(
        self,
        target: Optional[str] = None,
        action: Optional[ResponseAction] = None,
        limit: int = 100
    ) -> List[ResponseResult]:
        """Get response history with optional filters."""
        results = self._history
        
        if target:
            results = [r for r in results if r.target == target]
        
        if action:
            results = [r for r in results if r.action == action]
        
        return results[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        action_counts = {}
        for result in self._history:
            action = result.action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_responses': len(self._history),
            'successful_responses': sum(1 for r in self._history if r.success),
            'by_action': action_counts,
            'watchlist_size': len(self._watchlist),
            'blocks_this_hour': self._blocks_this_hour,
            'auto_respond': self.auto_respond,
            'policy_enabled': self.policy.enabled
        }


def create_response_engine(
    firewall_manager: FirewallManager,
    auto_respond: bool = False
) -> ResponseEngine:
    """Create a response engine."""
    return ResponseEngine(
        firewall_manager=firewall_manager,
        auto_respond=auto_respond
    )
