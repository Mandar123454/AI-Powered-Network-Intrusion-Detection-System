"""
AI-NIDS Notification System
===========================
Multi-channel alert notifications (Email, Slack, Telegram, Webhooks)
"""

import os
import json
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import requests
from threading import Thread
from queue import Queue

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    DISCORD = "discord"
    PAGERDUTY = "pagerduty"


class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    alert_id: Optional[str] = None
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    attack_type: Optional[str] = None
    severity: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'title': self.title,
            'body': self.body,
            'priority': self.priority.name,
            'alert_id': self.alert_id,
            'source_ip': self.source_ip,
            'dest_ip': self.dest_ip,
            'attack_type': self.attack_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class NotificationProvider(ABC):
    """Base class for notification providers"""
    
    @abstractmethod
    def send(self, message: NotificationMessage) -> bool:
        """Send notification"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass


class EmailNotificationProvider(NotificationProvider):
    """Email notification provider using SMTP"""
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = 587,
        username: str = None,
        password: str = None,
        from_email: str = None,
        recipients: List[str] = None,
        use_tls: bool = True
    ):
        self.smtp_server = smtp_server or os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.environ.get('SMTP_PORT', 587))
        self.username = username or os.environ.get('SMTP_USERNAME')
        self.password = password or os.environ.get('SMTP_PASSWORD')
        self.from_email = from_email or os.environ.get('SMTP_FROM_EMAIL', self.username)
        self.recipients = recipients or os.environ.get('ALERT_EMAIL_RECIPIENTS', '').split(',')
        self.use_tls = use_tls
    
    def validate_config(self) -> bool:
        """Validate email configuration"""
        return all([
            self.smtp_server,
            self.smtp_port,
            self.username,
            self.password,
            self.from_email,
            self.recipients
        ])
    
    def send(self, message: NotificationMessage) -> bool:
        """Send email notification"""
        if not self.validate_config():
            logger.warning("Email notification not configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[AI-NIDS] {message.priority.name}: {message.title}"
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.recipients)
            
            # Plain text version
            text_body = self._format_text_body(message)
            msg.attach(MIMEText(text_body, 'plain'))
            
            # HTML version
            html_body = self._format_html_body(message)
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_email, self.recipients, msg.as_string())
            
            logger.info(f"Email notification sent to {len(self.recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _format_text_body(self, message: NotificationMessage) -> str:
        """Format plain text email body"""
        return f"""
AI-NIDS Security Alert
======================

{message.title}

{message.body}

Details:
--------
Alert ID: {message.alert_id or 'N/A'}
Source IP: {message.source_ip or 'N/A'}
Destination IP: {message.dest_ip or 'N/A'}
Attack Type: {message.attack_type or 'N/A'}
Severity: {message.severity or 'N/A'}
Confidence: {f'{message.confidence:.1%}' if message.confidence else 'N/A'}
Timestamp: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

---
This is an automated alert from AI-NIDS.
"""
    
    def _format_html_body(self, message: NotificationMessage) -> str:
        """Format HTML email body"""
        severity_colors = {
            'critical': '#dc3545',
            'high': '#fd7e14',
            'medium': '#ffc107',
            'low': '#17a2b8',
            'info': '#6c757d'
        }
        color = severity_colors.get(message.severity, '#6c757d')
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; background-color: #1a1a2e; color: #eee; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, {color}, #16213e); padding: 20px; border-radius: 10px 10px 0 0; }}
        .header h1 {{ margin: 0; font-size: 24px; }}
        .content {{ background: #16213e; padding: 20px; border-radius: 0 0 10px 10px; }}
        .alert-badge {{ display: inline-block; background: {color}; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .details {{ background: #1a1a2e; padding: 15px; border-radius: 8px; margin-top: 15px; }}
        .detail-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }}
        .detail-label {{ color: #888; }}
        .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AI-NIDS Security Alert</h1>
            <span class="alert-badge">{message.priority.name}</span>
        </div>
        <div class="content">
            <h2>{message.title}</h2>
            <p>{message.body}</p>
            
            <div class="details">
                <div class="detail-row">
                    <span class="detail-label">Alert ID</span>
                    <span>{message.alert_id or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Source IP</span>
                    <span>{message.source_ip or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Destination IP</span>
                    <span>{message.dest_ip or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Attack Type</span>
                    <span>{message.attack_type or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Severity</span>
                    <span style="color: {color}; font-weight: bold;">{message.severity or 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Confidence</span>
                    <span>{f'{message.confidence:.1%}' if message.confidence else 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Timestamp</span>
                    <span>{message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                </div>
            </div>
        </div>
        <div class="footer">
            This is an automated alert from AI-NIDS | Network Intrusion Detection System
        </div>
    </div>
</body>
</html>
"""


class SlackNotificationProvider(NotificationProvider):
    """Slack notification provider using webhooks"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
    
    def validate_config(self) -> bool:
        """Validate Slack configuration"""
        return bool(self.webhook_url)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send Slack notification"""
        if not self.validate_config():
            logger.warning("Slack notification not configured")
            return False
        
        try:
            severity_colors = {
                'critical': '#dc3545',
                'high': '#fd7e14',
                'medium': '#ffc107',
                'low': '#17a2b8',
                'info': '#6c757d'
            }
            color = severity_colors.get(message.severity, '#6c757d')
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🛡️ {message.title}",
                    "text": message.body,
                    "fields": [
                        {"title": "Severity", "value": message.severity or "N/A", "short": True},
                        {"title": "Attack Type", "value": message.attack_type or "N/A", "short": True},
                        {"title": "Source IP", "value": message.source_ip or "N/A", "short": True},
                        {"title": "Destination IP", "value": message.dest_ip or "N/A", "short": True},
                        {"title": "Confidence", "value": f"{message.confidence:.1%}" if message.confidence else "N/A", "short": True},
                        {"title": "Alert ID", "value": message.alert_id or "N/A", "short": True}
                    ],
                    "footer": "AI-NIDS",
                    "ts": int(message.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class TelegramNotificationProvider(NotificationProvider):
    """Telegram notification provider using Bot API"""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.environ.get('TELEGRAM_CHAT_ID')
    
    def validate_config(self) -> bool:
        """Validate Telegram configuration"""
        return bool(self.bot_token and self.chat_id)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send Telegram notification"""
        if not self.validate_config():
            logger.warning("Telegram notification not configured")
            return False
        
        try:
            severity_emoji = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🔵',
                'info': '⚪'
            }
            emoji = severity_emoji.get(message.severity, '⚪')
            
            text = f"""
{emoji} *AI-NIDS Alert: {message.title}*

{message.body}

📊 *Details:*
• Severity: `{message.severity or 'N/A'}`
• Attack Type: `{message.attack_type or 'N/A'}`
• Source: `{message.source_ip or 'N/A'}`
• Destination: `{message.dest_ip or 'N/A'}`
• Confidence: `{f'{message.confidence:.1%}' if message.confidence else 'N/A'}`
• Time: `{message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}`
"""
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Telegram notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


class DiscordNotificationProvider(NotificationProvider):
    """Discord notification provider using webhooks"""
    
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.environ.get('DISCORD_WEBHOOK_URL')
    
    def validate_config(self) -> bool:
        """Validate Discord configuration"""
        return bool(self.webhook_url)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send Discord notification"""
        if not self.validate_config():
            logger.warning("Discord notification not configured")
            return False
        
        try:
            severity_colors = {
                'critical': 0xdc3545,
                'high': 0xfd7e14,
                'medium': 0xffc107,
                'low': 0x17a2b8,
                'info': 0x6c757d
            }
            color = severity_colors.get(message.severity, 0x6c757d)
            
            payload = {
                "embeds": [{
                    "title": f"🛡️ {message.title}",
                    "description": message.body,
                    "color": color,
                    "fields": [
                        {"name": "Severity", "value": message.severity or "N/A", "inline": True},
                        {"name": "Attack Type", "value": message.attack_type or "N/A", "inline": True},
                        {"name": "Source IP", "value": message.source_ip or "N/A", "inline": True},
                        {"name": "Destination IP", "value": message.dest_ip or "N/A", "inline": True},
                        {"name": "Confidence", "value": f"{message.confidence:.1%}" if message.confidence else "N/A", "inline": True},
                        {"name": "Alert ID", "value": message.alert_id or "N/A", "inline": True}
                    ],
                    "footer": {"text": "AI-NIDS"},
                    "timestamp": message.timestamp.isoformat()
                }]
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info("Discord notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False


class WebhookNotificationProvider(NotificationProvider):
    """Generic webhook notification provider"""
    
    def __init__(
        self,
        webhook_url: str = None,
        headers: Dict[str, str] = None,
        method: str = "POST"
    ):
        self.webhook_url = webhook_url or os.environ.get('WEBHOOK_URL')
        self.headers = headers or {"Content-Type": "application/json"}
        self.method = method
    
    def validate_config(self) -> bool:
        """Validate webhook configuration"""
        return bool(self.webhook_url)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send webhook notification"""
        if not self.validate_config():
            logger.warning("Webhook notification not configured")
            return False
        
        try:
            payload = message.to_dict()
            
            response = requests.request(
                method=self.method,
                url=self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info("Webhook notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False


class PagerDutyNotificationProvider(NotificationProvider):
    """PagerDuty notification provider"""
    
    def __init__(self, routing_key: str = None):
        self.routing_key = routing_key or os.environ.get('PAGERDUTY_ROUTING_KEY')
        self.api_url = "https://events.pagerduty.com/v2/enqueue"
    
    def validate_config(self) -> bool:
        """Validate PagerDuty configuration"""
        return bool(self.routing_key)
    
    def send(self, message: NotificationMessage) -> bool:
        """Send PagerDuty notification"""
        if not self.validate_config():
            logger.warning("PagerDuty notification not configured")
            return False
        
        try:
            severity_map = {
                'critical': 'critical',
                'high': 'error',
                'medium': 'warning',
                'low': 'info',
                'info': 'info'
            }
            
            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": message.alert_id or f"ai-nids-{message.timestamp.timestamp()}",
                "payload": {
                    "summary": message.title,
                    "source": "AI-NIDS",
                    "severity": severity_map.get(message.severity, 'info'),
                    "timestamp": message.timestamp.isoformat(),
                    "custom_details": {
                        "body": message.body,
                        "source_ip": message.source_ip,
                        "dest_ip": message.dest_ip,
                        "attack_type": message.attack_type,
                        "confidence": message.confidence
                    }
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info("PagerDuty notification sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False


class NotificationManager:
    """
    Central notification manager for AI-NIDS.
    Handles multiple notification channels with async delivery.
    """
    
    def __init__(self):
        self.providers: Dict[NotificationChannel, NotificationProvider] = {}
        self.enabled_channels: List[NotificationChannel] = []
        self._notification_queue: Queue = Queue()
        self._worker_thread: Optional[Thread] = None
        self._running = False
        
        # Initialize providers from environment
        self._init_providers()
    
    def _init_providers(self):
        """Initialize notification providers from environment"""
        # Email
        if os.environ.get('SMTP_SERVER'):
            self.providers[NotificationChannel.EMAIL] = EmailNotificationProvider()
            if self.providers[NotificationChannel.EMAIL].validate_config():
                self.enabled_channels.append(NotificationChannel.EMAIL)
        
        # Slack
        if os.environ.get('SLACK_WEBHOOK_URL'):
            self.providers[NotificationChannel.SLACK] = SlackNotificationProvider()
            self.enabled_channels.append(NotificationChannel.SLACK)
        
        # Telegram
        if os.environ.get('TELEGRAM_BOT_TOKEN'):
            self.providers[NotificationChannel.TELEGRAM] = TelegramNotificationProvider()
            if self.providers[NotificationChannel.TELEGRAM].validate_config():
                self.enabled_channels.append(NotificationChannel.TELEGRAM)
        
        # Discord
        if os.environ.get('DISCORD_WEBHOOK_URL'):
            self.providers[NotificationChannel.DISCORD] = DiscordNotificationProvider()
            self.enabled_channels.append(NotificationChannel.DISCORD)
        
        # Generic Webhook
        if os.environ.get('WEBHOOK_URL'):
            self.providers[NotificationChannel.WEBHOOK] = WebhookNotificationProvider()
            self.enabled_channels.append(NotificationChannel.WEBHOOK)
        
        # PagerDuty
        if os.environ.get('PAGERDUTY_ROUTING_KEY'):
            self.providers[NotificationChannel.PAGERDUTY] = PagerDutyNotificationProvider()
            self.enabled_channels.append(NotificationChannel.PAGERDUTY)
        
        logger.info(f"Notification manager initialized with channels: {[c.value for c in self.enabled_channels]}")
    
    def add_provider(self, channel: NotificationChannel, provider: NotificationProvider):
        """Add a notification provider"""
        self.providers[channel] = provider
        if channel not in self.enabled_channels and provider.validate_config():
            self.enabled_channels.append(channel)
    
    def remove_provider(self, channel: NotificationChannel):
        """Remove a notification provider"""
        if channel in self.providers:
            del self.providers[channel]
        if channel in self.enabled_channels:
            self.enabled_channels.remove(channel)
    
    def start_async_worker(self):
        """Start async notification worker"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        logger.info("Notification async worker started")
    
    def stop_async_worker(self):
        """Stop async notification worker"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
    
    def _process_queue(self):
        """Process notification queue"""
        while self._running:
            try:
                message, channels = self._notification_queue.get(timeout=1)
                self._send_to_channels(message, channels)
            except Exception:
                continue
    
    def _send_to_channels(
        self,
        message: NotificationMessage,
        channels: Optional[List[NotificationChannel]] = None
    ) -> Dict[NotificationChannel, bool]:
        """Send notification to specified channels"""
        channels = channels or self.enabled_channels
        results = {}
        
        for channel in channels:
            if channel in self.providers:
                results[channel] = self.providers[channel].send(message)
        
        return results
    
    def send(
        self,
        message: NotificationMessage,
        channels: Optional[List[NotificationChannel]] = None,
        async_send: bool = True
    ) -> Dict[NotificationChannel, bool]:
        """
        Send notification.
        
        Args:
            message: Notification message
            channels: Specific channels to send to (default: all enabled)
            async_send: Send asynchronously via queue
            
        Returns:
            Dictionary of channel -> success status
        """
        if not self.enabled_channels:
            logger.warning("No notification channels enabled")
            return {}
        
        channels = channels or self.enabled_channels
        
        if async_send and self._running:
            self._notification_queue.put((message, channels))
            return {c: True for c in channels}  # Queued
        else:
            return self._send_to_channels(message, channels)
    
    def send_alert_notification(
        self,
        alert_id: str,
        title: str,
        description: str,
        source_ip: str,
        dest_ip: str,
        attack_type: str,
        severity: str,
        confidence: float
    ):
        """Convenience method to send alert notification"""
        priority_map = {
            'critical': NotificationPriority.CRITICAL,
            'high': NotificationPriority.HIGH,
            'medium': NotificationPriority.MEDIUM,
            'low': NotificationPriority.LOW,
            'info': NotificationPriority.LOW
        }
        
        message = NotificationMessage(
            title=title,
            body=description,
            priority=priority_map.get(severity, NotificationPriority.MEDIUM),
            alert_id=alert_id,
            source_ip=source_ip,
            dest_ip=dest_ip,
            attack_type=attack_type,
            severity=severity,
            confidence=confidence
        )
        
        return self.send(message)


# Singleton instance
_notification_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager instance"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager


def create_notification_manager() -> NotificationManager:
    """Create new notification manager instance"""
    return NotificationManager()
