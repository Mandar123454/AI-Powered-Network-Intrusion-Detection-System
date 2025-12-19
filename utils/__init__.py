"""
AI-NIDS Utilities Module
========================
Common utility functions used across the application
"""

from utils.helpers import (
    get_ip_info,
    is_private_ip,
    validate_ip,
    format_bytes,
    format_duration,
    calculate_rate,
    hash_string,
    generate_api_key,
    sanitize_input,
    parse_timestamp
)

from utils.logger import setup_logger, get_logger

from utils.notifications import (
    NotificationManager,
    NotificationMessage,
    NotificationChannel,
    NotificationPriority,
    get_notification_manager,
    create_notification_manager
)

__all__ = [
    # Helpers
    'get_ip_info',
    'is_private_ip',
    'validate_ip',
    'format_bytes',
    'format_duration',
    'calculate_rate',
    'hash_string',
    'generate_api_key',
    'sanitize_input',
    'parse_timestamp',
    # Logger
    'setup_logger',
    'get_logger',
    # Notifications
    'NotificationManager',
    'NotificationMessage',
    'NotificationChannel',
    'NotificationPriority',
    'get_notification_manager',
    'create_notification_manager'
]
