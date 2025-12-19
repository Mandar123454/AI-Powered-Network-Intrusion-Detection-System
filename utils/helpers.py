"""
AI-NIDS Helper Functions
========================
Common utility functions for the application
"""

import re
import hashlib
import secrets
import ipaddress
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from functools import lru_cache


def validate_ip(ip: str) -> bool:
    """
    Validate an IP address.
    
    Args:
        ip: IP address string
        
    Returns:
        True if valid, False otherwise
    """
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False


def is_private_ip(ip: str) -> bool:
    """
    Check if an IP address is private (RFC 1918).
    
    Args:
        ip: IP address string
        
    Returns:
        True if private, False otherwise
    """
    try:
        addr = ipaddress.ip_address(ip)
        return addr.is_private
    except ValueError:
        return False


@lru_cache(maxsize=1000)
def get_ip_info(ip: str) -> Dict[str, Any]:
    """
    Get information about an IP address.
    
    Args:
        ip: IP address string
        
    Returns:
        Dictionary with IP information
    """
    try:
        addr = ipaddress.ip_address(ip)
        return {
            'ip': ip,
            'version': addr.version,
            'is_private': addr.is_private,
            'is_global': addr.is_global,
            'is_multicast': addr.is_multicast,
            'is_loopback': addr.is_loopback,
            'is_link_local': addr.is_link_local,
            'is_reserved': addr.is_reserved
        }
    except ValueError:
        return {'ip': ip, 'error': 'Invalid IP address'}


def format_bytes(bytes_count: int, precision: int = 2) -> str:
    """
    Format byte count to human-readable string.
    
    Args:
        bytes_count: Number of bytes
        precision: Decimal places
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    if bytes_count < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    size = float(bytes_count)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.{precision}f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2h 30m 15s")
    """
    if seconds < 0:
        return "0s"
    
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    
    parts = []
    
    days = int(seconds // 86400)
    if days:
        parts.append(f"{days}d")
        seconds %= 86400
    
    hours = int(seconds // 3600)
    if hours:
        parts.append(f"{hours}h")
        seconds %= 3600
    
    minutes = int(seconds // 60)
    if minutes:
        parts.append(f"{minutes}m")
        seconds %= 60
    
    if seconds or not parts:
        parts.append(f"{int(seconds)}s")
    
    return " ".join(parts)


def calculate_rate(count: int, duration: float, unit: str = "s") -> float:
    """
    Calculate rate (count per time unit).
    
    Args:
        count: Total count
        duration: Duration in seconds
        unit: Time unit ('s', 'm', 'h')
        
    Returns:
        Rate value
    """
    if duration <= 0:
        return 0.0
    
    rate = count / duration
    
    if unit == 'm':
        rate *= 60
    elif unit == 'h':
        rate *= 3600
    
    return rate


def hash_string(value: str, algorithm: str = 'sha256') -> str:
    """
    Hash a string using specified algorithm.
    
    Args:
        value: String to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
    Returns:
        Hex digest of hash
    """
    hasher = hashlib.new(algorithm)
    hasher.update(value.encode('utf-8'))
    return hasher.hexdigest()


def generate_api_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure API key.
    
    Args:
        length: Length of the key
        
    Returns:
        Random API key string
    """
    return secrets.token_hex(length)


def sanitize_input(value: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not value:
        return ""
    
    # Truncate to max length
    value = value[:max_length]
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # Remove control characters except newline and tab
    value = ''.join(
        char for char in value
        if char == '\n' or char == '\t' or (ord(char) >= 32)
    )
    
    # Escape HTML entities
    html_escape_table = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }
    for char, escape in html_escape_table.items():
        value = value.replace(char, escape)
    
    return value.strip()


def parse_timestamp(
    timestamp: Union[str, int, float, datetime],
    default: Optional[datetime] = None
) -> Optional[datetime]:
    """
    Parse various timestamp formats to datetime.
    
    Args:
        timestamp: Timestamp in various formats
        default: Default value if parsing fails
        
    Returns:
        datetime object or default
    """
    if isinstance(timestamp, datetime):
        return timestamp
    
    if isinstance(timestamp, (int, float)):
        try:
            # Unix timestamp
            return datetime.fromtimestamp(timestamp)
        except (ValueError, OSError):
            return default
    
    if isinstance(timestamp, str):
        # Common timestamp formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue
        
        # Try ISO format
        try:
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            pass
    
    return default


def truncate_string(value: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    Truncate a string to specified length.
    
    Args:
        value: String to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(value) <= max_length:
        return value
    return value[:max_length - len(suffix)] + suffix


def mask_ip(ip: str, mask_last: int = 1) -> str:
    """
    Mask parts of an IP address for privacy.
    
    Args:
        ip: IP address to mask
        mask_last: Number of octets to mask
        
    Returns:
        Masked IP address
    """
    try:
        addr = ipaddress.ip_address(ip)
        if addr.version == 4:
            parts = ip.split('.')
            for i in range(max(0, 4 - mask_last), 4):
                parts[i] = 'xxx'
            return '.'.join(parts)
        else:
            # IPv6
            return ip[:len(ip)//2] + ':xxxx:xxxx:xxxx'
    except ValueError:
        return ip


def is_valid_port(port: int) -> bool:
    """Check if port number is valid."""
    return 0 <= port <= 65535


def is_well_known_port(port: int) -> bool:
    """Check if port is a well-known port (0-1023)."""
    return 0 <= port <= 1023


def get_common_port_name(port: int) -> Optional[str]:
    """Get common service name for port."""
    common_ports = {
        20: 'FTP Data',
        21: 'FTP',
        22: 'SSH',
        23: 'Telnet',
        25: 'SMTP',
        53: 'DNS',
        67: 'DHCP Server',
        68: 'DHCP Client',
        80: 'HTTP',
        110: 'POP3',
        123: 'NTP',
        143: 'IMAP',
        443: 'HTTPS',
        445: 'SMB',
        993: 'IMAPS',
        995: 'POP3S',
        3306: 'MySQL',
        3389: 'RDP',
        5432: 'PostgreSQL',
        5900: 'VNC',
        6379: 'Redis',
        8080: 'HTTP Alt',
        8443: 'HTTPS Alt',
        27017: 'MongoDB'
    }
    return common_ports.get(port)
