"""
Test Utilities
==============
Tests for utility functions
"""

import pytest
from datetime import datetime


class TestHelpers:
    """Test helper functions."""
    
    def test_validate_ip_valid(self):
        """Test valid IP validation."""
        from utils.helpers import validate_ip
        
        assert validate_ip('192.168.1.1') is True
        assert validate_ip('10.0.0.1') is True
        assert validate_ip('::1') is True
        assert validate_ip('2001:db8::1') is True
    
    def test_validate_ip_invalid(self):
        """Test invalid IP validation."""
        from utils.helpers import validate_ip
        
        assert validate_ip('invalid') is False
        assert validate_ip('256.256.256.256') is False
        assert validate_ip('') is False
    
    def test_is_private_ip(self):
        """Test private IP detection."""
        from utils.helpers import is_private_ip
        
        assert is_private_ip('192.168.1.1') is True
        assert is_private_ip('10.0.0.1') is True
        assert is_private_ip('172.16.0.1') is True
        assert is_private_ip('8.8.8.8') is False
    
    def test_format_bytes(self):
        """Test byte formatting."""
        from utils.helpers import format_bytes
        
        assert format_bytes(0) == '0.00 B'
        assert format_bytes(1024) == '1.00 KB'
        assert format_bytes(1024 * 1024) == '1.00 MB'
        assert format_bytes(1024 * 1024 * 1024) == '1.00 GB'
    
    def test_format_duration(self):
        """Test duration formatting."""
        from utils.helpers import format_duration
        
        assert format_duration(0) == '0s'
        assert format_duration(60) == '1m'
        assert format_duration(3600) == '1h'
        assert format_duration(3661) == '1h 1m 1s'
        assert format_duration(86400) == '1d'
    
    def test_calculate_rate(self):
        """Test rate calculation."""
        from utils.helpers import calculate_rate
        
        assert calculate_rate(100, 10, 's') == 10.0
        assert calculate_rate(100, 10, 'm') == 600.0
        assert calculate_rate(100, 0, 's') == 0.0
    
    def test_hash_string(self):
        """Test string hashing."""
        from utils.helpers import hash_string
        
        result = hash_string('test')
        assert len(result) == 64  # SHA256 hex length
        
        # Same input produces same output
        assert hash_string('test') == hash_string('test')
        
        # Different inputs produce different outputs
        assert hash_string('test1') != hash_string('test2')
    
    def test_generate_api_key(self):
        """Test API key generation."""
        from utils.helpers import generate_api_key
        
        key1 = generate_api_key()
        key2 = generate_api_key()
        
        assert len(key1) == 64  # 32 bytes in hex
        assert key1 != key2
    
    def test_sanitize_input(self):
        """Test input sanitization."""
        from utils.helpers import sanitize_input
        
        # Test HTML escaping
        result = sanitize_input('<script>alert("xss")</script>')
        assert '<script>' not in result
        assert '&lt;' in result
        
        # Test null byte removal
        result = sanitize_input('test\x00value')
        assert '\x00' not in result
        
        # Test truncation
        long_input = 'a' * 2000
        result = sanitize_input(long_input, max_length=100)
        assert len(result) == 100
    
    def test_parse_timestamp(self):
        """Test timestamp parsing."""
        from utils.helpers import parse_timestamp
        
        # ISO format
        result = parse_timestamp('2024-01-15T10:30:00')
        assert isinstance(result, datetime)
        
        # Unix timestamp
        result = parse_timestamp(1705312200)
        assert isinstance(result, datetime)
        
        # Already datetime
        now = datetime.now()
        result = parse_timestamp(now)
        assert result == now
        
        # Invalid
        result = parse_timestamp('invalid', default=None)
        assert result is None
    
    def test_get_common_port_name(self):
        """Test port name lookup."""
        from utils.helpers import get_common_port_name
        
        assert get_common_port_name(80) == 'HTTP'
        assert get_common_port_name(443) == 'HTTPS'
        assert get_common_port_name(22) == 'SSH'
        assert get_common_port_name(99999) is None


class TestLogger:
    """Test logger functionality."""
    
    def test_setup_logger(self, tmp_path):
        """Test logger setup."""
        from utils.logger import setup_logger
        
        logger = setup_logger(
            name='test_logger',
            level='DEBUG',
            log_dir=str(tmp_path),
            console=False,
            file=True
        )
        
        assert logger is not None
        logger.info('Test message')
        
        # Check log file exists
        log_file = tmp_path / 'test_logger.log'
        assert log_file.exists()
    
    def test_get_logger(self):
        """Test getting logger."""
        from utils.logger import get_logger
        
        logger = get_logger('test')
        assert logger is not None
