"""
AI-NIDS Logger Configuration
============================
Centralized logging configuration for the application
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for terminal."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        import json
        
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry)


def setup_logger(
    name: str = 'ai_nids',
    level: str = 'INFO',
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up application logger with console and file handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        console: Enable console logging
        file: Enable file logging
        json_format: Use JSON format for file logs
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        # Use colored output for terminal
        if sys.stdout.isatty():
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file:
        if log_dir is None:
            log_dir = os.environ.get('LOG_DIR', 'logs')
        
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f'{name}.log'
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_formatter = JSONFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log
        error_log = log_path / f'{name}_errors.log'
        error_handler = RotatingFileHandler(
            error_log,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str = 'ai_nids') -> logging.Logger:
    """
    Get or create a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        level = os.environ.get('LOG_LEVEL', 'INFO')
        return setup_logger(name, level=level)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding contextual information."""
    
    def process(self, msg, kwargs):
        # Add extra context to log messages
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        
        # Prepend context to message
        context_parts = [f"{k}={v}" for k, v in self.extra.items()]
        if context_parts:
            msg = f"[{', '.join(context_parts)}] {msg}"
        
        return msg, kwargs


def get_request_logger(request_id: str, user_id: Optional[str] = None) -> LoggerAdapter:
    """
    Get a logger with request context.
    
    Args:
        request_id: Unique request identifier
        user_id: Optional user identifier
        
    Returns:
        Logger adapter with context
    """
    logger = get_logger('ai_nids.request')
    extra = {'request_id': request_id}
    if user_id:
        extra['user_id'] = user_id
    return LoggerAdapter(logger, extra)


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls.
    
    Usage:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator
