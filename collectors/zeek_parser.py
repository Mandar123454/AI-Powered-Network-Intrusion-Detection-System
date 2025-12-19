"""
Zeek Log Parser for AI-NIDS
Parses Zeek (Bro) network logs
"""

import os
import csv
import gzip
from typing import Dict, List, Optional, Generator, Callable
from datetime import datetime
from dataclasses import dataclass
import threading
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ZeekConn:
    """Represents a Zeek connection log entry."""
    ts: datetime
    uid: str
    id_orig_h: str  # Source IP
    id_orig_p: int  # Source port
    id_resp_h: str  # Destination IP
    id_resp_p: int  # Destination port
    proto: str
    service: Optional[str] = None
    duration: float = 0.0
    orig_bytes: int = 0
    resp_bytes: int = 0
    conn_state: Optional[str] = None
    local_orig: bool = False
    local_resp: bool = False
    missed_bytes: int = 0
    history: Optional[str] = None
    orig_pkts: int = 0
    orig_ip_bytes: int = 0
    resp_pkts: int = 0
    resp_ip_bytes: int = 0
    
    def to_features(self) -> Dict:
        """Convert to feature dictionary for ML model."""
        total_bytes = self.orig_bytes + self.resp_bytes
        total_pkts = self.orig_pkts + self.resp_pkts
        duration_us = self.duration * 1000000  # Convert to microseconds
        
        return {
            'Source IP': self.id_orig_h,
            'Destination IP': self.id_resp_h,
            'Source Port': self.id_orig_p,
            'Destination Port': self.id_resp_p,
            'Protocol': self._proto_to_int(),
            'Flow Duration': duration_us,
            'Total Fwd Packets': self.orig_pkts,
            'Total Backward Packets': self.resp_pkts,
            'Total Length of Fwd Packets': self.orig_bytes,
            'Total Length of Bwd Packets': self.resp_bytes,
            'Flow Bytes/s': total_bytes / max(self.duration, 0.001),
            'Flow Packets/s': total_pkts / max(self.duration, 0.001),
            'Fwd Packets/s': self.orig_pkts / max(self.duration, 0.001),
            'Bwd Packets/s': self.resp_pkts / max(self.duration, 0.001),
            'Down/Up Ratio': self.resp_bytes / max(self.orig_bytes, 1),
            'Average Packet Size': total_bytes / max(total_pkts, 1),
            # Zeek-specific features
            'Conn_State': self._conn_state_to_int(),
            'Service': self.service or 'unknown',
            'History_Length': len(self.history) if self.history else 0
        }
    
    def _proto_to_int(self) -> int:
        """Convert protocol string to integer."""
        proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1, 'sctp': 132}
        return proto_map.get(self.proto.lower(), 0)
    
    def _conn_state_to_int(self) -> int:
        """Convert connection state to integer."""
        state_map = {
            'S0': 0, 'S1': 1, 'SF': 2, 'REJ': 3, 'S2': 4, 'S3': 5,
            'RSTO': 6, 'RSTR': 7, 'RSTOS0': 8, 'RSTRH': 9, 'SH': 10,
            'SHR': 11, 'OTH': 12
        }
        return state_map.get(self.conn_state, 12)


class ZeekParser:
    """
    Parser for Zeek log files.
    Supports conn.log, dns.log, http.log, ssl.log, and other Zeek logs.
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Zeek parser.
        
        Args:
            log_dir: Directory containing Zeek logs
            config: Configuration dictionary
        """
        self.log_dir = log_dir or '/var/log/zeek/current'
        self.config = config or {}
        
        # Field separators and types
        self.separator = '\t'
        self.set_separator = ','
        self.empty_field = '(empty)'
        self.unset_field = '-'
        
        # Parsing state
        self._last_position: Dict[str, int] = {}
        self._running = False
        self._watch_thread: Optional[threading.Thread] = None
        
        # Event handlers
        self.conn_handlers: List[Callable] = []
        self.dns_handlers: List[Callable] = []
        self.http_handlers: List[Callable] = []
        
        logger.info(f"ZeekParser initialized with log dir: {self.log_dir}")
    
    def _parse_header(self, file_handle) -> Dict:
        """Parse Zeek log header to get field names and types."""
        header = {}
        
        for line in file_handle:
            line = line.strip()
            if line.startswith('#separator'):
                parts = line.split(' ', 1)
                if len(parts) > 1:
                    # Handle hex separator
                    sep = parts[1]
                    if sep.startswith('\\x'):
                        header['separator'] = bytes.fromhex(sep[2:]).decode()
                    else:
                        header['separator'] = sep
            elif line.startswith('#set_separator'):
                header['set_separator'] = line.split('\t', 1)[1] if '\t' in line else ','
            elif line.startswith('#empty_field'):
                header['empty_field'] = line.split('\t', 1)[1] if '\t' in line else '(empty)'
            elif line.startswith('#unset_field'):
                header['unset_field'] = line.split('\t', 1)[1] if '\t' in line else '-'
            elif line.startswith('#fields'):
                header['fields'] = line.split('\t')[1:]
            elif line.startswith('#types'):
                header['types'] = line.split('\t')[1:]
            elif not line.startswith('#'):
                # End of header
                break
        
        return header
    
    def _convert_value(self, value: str, field_type: str, header: Dict) -> any:
        """Convert string value to appropriate type."""
        if value == header.get('unset_field', '-') or value == header.get('empty_field', '(empty)'):
            return None
        
        try:
            if field_type == 'time':
                return datetime.fromtimestamp(float(value))
            elif field_type == 'interval':
                return float(value)
            elif field_type == 'count' or field_type == 'port':
                return int(value)
            elif field_type == 'int':
                return int(value)
            elif field_type == 'double':
                return float(value)
            elif field_type == 'bool':
                return value.lower() in ('t', 'true', '1')
            elif field_type.startswith('set[') or field_type.startswith('vector['):
                sep = header.get('set_separator', ',')
                return value.split(sep) if value else []
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def parse_conn_line(self, line: str, header: Dict) -> Optional[ZeekConn]:
        """Parse a single conn.log line."""
        if line.startswith('#') or not line.strip():
            return None
        
        sep = header.get('separator', '\t')
        fields = header.get('fields', [])
        types = header.get('types', [])
        
        values = line.strip().split(sep)
        
        if len(values) != len(fields):
            return None
        
        # Build record dictionary
        record = {}
        for i, (field, value) in enumerate(zip(fields, values)):
            field_type = types[i] if i < len(types) else 'string'
            record[field] = self._convert_value(value, field_type, header)
        
        # Create ZeekConn object
        try:
            return ZeekConn(
                ts=record.get('ts', datetime.now()),
                uid=record.get('uid', ''),
                id_orig_h=record.get('id.orig_h', ''),
                id_orig_p=record.get('id.orig_p', 0) or 0,
                id_resp_h=record.get('id.resp_h', ''),
                id_resp_p=record.get('id.resp_p', 0) or 0,
                proto=record.get('proto', 'tcp'),
                service=record.get('service'),
                duration=record.get('duration', 0) or 0,
                orig_bytes=record.get('orig_bytes', 0) or 0,
                resp_bytes=record.get('resp_bytes', 0) or 0,
                conn_state=record.get('conn_state'),
                local_orig=record.get('local_orig', False) or False,
                local_resp=record.get('local_resp', False) or False,
                missed_bytes=record.get('missed_bytes', 0) or 0,
                history=record.get('history'),
                orig_pkts=record.get('orig_pkts', 0) or 0,
                orig_ip_bytes=record.get('orig_ip_bytes', 0) or 0,
                resp_pkts=record.get('resp_pkts', 0) or 0,
                resp_ip_bytes=record.get('resp_ip_bytes', 0) or 0
            )
        except Exception as e:
            logger.warning(f"Failed to create ZeekConn: {e}")
            return None
    
    def parse_conn_file(
        self,
        file_path: Optional[str] = None
    ) -> Generator[ZeekConn, None, None]:
        """
        Parse Zeek conn.log file.
        
        Args:
            file_path: Path to conn.log file
            
        Yields:
            ZeekConn objects
        """
        path = file_path or os.path.join(self.log_dir, 'conn.log')
        
        if not os.path.exists(path):
            # Try gzipped version
            if os.path.exists(path + '.gz'):
                path = path + '.gz'
            else:
                logger.warning(f"Conn log not found: {path}")
                return
        
        # Open file (gzipped or plain)
        open_func = gzip.open if path.endswith('.gz') else open
        mode = 'rt' if path.endswith('.gz') else 'r'
        
        with open_func(path, mode) as f:
            header = self._parse_header(f)
            
            for line in f:
                conn = self.parse_conn_line(line, header)
                if conn:
                    yield conn
    
    def parse_generic_log(
        self,
        log_type: str,
        file_path: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """
        Parse any Zeek log file.
        
        Args:
            log_type: Type of log (dns, http, ssl, etc.)
            file_path: Path to log file
            
        Yields:
            Record dictionaries
        """
        path = file_path or os.path.join(self.log_dir, f'{log_type}.log')
        
        if not os.path.exists(path):
            if os.path.exists(path + '.gz'):
                path = path + '.gz'
            else:
                logger.warning(f"{log_type} log not found: {path}")
                return
        
        open_func = gzip.open if path.endswith('.gz') else open
        mode = 'rt' if path.endswith('.gz') else 'r'
        
        with open_func(path, mode) as f:
            header = self._parse_header(f)
            sep = header.get('separator', '\t')
            fields = header.get('fields', [])
            types = header.get('types', [])
            
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                values = line.strip().split(sep)
                
                if len(values) != len(fields):
                    continue
                
                record = {}
                for i, (field, value) in enumerate(zip(fields, values)):
                    field_type = types[i] if i < len(types) else 'string'
                    record[field] = self._convert_value(value, field_type, header)
                
                yield record
    
    def register_conn_handler(self, handler: Callable) -> None:
        """Register a connection event handler."""
        self.conn_handlers.append(handler)
    
    def register_dns_handler(self, handler: Callable) -> None:
        """Register a DNS event handler."""
        self.dns_handlers.append(handler)
    
    def register_http_handler(self, handler: Callable) -> None:
        """Register an HTTP event handler."""
        self.http_handlers.append(handler)
    
    def start_watching(self, poll_interval: float = 1.0) -> None:
        """Start watching log directory for new entries."""
        if self._running:
            return
        
        self._running = True
        self._watch_thread = threading.Thread(
            target=self._watch_logs,
            args=(poll_interval,),
            daemon=True
        )
        self._watch_thread.start()
        logger.info("Started watching Zeek logs")
    
    def stop_watching(self) -> None:
        """Stop watching log files."""
        self._running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("Stopped watching Zeek logs")
    
    def _watch_logs(self, poll_interval: float) -> None:
        """Watch logs for new entries."""
        conn_path = os.path.join(self.log_dir, 'conn.log')
        
        while self._running:
            try:
                if os.path.exists(conn_path):
                    current_size = os.path.getsize(conn_path)
                    last_pos = self._last_position.get(conn_path, 0)
                    
                    if current_size > last_pos:
                        with open(conn_path, 'r') as f:
                            # Skip header if at beginning
                            if last_pos == 0:
                                header = self._parse_header(f)
                            else:
                                f.seek(last_pos)
                                # Re-read header for field info
                                with open(conn_path, 'r') as hf:
                                    header = self._parse_header(hf)
                            
                            for line in f:
                                conn = self.parse_conn_line(line, header)
                                if conn:
                                    for handler in self.conn_handlers:
                                        try:
                                            handler(conn)
                                        except Exception as e:
                                            logger.error(f"Conn handler error: {e}")
                            
                            self._last_position[conn_path] = f.tell()
                    
                    elif current_size < last_pos:
                        # Log was rotated
                        self._last_position[conn_path] = 0
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Log watch error: {e}")
                time.sleep(poll_interval)


def create_zeek_parser(
    log_dir: Optional[str] = None,
    config: Optional[Dict] = None
) -> ZeekParser:
    """Factory function to create ZeekParser."""
    return ZeekParser(log_dir=log_dir, config=config)
