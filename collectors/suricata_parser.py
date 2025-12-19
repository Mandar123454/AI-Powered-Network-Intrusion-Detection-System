"""
Suricata Log Parser for AI-NIDS
Parses Suricata EVE JSON logs and alert logs
"""

import json
import os
import time
from typing import Dict, List, Optional, Generator, Callable
from datetime import datetime
from dataclasses import dataclass
import threading
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class SuricataFlow:
    """Represents a parsed Suricata flow."""
    timestamp: datetime
    src_ip: str
    src_port: int
    dest_ip: str
    dest_port: int
    proto: str
    app_proto: Optional[str] = None
    
    # Flow statistics
    bytes_toserver: int = 0
    bytes_toclient: int = 0
    pkts_toserver: int = 0
    pkts_toclient: int = 0
    
    # Duration
    flow_start: Optional[datetime] = None
    flow_end: Optional[datetime] = None
    
    # Suricata alert info (if any)
    alert_signature: Optional[str] = None
    alert_category: Optional[str] = None
    alert_severity: Optional[int] = None
    
    # Raw event
    raw_event: Optional[Dict] = None
    
    def to_features(self) -> Dict:
        """Convert to feature dictionary for ML model."""
        duration = 0
        if self.flow_start and self.flow_end:
            duration = (self.flow_end - self.flow_start).total_seconds() * 1000000  # microseconds
        
        total_bytes = self.bytes_toserver + self.bytes_toclient
        total_pkts = self.pkts_toserver + self.pkts_toclient
        
        return {
            'Source IP': self.src_ip,
            'Destination IP': self.dest_ip,
            'Source Port': self.src_port,
            'Destination Port': self.dest_port,
            'Protocol': self._proto_to_int(),
            'Flow Duration': duration,
            'Total Fwd Packets': self.pkts_toserver,
            'Total Backward Packets': self.pkts_toclient,
            'Total Length of Fwd Packets': self.bytes_toserver,
            'Total Length of Bwd Packets': self.bytes_toclient,
            'Flow Bytes/s': total_bytes / max(duration / 1000000, 0.001),
            'Flow Packets/s': total_pkts / max(duration / 1000000, 0.001),
            'Fwd Packets/s': self.pkts_toserver / max(duration / 1000000, 0.001),
            'Bwd Packets/s': self.pkts_toclient / max(duration / 1000000, 0.001),
            'Down/Up Ratio': self.bytes_toclient / max(self.bytes_toserver, 1),
            'Average Packet Size': total_bytes / max(total_pkts, 1)
        }
    
    def _proto_to_int(self) -> int:
        """Convert protocol string to integer."""
        proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1, 'sctp': 132}
        return proto_map.get(self.proto.lower(), 0)


class SuricataParser:
    """
    Parser for Suricata log files.
    Supports EVE JSON format and traditional alert format.
    """
    
    def __init__(
        self,
        eve_log_path: Optional[str] = None,
        alert_log_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize Suricata parser.
        
        Args:
            eve_log_path: Path to EVE JSON log file
            alert_log_path: Path to traditional alert log
            config: Configuration dictionary
        """
        self.eve_log_path = eve_log_path or '/var/log/suricata/eve.json'
        self.alert_log_path = alert_log_path or '/var/log/suricata/fast.log'
        self.config = config or {}
        
        # Parsing state
        self._last_position: Dict[str, int] = {}
        self._running = False
        self._watch_thread: Optional[threading.Thread] = None
        
        # Event handlers
        self.flow_handlers: List[Callable] = []
        self.alert_handlers: List[Callable] = []
        
        logger.info(f"SuricataParser initialized with EVE log: {self.eve_log_path}")
    
    def parse_eve_event(self, line: str) -> Optional[SuricataFlow]:
        """
        Parse a single EVE JSON event.
        
        Args:
            line: JSON line from EVE log
            
        Returns:
            SuricataFlow or None
        """
        try:
            event = json.loads(line.strip())
            
            event_type = event.get('event_type')
            
            if event_type == 'flow':
                return self._parse_flow_event(event)
            elif event_type == 'alert':
                return self._parse_alert_event(event)
            elif event_type in ['http', 'dns', 'tls', 'ssh', 'smtp']:
                return self._parse_protocol_event(event)
            
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse EVE event: {e}")
            return None
    
    def _parse_flow_event(self, event: Dict) -> SuricataFlow:
        """Parse flow event."""
        flow_data = event.get('flow', {})
        
        return SuricataFlow(
            timestamp=datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')),
            src_ip=event.get('src_ip', ''),
            src_port=event.get('src_port', 0),
            dest_ip=event.get('dest_ip', ''),
            dest_port=event.get('dest_port', 0),
            proto=event.get('proto', 'tcp'),
            app_proto=event.get('app_proto'),
            bytes_toserver=flow_data.get('bytes_toserver', 0),
            bytes_toclient=flow_data.get('bytes_toclient', 0),
            pkts_toserver=flow_data.get('pkts_toserver', 0),
            pkts_toclient=flow_data.get('pkts_toclient', 0),
            raw_event=event
        )
    
    def _parse_alert_event(self, event: Dict) -> SuricataFlow:
        """Parse alert event."""
        alert_data = event.get('alert', {})
        flow_data = event.get('flow', {})
        
        return SuricataFlow(
            timestamp=datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')),
            src_ip=event.get('src_ip', ''),
            src_port=event.get('src_port', 0),
            dest_ip=event.get('dest_ip', ''),
            dest_port=event.get('dest_port', 0),
            proto=event.get('proto', 'tcp'),
            app_proto=event.get('app_proto'),
            bytes_toserver=flow_data.get('bytes_toserver', 0),
            bytes_toclient=flow_data.get('bytes_toclient', 0),
            pkts_toserver=flow_data.get('pkts_toserver', 0),
            pkts_toclient=flow_data.get('pkts_toclient', 0),
            alert_signature=alert_data.get('signature'),
            alert_category=alert_data.get('category'),
            alert_severity=alert_data.get('severity'),
            raw_event=event
        )
    
    def _parse_protocol_event(self, event: Dict) -> SuricataFlow:
        """Parse protocol-specific event (HTTP, DNS, etc.)."""
        return SuricataFlow(
            timestamp=datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')),
            src_ip=event.get('src_ip', ''),
            src_port=event.get('src_port', 0),
            dest_ip=event.get('dest_ip', ''),
            dest_port=event.get('dest_port', 0),
            proto=event.get('proto', 'tcp'),
            app_proto=event.get('event_type'),
            raw_event=event
        )
    
    def parse_eve_file(
        self,
        file_path: Optional[str] = None,
        from_position: int = 0
    ) -> Generator[SuricataFlow, None, None]:
        """
        Parse EVE JSON log file.
        
        Args:
            file_path: Path to EVE log file
            from_position: Start position in file
            
        Yields:
            SuricataFlow objects
        """
        path = file_path or self.eve_log_path
        
        if not os.path.exists(path):
            logger.warning(f"EVE log not found: {path}")
            return
        
        with open(path, 'r') as f:
            f.seek(from_position)
            
            for line in f:
                flow = self.parse_eve_event(line)
                if flow:
                    yield flow
            
            self._last_position[path] = f.tell()
    
    def parse_alert_line(self, line: str) -> Optional[Dict]:
        """
        Parse Suricata fast.log alert line.
        
        Format: MM/DD/YYYY-HH:MM:SS.ssssss [**] [gid:sid:rev] signature [**] [Classification: class] [Priority: N] {proto} src:port -> dst:port
        """
        pattern = r'(\d{2}/\d{2}/\d{4}-\d{2}:\d{2}:\d{2}\.\d+)\s+\[\*\*\]\s+\[(\d+:\d+:\d+)\]\s+(.+?)\s+\[\*\*\]\s+\[Classification:\s*(.+?)\]\s+\[Priority:\s*(\d+)\]\s+\{(\w+)\}\s+([\d\.]+):(\d+)\s+->\s+([\d\.]+):(\d+)'
        
        match = re.match(pattern, line.strip())
        if match:
            return {
                'timestamp': match.group(1),
                'sid': match.group(2),
                'signature': match.group(3),
                'classification': match.group(4),
                'priority': int(match.group(5)),
                'proto': match.group(6),
                'src_ip': match.group(7),
                'src_port': int(match.group(8)),
                'dest_ip': match.group(9),
                'dest_port': int(match.group(10))
            }
        
        return None
    
    def parse_alert_file(
        self,
        file_path: Optional[str] = None
    ) -> Generator[Dict, None, None]:
        """
        Parse Suricata fast.log file.
        
        Args:
            file_path: Path to alert log file
            
        Yields:
            Alert dictionaries
        """
        path = file_path or self.alert_log_path
        
        if not os.path.exists(path):
            logger.warning(f"Alert log not found: {path}")
            return
        
        with open(path, 'r') as f:
            for line in f:
                alert = self.parse_alert_line(line)
                if alert:
                    yield alert
    
    def register_flow_handler(self, handler: Callable) -> None:
        """Register a flow event handler."""
        self.flow_handlers.append(handler)
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register an alert event handler."""
        self.alert_handlers.append(handler)
    
    def start_watching(self, poll_interval: float = 1.0) -> None:
        """
        Start watching log files for new entries.
        
        Args:
            poll_interval: Seconds between polls
        """
        if self._running:
            return
        
        self._running = True
        self._watch_thread = threading.Thread(
            target=self._watch_logs,
            args=(poll_interval,),
            daemon=True
        )
        self._watch_thread.start()
        logger.info("Started watching Suricata logs")
    
    def stop_watching(self) -> None:
        """Stop watching log files."""
        self._running = False
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("Stopped watching Suricata logs")
    
    def _watch_logs(self, poll_interval: float) -> None:
        """Watch logs for new entries."""
        while self._running:
            try:
                # Read new EVE events
                for flow in self.parse_eve_file(
                    from_position=self._last_position.get(self.eve_log_path, 0)
                ):
                    # Dispatch to handlers
                    if flow.alert_signature:
                        for handler in self.alert_handlers:
                            try:
                                handler(flow)
                            except Exception as e:
                                logger.error(f"Alert handler error: {e}")
                    else:
                        for handler in self.flow_handlers:
                            try:
                                handler(flow)
                            except Exception as e:
                                logger.error(f"Flow handler error: {e}")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Log watch error: {e}")
                time.sleep(poll_interval)


def create_suricata_parser(
    eve_log_path: Optional[str] = None,
    config: Optional[Dict] = None
) -> SuricataParser:
    """Factory function to create SuricataParser."""
    return SuricataParser(eve_log_path=eve_log_path, config=config)
