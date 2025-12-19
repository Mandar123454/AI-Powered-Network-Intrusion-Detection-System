"""
AI-NIDS Log Processor
======================
Background task for processing Suricata and Zeek logs
"""

import os
import time
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
import threading
from queue import Queue, Empty

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for log processing"""
    files_processed: int = 0
    events_processed: int = 0
    alerts_generated: int = 0
    errors_encountered: int = 0
    last_processed: Optional[datetime] = None
    processing_rate: float = 0.0  # events per second


class LogProcessor:
    """
    Background log processor for network security logs.
    Monitors Suricata and Zeek log directories and processes new entries.
    """
    
    def __init__(
        self,
        suricata_log_path: Optional[str] = None,
        zeek_log_dir: Optional[str] = None,
        batch_size: int = 100,
        poll_interval: float = 1.0
    ):
        """
        Initialize the log processor.
        
        Args:
            suricata_log_path: Path to Suricata EVE JSON log
            zeek_log_dir: Directory containing Zeek logs
            batch_size: Number of events to process in each batch
            poll_interval: Seconds between log file checks
        """
        self.suricata_log_path = suricata_log_path or os.environ.get(
            'SURICATA_LOG_PATH', '/var/log/suricata/eve.json'
        )
        self.zeek_log_dir = zeek_log_dir or os.environ.get(
            'ZEEK_LOG_DIR', '/var/log/zeek/current'
        )
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        
        # Processing state
        self.running = False
        self.stats = ProcessingStats()
        self._shutdown_event = threading.Event()
        self._event_queue: Queue = Queue(maxsize=10000)
        
        # File tracking
        self._file_positions: Dict[str, int] = {}
        
        # Initialize parsers and detector (lazy loading)
        self._suricata_parser = None
        self._zeek_parser = None
        self._detector = None
        self._alert_manager = None
        
        logger.info("LogProcessor initialized")
    
    def _get_suricata_parser(self):
        """Lazy load Suricata parser"""
        if self._suricata_parser is None:
            from collectors.suricata_parser import SuricataParser
            self._suricata_parser = SuricataParser()
        return self._suricata_parser
    
    def _get_zeek_parser(self):
        """Lazy load Zeek parser"""
        if self._zeek_parser is None:
            from collectors.zeek_parser import ZeekParser
            self._zeek_parser = ZeekParser()
        return self._zeek_parser
    
    def _get_detector(self):
        """Lazy load detection engine"""
        if self._detector is None:
            from detection.detector import DetectionEngine
            self._detector = DetectionEngine()
        return self._detector
    
    def _get_alert_manager(self):
        """Lazy load alert manager"""
        if self._alert_manager is None:
            from detection.alert_manager import AlertManager
            self._alert_manager = AlertManager()
        return self._alert_manager
    
    def start(self):
        """Start the log processor"""
        logger.info("Starting log processor...")
        self.running = True
        self._shutdown_event.clear()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start worker threads
        threads = []
        
        # Suricata log watcher
        if Path(self.suricata_log_path).exists():
            t = threading.Thread(
                target=self._watch_suricata_logs,
                name="SuricataWatcher",
                daemon=True
            )
            t.start()
            threads.append(t)
            logger.info(f"Started Suricata watcher for {self.suricata_log_path}")
        else:
            logger.warning(f"Suricata log not found: {self.suricata_log_path}")
        
        # Zeek log watcher
        if Path(self.zeek_log_dir).exists():
            t = threading.Thread(
                target=self._watch_zeek_logs,
                name="ZeekWatcher",
                daemon=True
            )
            t.start()
            threads.append(t)
            logger.info(f"Started Zeek watcher for {self.zeek_log_dir}")
        else:
            logger.warning(f"Zeek log directory not found: {self.zeek_log_dir}")
        
        # Event processor
        t = threading.Thread(
            target=self._process_events,
            name="EventProcessor",
            daemon=True
        )
        t.start()
        threads.append(t)
        
        # Stats reporter
        t = threading.Thread(
            target=self._report_stats,
            name="StatsReporter",
            daemon=True
        )
        t.start()
        threads.append(t)
        
        # Wait for shutdown
        self._shutdown_event.wait()
        
        # Cleanup
        logger.info("Shutting down log processor...")
        self.running = False
        
        for t in threads:
            t.join(timeout=5.0)
        
        logger.info("Log processor stopped")
    
    def stop(self):
        """Stop the log processor"""
        self.running = False
        self._shutdown_event.set()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
    
    def _watch_suricata_logs(self):
        """Watch and process Suricata EVE JSON logs"""
        parser = self._get_suricata_parser()
        log_path = Path(self.suricata_log_path)
        
        # Initialize file position
        if log_path.exists():
            self._file_positions[str(log_path)] = log_path.stat().st_size
        
        while self.running:
            try:
                if not log_path.exists():
                    time.sleep(self.poll_interval)
                    continue
                
                current_size = log_path.stat().st_size
                last_position = self._file_positions.get(str(log_path), 0)
                
                # Check for log rotation
                if current_size < last_position:
                    logger.info("Detected Suricata log rotation")
                    last_position = 0
                
                if current_size > last_position:
                    # Read new lines
                    with open(log_path, 'r') as f:
                        f.seek(last_position)
                        lines = f.readlines()
                        self._file_positions[str(log_path)] = f.tell()
                    
                    # Parse and queue events
                    for line in lines:
                        try:
                            event = parser.parse_eve_line(line)
                            if event:
                                self._event_queue.put({
                                    'source': 'suricata',
                                    'data': event,
                                    'timestamp': datetime.now()
                                })
                        except Exception as e:
                            self.stats.errors_encountered += 1
                            logger.debug(f"Error parsing Suricata line: {e}")
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.stats.errors_encountered += 1
                logger.error(f"Error in Suricata watcher: {e}")
                time.sleep(self.poll_interval * 2)
    
    def _watch_zeek_logs(self):
        """Watch and process Zeek logs"""
        parser = self._get_zeek_parser()
        log_dir = Path(self.zeek_log_dir)
        
        # Log files to watch
        log_files = ['conn.log', 'dns.log', 'http.log', 'ssl.log']
        
        while self.running:
            try:
                for log_file in log_files:
                    log_path = log_dir / log_file
                    
                    if not log_path.exists():
                        continue
                    
                    current_size = log_path.stat().st_size
                    last_position = self._file_positions.get(str(log_path), 0)
                    
                    # Check for log rotation
                    if current_size < last_position:
                        logger.info(f"Detected Zeek log rotation: {log_file}")
                        last_position = 0
                    
                    if current_size > last_position:
                        # Read new lines
                        with open(log_path, 'r') as f:
                            f.seek(last_position)
                            lines = f.readlines()
                            self._file_positions[str(log_path)] = f.tell()
                        
                        # Parse and queue events
                        for line in lines:
                            try:
                                if line.startswith('#'):
                                    continue
                                
                                event = parser.parse_log_line(line, log_file)
                                if event:
                                    self._event_queue.put({
                                        'source': 'zeek',
                                        'type': log_file,
                                        'data': event,
                                        'timestamp': datetime.now()
                                    })
                            except Exception as e:
                                self.stats.errors_encountered += 1
                                logger.debug(f"Error parsing Zeek line: {e}")
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.stats.errors_encountered += 1
                logger.error(f"Error in Zeek watcher: {e}")
                time.sleep(self.poll_interval * 2)
    
    def _process_events(self):
        """Process queued events through detection pipeline"""
        detector = self._get_detector()
        alert_manager = self._get_alert_manager()
        
        batch: List[Dict[str, Any]] = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # Collect batch
                try:
                    event = self._event_queue.get(timeout=0.1)
                    batch.append(event)
                except Empty:
                    pass
                
                # Process batch when full or on timeout
                current_time = time.time()
                time_elapsed = current_time - last_process_time
                
                should_process = (
                    len(batch) >= self.batch_size or
                    (len(batch) > 0 and time_elapsed >= 1.0)
                )
                
                if should_process:
                    self._process_batch(batch, detector, alert_manager)
                    
                    # Update stats
                    self.stats.events_processed += len(batch)
                    self.stats.last_processed = datetime.now()
                    if time_elapsed > 0:
                        self.stats.processing_rate = len(batch) / time_elapsed
                    
                    batch = []
                    last_process_time = current_time
                    
            except Exception as e:
                self.stats.errors_encountered += 1
                logger.error(f"Error in event processor: {e}")
                batch = []
    
    def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        detector,
        alert_manager
    ):
        """Process a batch of events"""
        for event in batch:
            try:
                source = event.get('source')
                data = event.get('data')
                
                if source == 'suricata':
                    # Convert Suricata event to features
                    features = self._suricata_to_features(data)
                elif source == 'zeek':
                    # Convert Zeek event to features
                    features = self._zeek_to_features(data, event.get('type'))
                else:
                    continue
                
                if features is None:
                    continue
                
                # Run through detector
                result = detector.analyze(features)
                
                # Generate alert if needed
                if result and result.is_threat:
                    alert_manager.create_alert(
                        source_ip=features.get('src_ip', 'unknown'),
                        dest_ip=features.get('dst_ip', 'unknown'),
                        attack_type=result.attack_type,
                        severity=result.severity.value,
                        confidence=result.confidence,
                        raw_data=data
                    )
                    self.stats.alerts_generated += 1
                    
            except Exception as e:
                self.stats.errors_encountered += 1
                logger.debug(f"Error processing event: {e}")
    
    def _suricata_to_features(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Suricata event to feature dict"""
        try:
            # Extract relevant features
            features = {
                'src_ip': event.get('src_ip'),
                'dst_ip': event.get('dest_ip'),
                'src_port': event.get('src_port', 0),
                'dst_port': event.get('dest_port', 0),
                'protocol': event.get('proto', 'unknown'),
                'timestamp': event.get('timestamp')
            }
            
            # Add flow stats if available
            flow = event.get('flow', {})
            if flow:
                features['bytes_sent'] = flow.get('bytes_toserver', 0)
                features['bytes_recv'] = flow.get('bytes_toclient', 0)
                features['pkts_sent'] = flow.get('pkts_toserver', 0)
                features['pkts_recv'] = flow.get('pkts_toclient', 0)
            
            # Add alert info if present
            alert = event.get('alert', {})
            if alert:
                features['signature_id'] = alert.get('signature_id')
                features['signature'] = alert.get('signature')
                features['severity'] = alert.get('severity', 3)
            
            return features
            
        except Exception:
            return None
    
    def _zeek_to_features(
        self,
        event: Dict[str, Any],
        log_type: str
    ) -> Optional[Dict[str, Any]]:
        """Convert Zeek event to feature dict"""
        try:
            features = {
                'src_ip': event.get('id.orig_h'),
                'dst_ip': event.get('id.resp_h'),
                'src_port': event.get('id.orig_p', 0),
                'dst_port': event.get('id.resp_p', 0),
                'timestamp': event.get('ts')
            }
            
            if log_type == 'conn.log':
                features['protocol'] = event.get('proto', 'unknown')
                features['duration'] = event.get('duration', 0)
                features['bytes_sent'] = event.get('orig_bytes', 0)
                features['bytes_recv'] = event.get('resp_bytes', 0)
                features['pkts_sent'] = event.get('orig_pkts', 0)
                features['pkts_recv'] = event.get('resp_pkts', 0)
                features['conn_state'] = event.get('conn_state')
            
            elif log_type == 'dns.log':
                features['query'] = event.get('query')
                features['qtype'] = event.get('qtype_name')
                features['rcode'] = event.get('rcode_name')
            
            elif log_type == 'http.log':
                features['method'] = event.get('method')
                features['host'] = event.get('host')
                features['uri'] = event.get('uri')
                features['status_code'] = event.get('status_code')
                features['user_agent'] = event.get('user_agent')
            
            return features
            
        except Exception:
            return None
    
    def _report_stats(self):
        """Periodically report processing statistics"""
        while self.running:
            try:
                time.sleep(60)  # Report every minute
                
                logger.info(
                    f"Processing stats - "
                    f"Events: {self.stats.events_processed}, "
                    f"Alerts: {self.stats.alerts_generated}, "
                    f"Errors: {self.stats.errors_encountered}, "
                    f"Rate: {self.stats.processing_rate:.1f} events/sec, "
                    f"Queue size: {self._event_queue.qsize()}"
                )
                
            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        return {
            'files_processed': self.stats.files_processed,
            'events_processed': self.stats.events_processed,
            'alerts_generated': self.stats.alerts_generated,
            'errors_encountered': self.stats.errors_encountered,
            'last_processed': self.stats.last_processed.isoformat() if self.stats.last_processed else None,
            'processing_rate': self.stats.processing_rate,
            'queue_size': self._event_queue.qsize(),
            'running': self.running
        }


def main():
    """Main entry point for log processor"""
    logger.info("=" * 60)
    logger.info("AI-NIDS Log Processor Starting")
    logger.info("=" * 60)
    
    processor = LogProcessor(
        batch_size=int(os.environ.get('BATCH_SIZE', 100)),
        poll_interval=float(os.environ.get('POLL_INTERVAL', 1.0))
    )
    
    try:
        processor.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        processor.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
