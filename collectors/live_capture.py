"""
Live Network Capture Module for AI-NIDS
Real-time packet capture and analysis using Scapy.
Requires: Npcap (Windows) or libpcap (Linux/Mac)
"""

import os
import sys
import logging
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CapturedPacket:
    """Represents a captured network packet."""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    flags: Dict[str, bool] = field(default_factory=dict)
    payload: bytes = b''
    raw: bytes = b''
    interface: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'length': self.length,
            'flags': self.flags,
            'interface': self.interface
        }


class PacketCallback(ABC):
    """Abstract base class for packet callbacks."""
    
    @abstractmethod
    def on_packet(self, packet: CapturedPacket) -> None:
        """Called for each captured packet."""
        pass
    
    def on_start(self) -> None:
        """Called when capture starts."""
        pass
    
    def on_stop(self) -> None:
        """Called when capture stops."""
        pass


class PrintCallback(PacketCallback):
    """Simple callback that prints packet info."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.count = 0
    
    def on_packet(self, packet: CapturedPacket) -> None:
        self.count += 1
        if self.verbose:
            print(f"[{self.count}] {packet.timestamp.strftime('%H:%M:%S.%f')[:-3]} "
                  f"{packet.src_ip}:{packet.src_port} -> {packet.dst_ip}:{packet.dst_port} "
                  f"[{packet.protocol}] {packet.length} bytes")
        else:
            if self.count % 100 == 0:
                print(f"Captured {self.count} packets...")
    
    def on_start(self) -> None:
        print("Starting capture...")
        self.count = 0
    
    def on_stop(self) -> None:
        print(f"Capture stopped. Total packets: {self.count}")


class QueueCallback(PacketCallback):
    """Callback that puts packets into a queue for processing."""
    
    def __init__(self, max_size: int = 10000):
        self.packet_queue: queue.Queue = queue.Queue(maxsize=max_size)
    
    def on_packet(self, packet: CapturedPacket) -> None:
        try:
            self.packet_queue.put_nowait(packet)
        except queue.Full:
            # Drop oldest packet
            try:
                self.packet_queue.get_nowait()
                self.packet_queue.put_nowait(packet)
            except queue.Empty:
                pass
    
    def get_packet(self, timeout: float = 1.0) -> Optional[CapturedPacket]:
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_all_packets(self) -> List[CapturedPacket]:
        packets = []
        while not self.packet_queue.empty():
            try:
                packets.append(self.packet_queue.get_nowait())
            except queue.Empty:
                break
        return packets


class StatisticsCallback(PacketCallback):
    """Callback that collects traffic statistics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_packets = 0
        self.total_bytes = 0
        self.protocols: Dict[str, int] = defaultdict(int)
        self.src_ips: Dict[str, int] = defaultdict(int)
        self.dst_ips: Dict[str, int] = defaultdict(int)
        self.ports: Dict[int, int] = defaultdict(int)
        self.start_time: Optional[datetime] = None
        self.last_time: Optional[datetime] = None
    
    def on_packet(self, packet: CapturedPacket) -> None:
        self.total_packets += 1
        self.total_bytes += packet.length
        self.protocols[packet.protocol] += 1
        self.src_ips[packet.src_ip] += 1
        self.dst_ips[packet.dst_ip] += 1
        
        if packet.src_port:
            self.ports[packet.src_port] += 1
        if packet.dst_port:
            self.ports[packet.dst_port] += 1
        
        if self.start_time is None:
            self.start_time = packet.timestamp
        self.last_time = packet.timestamp
    
    def on_start(self) -> None:
        self.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        duration = 0
        if self.start_time and self.last_time:
            duration = (self.last_time - self.start_time).total_seconds()
        
        return {
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'duration_seconds': duration,
            'packets_per_second': self.total_packets / max(duration, 0.001),
            'bytes_per_second': self.total_bytes / max(duration, 0.001),
            'protocols': dict(self.protocols),
            'top_source_ips': dict(sorted(self.src_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_destination_ips': dict(sorted(self.dst_ips.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_ports': dict(sorted(self.ports.items(), key=lambda x: x[1], reverse=True)[:10])
        }


class DetectionCallback(PacketCallback):
    """
    Callback that performs real-time anomaly detection.
    Integrates with the AI-NIDS detection engine.
    """
    
    def __init__(self, detector=None, alert_callback: Optional[Callable] = None):
        self.detector = detector
        self.alert_callback = alert_callback
        self.flow_buffer: Dict[str, List[CapturedPacket]] = defaultdict(list)
        self.buffer_size = 100
        self.alerts: List[Dict] = []
    
    def _get_flow_key(self, packet: CapturedPacket) -> str:
        """Generate bidirectional flow key."""
        ips = tuple(sorted([packet.src_ip, packet.dst_ip]))
        ports = tuple(sorted([packet.src_port, packet.dst_port]))
        return f"{ips[0]}:{ports[0]}-{ips[1]}:{ports[1]}-{packet.protocol}"
    
    def on_packet(self, packet: CapturedPacket) -> None:
        flow_key = self._get_flow_key(packet)
        self.flow_buffer[flow_key].append(packet)
        
        # Process when buffer is full
        if len(self.flow_buffer[flow_key]) >= self.buffer_size:
            self._process_flow(flow_key)
    
    def _process_flow(self, flow_key: str) -> None:
        """Process buffered packets for a flow."""
        packets = self.flow_buffer[flow_key]
        
        if not packets or not self.detector:
            self.flow_buffer[flow_key] = []
            return
        
        # Extract features
        features = self._extract_features(packets)
        
        # Run detection
        try:
            result = self.detector.analyze_features(features)
            
            if result and result.get('is_anomaly', False):
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'flow_key': flow_key,
                    'risk_score': result.get('risk_score', 0),
                    'attack_type': result.get('attack_type', 'Unknown'),
                    'details': result
                }
                self.alerts.append(alert)
                
                if self.alert_callback:
                    self.alert_callback(alert)
                    
        except Exception as e:
            logger.error(f"Detection error: {e}")
        
        # Clear buffer
        self.flow_buffer[flow_key] = []
    
    def _extract_features(self, packets: List[CapturedPacket]) -> Dict[str, Any]:
        """Extract ML features from packet list."""
        import numpy as np
        
        if not packets:
            return {}
        
        lengths = [p.length for p in packets]
        
        # Calculate inter-arrival times
        iats = []
        for i in range(1, len(packets)):
            iat = (packets[i].timestamp - packets[i-1].timestamp).total_seconds()
            iats.append(iat)
        iats = iats if iats else [0]
        
        # Count directions
        first_src = packets[0].src_ip
        forward = sum(1 for p in packets if p.src_ip == first_src)
        backward = len(packets) - forward
        
        # Count flags
        flags = defaultdict(int)
        for p in packets:
            for flag, val in p.flags.items():
                if val:
                    flags[flag] += 1
        
        duration = (packets[-1].timestamp - packets[0].timestamp).total_seconds()
        
        return {
            'duration': duration,
            'protocol': packets[0].protocol,
            'src_port': packets[0].src_port,
            'dst_port': packets[0].dst_port,
            'total_packets': len(packets),
            'packets_forward': forward,
            'packets_backward': backward,
            'total_bytes': sum(lengths),
            'packet_length_mean': float(np.mean(lengths)),
            'packet_length_std': float(np.std(lengths)),
            'packet_length_min': min(lengths),
            'packet_length_max': max(lengths),
            'iat_mean': float(np.mean(iats)),
            'iat_std': float(np.std(iats)),
            'syn_count': flags.get('SYN', 0),
            'ack_count': flags.get('ACK', 0),
            'fin_count': flags.get('FIN', 0),
            'rst_count': flags.get('RST', 0)
        }
    
    def on_stop(self) -> None:
        # Process remaining flows
        for flow_key in list(self.flow_buffer.keys()):
            if self.flow_buffer[flow_key]:
                self._process_flow(flow_key)


class LiveCapture:
    """
    Real-time network packet capture using Scapy.
    
    Features:
    - Multi-interface support
    - BPF filter support
    - Callback-based packet processing
    - Asynchronous capture with threading
    - Graceful start/stop
    
    Requirements:
    - Windows: Npcap (https://npcap.com/)
    - Linux: libpcap (usually pre-installed)
    - Mac: libpcap (pre-installed)
    """
    
    PROTOCOL_MAP = {
        1: 'ICMP',
        6: 'TCP',
        17: 'UDP',
        47: 'GRE',
        58: 'ICMPv6'
    }
    
    def __init__(self):
        self._scapy = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[PacketCallback] = []
        self._is_capturing = False
        self.interface: Optional[str] = None
        self.filter: Optional[str] = None
        
        # Load Scapy
        self._load_scapy()
    
    def _load_scapy(self):
        """Load Scapy library."""
        try:
            from scapy import all as scapy
            self._scapy = scapy
            
            # Suppress Scapy warnings
            logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
            
        except ImportError:
            raise ImportError(
                "Scapy is required for live capture. "
                "Install with: pip install scapy\n"
                "Windows users also need Npcap: https://npcap.com/"
            )
    
    @staticmethod
    def list_interfaces() -> List[Dict[str, Any]]:
        """List available network interfaces."""
        try:
            from scapy.all import get_if_list, get_if_hwaddr, conf
            
            interfaces = []
            for iface in get_if_list():
                try:
                    info = {
                        'name': iface,
                        'mac': get_if_hwaddr(iface) if hasattr(get_if_hwaddr, '__call__') else 'N/A'
                    }
                    interfaces.append(info)
                except:
                    interfaces.append({'name': iface, 'mac': 'N/A'})
            
            return interfaces
            
        except Exception as e:
            logger.error(f"Error listing interfaces: {e}")
            return []
    
    @staticmethod
    def get_default_interface() -> Optional[str]:
        """Get the default network interface."""
        try:
            from scapy.all import conf
            return conf.iface
        except:
            return None
    
    def add_callback(self, callback: PacketCallback) -> None:
        """Add a packet processing callback."""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: PacketCallback) -> None:
        """Remove a packet processing callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _parse_packet(self, scapy_pkt) -> Optional[CapturedPacket]:
        """Parse Scapy packet into CapturedPacket."""
        try:
            scapy = self._scapy
            
            # Check for IP layer
            if scapy.IP in scapy_pkt:
                ip = scapy_pkt[scapy.IP]
                src_ip = ip.src
                dst_ip = ip.dst
                proto_num = ip.proto
            elif scapy.IPv6 in scapy_pkt:
                ip = scapy_pkt[scapy.IPv6]
                src_ip = ip.src
                dst_ip = ip.dst
                proto_num = ip.nh
            else:
                return None
            
            protocol = self.PROTOCOL_MAP.get(proto_num, f'OTHER({proto_num})')
            
            # Get transport layer
            src_port = 0
            dst_port = 0
            flags = {}
            
            if scapy.TCP in scapy_pkt:
                tcp = scapy_pkt[scapy.TCP]
                src_port = tcp.sport
                dst_port = tcp.dport
                flag_str = str(tcp.flags) if hasattr(tcp, 'flags') else ''
                flags = {
                    'SYN': 'S' in flag_str,
                    'ACK': 'A' in flag_str,
                    'FIN': 'F' in flag_str,
                    'RST': 'R' in flag_str,
                    'PSH': 'P' in flag_str,
                    'URG': 'U' in flag_str
                }
            elif scapy.UDP in scapy_pkt:
                udp = scapy_pkt[scapy.UDP]
                src_port = udp.sport
                dst_port = udp.dport
            
            # Get payload
            payload = bytes(scapy_pkt.payload) if hasattr(scapy_pkt, 'payload') else b''
            
            return CapturedPacket(
                timestamp=datetime.fromtimestamp(float(scapy_pkt.time)),
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=len(scapy_pkt),
                flags=flags,
                payload=payload[:1000],  # Limit payload size
                raw=bytes(scapy_pkt),
                interface=self.interface or ''
            )
            
        except Exception as e:
            logger.debug(f"Error parsing packet: {e}")
            return None
    
    def _packet_handler(self, scapy_pkt):
        """Handle captured packet."""
        if self._stop_event.is_set():
            return
        
        packet = self._parse_packet(scapy_pkt)
        if packet:
            for callback in self._callbacks:
                try:
                    callback.on_packet(packet)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def _capture_loop(self, count: int, timeout: Optional[int]):
        """Main capture loop running in thread."""
        scapy = self._scapy
        
        try:
            # Notify callbacks
            for callback in self._callbacks:
                callback.on_start()
            
            # Start sniffing
            scapy.sniff(
                iface=self.interface,
                filter=self.filter,
                prn=self._packet_handler,
                count=count if count > 0 else 0,
                timeout=timeout,
                stop_filter=lambda x: self._stop_event.is_set(),
                store=False
            )
            
        except Exception as e:
            logger.error(f"Capture error: {e}")
            
        finally:
            self._is_capturing = False
            # Notify callbacks
            for callback in self._callbacks:
                callback.on_stop()
    
    def start(
        self,
        interface: Optional[str] = None,
        filter: Optional[str] = None,
        count: int = 0,
        timeout: Optional[int] = None,
        async_capture: bool = True
    ) -> None:
        """
        Start packet capture.
        
        Args:
            interface: Network interface (None for default)
            filter: BPF filter string (e.g., "tcp port 80")
            count: Number of packets to capture (0 for unlimited)
            timeout: Capture timeout in seconds (None for unlimited)
            async_capture: Run capture in background thread
        """
        if self._is_capturing:
            logger.warning("Capture already running")
            return
        
        self.interface = interface or self.get_default_interface()
        self.filter = filter
        self._stop_event.clear()
        self._is_capturing = True
        
        logger.info(f"Starting capture on interface: {self.interface}")
        if filter:
            logger.info(f"Using filter: {filter}")
        
        if async_capture:
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                args=(count, timeout),
                daemon=True
            )
            self._capture_thread.start()
        else:
            self._capture_loop(count, timeout)
    
    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """
        Stop packet capture.
        
        Args:
            wait: Wait for capture thread to finish
            timeout: Maximum time to wait
        """
        if not self._is_capturing:
            return
        
        logger.info("Stopping capture...")
        self._stop_event.set()
        
        if wait and self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=timeout)
    
    @property
    def is_capturing(self) -> bool:
        """Check if capture is currently running."""
        return self._is_capturing
    
    def capture_to_file(
        self,
        filepath: str,
        interface: Optional[str] = None,
        filter: Optional[str] = None,
        count: int = 0,
        timeout: Optional[int] = None
    ) -> str:
        """
        Capture packets and save to PCAP file.
        
        Args:
            filepath: Output PCAP file path
            interface: Network interface
            filter: BPF filter
            count: Packet count
            timeout: Timeout in seconds
            
        Returns:
            Path to saved file
        """
        scapy = self._scapy
        
        interface = interface or self.get_default_interface()
        
        logger.info(f"Capturing to file: {filepath}")
        
        packets = scapy.sniff(
            iface=interface,
            filter=filter,
            count=count if count > 0 else 0,
            timeout=timeout
        )
        
        scapy.wrpcap(filepath, packets)
        logger.info(f"Saved {len(packets)} packets to {filepath}")
        
        return filepath


class LiveCaptureManager:
    """
    High-level manager for live capture with integrated detection.
    Suitable for integration with the AI-NIDS Flask application.
    """
    
    def __init__(self, detector=None):
        self.capture = LiveCapture()
        self.stats_callback = StatisticsCallback()
        self.queue_callback = QueueCallback()
        self.detection_callback = DetectionCallback(detector=detector)
        
        # Add callbacks
        self.capture.add_callback(self.stats_callback)
        self.capture.add_callback(self.queue_callback)
        
        if detector:
            self.capture.add_callback(self.detection_callback)
    
    def start_capture(
        self,
        interface: Optional[str] = None,
        filter: str = "ip",
        timeout: Optional[int] = None
    ) -> bool:
        """Start live capture with detection."""
        try:
            self.capture.start(
                interface=interface,
                filter=filter,
                timeout=timeout,
                async_capture=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return False
    
    def stop_capture(self) -> None:
        """Stop live capture."""
        self.capture.stop()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current capture statistics."""
        return self.stats_callback.get_statistics()
    
    def get_recent_packets(self, limit: int = 100) -> List[Dict]:
        """Get recent captured packets."""
        packets = self.queue_callback.get_all_packets()
        return [p.to_dict() for p in packets[-limit:]]
    
    def get_alerts(self) -> List[Dict]:
        """Get detection alerts."""
        return self.detection_callback.alerts
    
    @property
    def is_running(self) -> bool:
        return self.capture.is_capturing


# Quick capture function
def quick_capture(
    duration: int = 10,
    interface: Optional[str] = None,
    filter: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Quick packet capture with statistics.
    
    Args:
        duration: Capture duration in seconds
        interface: Network interface
        filter: BPF filter
        verbose: Print packet info
        
    Returns:
        Capture statistics
    """
    capture = LiveCapture()
    stats = StatisticsCallback()
    
    capture.add_callback(stats)
    if verbose:
        capture.add_callback(PrintCallback(verbose=True))
    
    print(f"Capturing for {duration} seconds...")
    capture.start(
        interface=interface,
        filter=filter,
        timeout=duration,
        async_capture=False
    )
    
    return stats.get_statistics()


if __name__ == '__main__':
    # Example usage
    print("=" * 60)
    print("AI-NIDS Live Capture Module")
    print("=" * 60)
    
    # List interfaces
    print("\nAvailable interfaces:")
    for iface in LiveCapture.list_interfaces():
        print(f"  - {iface['name']} (MAC: {iface['mac']})")
    
    print(f"\nDefault interface: {LiveCapture.get_default_interface()}")
    
    # Quick capture demo
    if len(sys.argv) > 1 and sys.argv[1] == '--capture':
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        stats = quick_capture(duration=duration, verbose=True)
        
        print("\n" + "=" * 60)
        print("Capture Statistics:")
        print("=" * 60)
        for key, value in stats.items():
            print(f"  {key}: {value}")
    else:
        print("\nUsage: python live_capture.py --capture [duration]")
        print("Example: python live_capture.py --capture 30")
