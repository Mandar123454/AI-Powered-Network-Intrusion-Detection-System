"""
PCAP File Handler for AI-NIDS
Processes PCAP files for offline analysis and feature extraction.
Supports both PyShark and Scapy backends.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Generator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import hashlib
import struct
import socket

logger = logging.getLogger(__name__)


@dataclass
class PacketInfo:
    """Represents extracted information from a single packet."""
    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    flags: Dict[str, bool] = field(default_factory=dict)
    payload_size: int = 0
    ttl: int = 0
    raw_data: bytes = b''
    
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
            'payload_size': self.payload_size,
            'ttl': self.ttl
        }


@dataclass
class FlowKey:
    """Unique identifier for a network flow (bidirectional)."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    
    def __hash__(self):
        # Create bidirectional hash (A->B same as B->A)
        ips = tuple(sorted([self.src_ip, self.dst_ip]))
        ports = tuple(sorted([self.src_port, self.dst_port]))
        return hash((ips, ports, self.protocol))
    
    def __eq__(self, other):
        if not isinstance(other, FlowKey):
            return False
        ips_self = set([self.src_ip, self.dst_ip])
        ips_other = set([other.src_ip, other.dst_ip])
        ports_self = set([self.src_port, self.dst_port])
        ports_other = set([other.src_port, other.dst_port])
        return ips_self == ips_other and ports_self == ports_other and self.protocol == other.protocol


@dataclass
class NetworkFlow:
    """Aggregated flow statistics for ML feature extraction."""
    flow_key: FlowKey
    start_time: datetime
    end_time: datetime
    packets_forward: int = 0
    packets_backward: int = 0
    bytes_forward: int = 0
    bytes_backward: int = 0
    packet_lengths: List[int] = field(default_factory=list)
    inter_arrival_times: List[float] = field(default_factory=list)
    flags: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def total_packets(self) -> int:
        return self.packets_forward + self.packets_backward
    
    @property
    def total_bytes(self) -> int:
        return self.bytes_forward + self.bytes_backward
    
    def to_features(self) -> Dict[str, Any]:
        """Convert flow to ML-ready features."""
        import numpy as np
        
        pkt_lengths = self.packet_lengths if self.packet_lengths else [0]
        iat = self.inter_arrival_times if self.inter_arrival_times else [0]
        
        return {
            'duration': self.duration,
            'protocol': self.flow_key.protocol,
            'src_port': self.flow_key.src_port,
            'dst_port': self.flow_key.dst_port,
            'packets_forward': self.packets_forward,
            'packets_backward': self.packets_backward,
            'bytes_forward': self.bytes_forward,
            'bytes_backward': self.bytes_backward,
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'packets_per_second': self.total_packets / max(self.duration, 0.001),
            'bytes_per_second': self.total_bytes / max(self.duration, 0.001),
            'packet_length_mean': float(np.mean(pkt_lengths)),
            'packet_length_std': float(np.std(pkt_lengths)),
            'packet_length_min': int(np.min(pkt_lengths)),
            'packet_length_max': int(np.max(pkt_lengths)),
            'iat_mean': float(np.mean(iat)),
            'iat_std': float(np.std(iat)),
            'iat_min': float(np.min(iat)),
            'iat_max': float(np.max(iat)),
            'syn_count': self.flags.get('SYN', 0),
            'ack_count': self.flags.get('ACK', 0),
            'fin_count': self.flags.get('FIN', 0),
            'rst_count': self.flags.get('RST', 0),
            'psh_count': self.flags.get('PSH', 0),
            'urg_count': self.flags.get('URG', 0),
            'flow_bytes_ratio': self.bytes_forward / max(self.bytes_backward, 1),
            'flow_packets_ratio': self.packets_forward / max(self.packets_backward, 1)
        }


class PCAPHandler:
    """
    Handles PCAP file processing for network traffic analysis.
    Supports multiple backends: PyShark (preferred) and Scapy (fallback).
    """
    
    # Protocol number mappings
    PROTOCOL_MAP = {
        1: 'ICMP',
        6: 'TCP',
        17: 'UDP',
        47: 'GRE',
        50: 'ESP',
        51: 'AH',
        58: 'ICMPv6',
        89: 'OSPF',
        132: 'SCTP'
    }
    
    # TCP Flag masks
    TCP_FLAGS = {
        'FIN': 0x01,
        'SYN': 0x02,
        'RST': 0x04,
        'PSH': 0x08,
        'ACK': 0x10,
        'URG': 0x20,
        'ECE': 0x40,
        'CWR': 0x80
    }
    
    def __init__(self, backend: str = 'auto'):
        """
        Initialize PCAP handler.
        
        Args:
            backend: 'pyshark', 'scapy', or 'auto' (tries pyshark first)
        """
        self.backend = self._select_backend(backend)
        self.flows: Dict[FlowKey, NetworkFlow] = {}
        self.packets: List[PacketInfo] = []
        self._pyshark = None
        self._scapy = None
        
        logger.info(f"PCAPHandler initialized with backend: {self.backend}")
    
    def _select_backend(self, preference: str) -> str:
        """Select available backend based on preference."""
        if preference == 'auto':
            try:
                import pyshark
                return 'pyshark'
            except ImportError:
                try:
                    from scapy.all import rdpcap
                    return 'scapy'
                except ImportError:
                    return 'native'
        return preference
    
    def _load_pyshark(self):
        """Lazy load PyShark."""
        if self._pyshark is None:
            try:
                import pyshark
                self._pyshark = pyshark
            except ImportError:
                raise ImportError("PyShark not installed. Install with: pip install pyshark")
        return self._pyshark
    
    def _load_scapy(self):
        """Lazy load Scapy."""
        if self._scapy is None:
            try:
                from scapy import all as scapy_all
                self._scapy = scapy_all
            except ImportError:
                raise ImportError("Scapy not installed. Install with: pip install scapy")
        return self._scapy
    
    def read_pcap(self, filepath: str, max_packets: Optional[int] = None) -> List[PacketInfo]:
        """
        Read packets from a PCAP file.
        
        Args:
            filepath: Path to PCAP file
            max_packets: Maximum number of packets to read (None for all)
            
        Returns:
            List of PacketInfo objects
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PCAP file not found: {filepath}")
        
        logger.info(f"Reading PCAP file: {filepath}")
        
        if self.backend == 'pyshark':
            return self._read_with_pyshark(filepath, max_packets)
        elif self.backend == 'scapy':
            return self._read_with_scapy(filepath, max_packets)
        else:
            return self._read_native(filepath, max_packets)
    
    def _read_with_pyshark(self, filepath: str, max_packets: Optional[int]) -> List[PacketInfo]:
        """Read PCAP using PyShark."""
        pyshark = self._load_pyshark()
        packets = []
        
        try:
            cap = pyshark.FileCapture(filepath, keep_packets=False)
            
            for i, pkt in enumerate(cap):
                if max_packets and i >= max_packets:
                    break
                
                try:
                    packet_info = self._parse_pyshark_packet(pkt)
                    if packet_info:
                        packets.append(packet_info)
                except Exception as e:
                    logger.debug(f"Error parsing packet {i}: {e}")
                    continue
            
            cap.close()
            
        except Exception as e:
            logger.error(f"Error reading PCAP with PyShark: {e}")
            raise
        
        logger.info(f"Read {len(packets)} packets from {filepath}")
        self.packets = packets
        return packets
    
    def _parse_pyshark_packet(self, pkt) -> Optional[PacketInfo]:
        """Parse a PyShark packet into PacketInfo."""
        try:
            # Get timestamp
            timestamp = datetime.fromtimestamp(float(pkt.sniff_timestamp))
            
            # Get IP layer info
            if hasattr(pkt, 'ip'):
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst
                ttl = int(pkt.ip.ttl)
                proto_num = int(pkt.ip.proto)
            elif hasattr(pkt, 'ipv6'):
                src_ip = pkt.ipv6.src
                dst_ip = pkt.ipv6.dst
                ttl = int(pkt.ipv6.hlim)
                proto_num = int(pkt.ipv6.nxt)
            else:
                return None
            
            protocol = self.PROTOCOL_MAP.get(proto_num, f'PROTO_{proto_num}')
            
            # Get transport layer info
            src_port = 0
            dst_port = 0
            flags = {}
            
            if hasattr(pkt, 'tcp'):
                src_port = int(pkt.tcp.srcport)
                dst_port = int(pkt.tcp.dstport)
                flags = self._parse_tcp_flags_pyshark(pkt.tcp)
            elif hasattr(pkt, 'udp'):
                src_port = int(pkt.udp.srcport)
                dst_port = int(pkt.udp.dstport)
            
            # Get packet length
            length = int(pkt.length)
            payload_size = length - 40 if protocol == 'TCP' else length - 28
            payload_size = max(0, payload_size)
            
            return PacketInfo(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=length,
                flags=flags,
                payload_size=payload_size,
                ttl=ttl
            )
            
        except Exception as e:
            logger.debug(f"Error parsing PyShark packet: {e}")
            return None
    
    def _parse_tcp_flags_pyshark(self, tcp_layer) -> Dict[str, bool]:
        """Parse TCP flags from PyShark TCP layer."""
        flags = {}
        try:
            flags_int = int(tcp_layer.flags, 16)
            for flag_name, flag_mask in self.TCP_FLAGS.items():
                flags[flag_name] = bool(flags_int & flag_mask)
        except:
            pass
        return flags
    
    def _read_with_scapy(self, filepath: str, max_packets: Optional[int]) -> List[PacketInfo]:
        """Read PCAP using Scapy."""
        scapy = self._load_scapy()
        packets = []
        
        try:
            pcap_packets = scapy.rdpcap(filepath, count=max_packets or -1)
            
            for pkt in pcap_packets:
                try:
                    packet_info = self._parse_scapy_packet(pkt, scapy)
                    if packet_info:
                        packets.append(packet_info)
                except Exception as e:
                    logger.debug(f"Error parsing Scapy packet: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading PCAP with Scapy: {e}")
            raise
        
        logger.info(f"Read {len(packets)} packets from {filepath}")
        self.packets = packets
        return packets
    
    def _parse_scapy_packet(self, pkt, scapy) -> Optional[PacketInfo]:
        """Parse a Scapy packet into PacketInfo."""
        try:
            # Check for IP layer
            if scapy.IP in pkt:
                ip_layer = pkt[scapy.IP]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                ttl = ip_layer.ttl
                proto_num = ip_layer.proto
            elif scapy.IPv6 in pkt:
                ip_layer = pkt[scapy.IPv6]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                ttl = ip_layer.hlim
                proto_num = ip_layer.nh
            else:
                return None
            
            protocol = self.PROTOCOL_MAP.get(proto_num, f'PROTO_{proto_num}')
            
            # Get transport layer info
            src_port = 0
            dst_port = 0
            flags = {}
            
            if scapy.TCP in pkt:
                tcp_layer = pkt[scapy.TCP]
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                flags = self._parse_tcp_flags_scapy(tcp_layer)
            elif scapy.UDP in pkt:
                udp_layer = pkt[scapy.UDP]
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
            
            # Get timestamp and length
            timestamp = datetime.fromtimestamp(float(pkt.time))
            length = len(pkt)
            payload_size = len(pkt.payload) if hasattr(pkt, 'payload') else 0
            
            return PacketInfo(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=length,
                flags=flags,
                payload_size=payload_size,
                ttl=ttl,
                raw_data=bytes(pkt)
            )
            
        except Exception as e:
            logger.debug(f"Error parsing Scapy packet: {e}")
            return None
    
    def _parse_tcp_flags_scapy(self, tcp_layer) -> Dict[str, bool]:
        """Parse TCP flags from Scapy TCP layer."""
        flags = {}
        flag_str = str(tcp_layer.flags) if hasattr(tcp_layer, 'flags') else ''
        flags['SYN'] = 'S' in flag_str
        flags['ACK'] = 'A' in flag_str
        flags['FIN'] = 'F' in flag_str
        flags['RST'] = 'R' in flag_str
        flags['PSH'] = 'P' in flag_str
        flags['URG'] = 'U' in flag_str
        return flags
    
    def _read_native(self, filepath: str, max_packets: Optional[int]) -> List[PacketInfo]:
        """Read PCAP using native Python (limited functionality)."""
        packets = []
        
        with open(filepath, 'rb') as f:
            # Read global header
            global_header = f.read(24)
            if len(global_header) < 24:
                raise ValueError("Invalid PCAP file: too short")
            
            magic = struct.unpack('I', global_header[:4])[0]
            if magic == 0xa1b2c3d4:
                byte_order = '<'  # Little endian
            elif magic == 0xd4c3b2a1:
                byte_order = '>'  # Big endian
            else:
                raise ValueError(f"Invalid PCAP magic number: {hex(magic)}")
            
            packet_count = 0
            
            while True:
                if max_packets and packet_count >= max_packets:
                    break
                
                # Read packet header
                pkt_header = f.read(16)
                if len(pkt_header) < 16:
                    break
                
                ts_sec, ts_usec, incl_len, orig_len = struct.unpack(
                    f'{byte_order}IIII', pkt_header
                )
                
                # Read packet data
                pkt_data = f.read(incl_len)
                if len(pkt_data) < incl_len:
                    break
                
                # Parse packet (simplified - Ethernet + IP + TCP/UDP)
                packet_info = self._parse_native_packet(pkt_data, ts_sec, ts_usec)
                if packet_info:
                    packets.append(packet_info)
                
                packet_count += 1
        
        logger.info(f"Read {len(packets)} packets from {filepath} (native parser)")
        self.packets = packets
        return packets
    
    def _parse_native_packet(self, data: bytes, ts_sec: int, ts_usec: int) -> Optional[PacketInfo]:
        """Parse packet data natively (Ethernet + IP + TCP/UDP)."""
        try:
            timestamp = datetime.fromtimestamp(ts_sec + ts_usec / 1e6)
            
            # Skip Ethernet header (14 bytes)
            if len(data) < 34:  # 14 + 20 minimum
                return None
            
            ip_data = data[14:]
            
            # Parse IP header
            version = (ip_data[0] >> 4) & 0xF
            if version != 4:
                return None  # Only IPv4 for native parser
            
            ihl = (ip_data[0] & 0xF) * 4
            ttl = ip_data[8]
            proto = ip_data[9]
            src_ip = socket.inet_ntoa(ip_data[12:16])
            dst_ip = socket.inet_ntoa(ip_data[16:20])
            
            protocol = self.PROTOCOL_MAP.get(proto, f'PROTO_{proto}')
            
            # Parse transport layer
            transport_data = ip_data[ihl:]
            src_port = 0
            dst_port = 0
            flags = {}
            
            if proto == 6 and len(transport_data) >= 20:  # TCP
                src_port = struct.unpack('!H', transport_data[0:2])[0]
                dst_port = struct.unpack('!H', transport_data[2:4])[0]
                tcp_flags = transport_data[13]
                flags = {
                    'FIN': bool(tcp_flags & 0x01),
                    'SYN': bool(tcp_flags & 0x02),
                    'RST': bool(tcp_flags & 0x04),
                    'PSH': bool(tcp_flags & 0x08),
                    'ACK': bool(tcp_flags & 0x10),
                    'URG': bool(tcp_flags & 0x20)
                }
            elif proto == 17 and len(transport_data) >= 8:  # UDP
                src_port = struct.unpack('!H', transport_data[0:2])[0]
                dst_port = struct.unpack('!H', transport_data[2:4])[0]
            
            return PacketInfo(
                timestamp=timestamp,
                src_ip=src_ip,
                dst_ip=dst_ip,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                length=len(data),
                flags=flags,
                payload_size=max(0, len(data) - 14 - ihl - 20),
                ttl=ttl,
                raw_data=data
            )
            
        except Exception as e:
            logger.debug(f"Error parsing native packet: {e}")
            return None
    
    def extract_flows(self, packets: Optional[List[PacketInfo]] = None) -> Dict[FlowKey, NetworkFlow]:
        """
        Extract network flows from packets.
        
        Args:
            packets: List of packets (uses self.packets if None)
            
        Returns:
            Dictionary of FlowKey -> NetworkFlow
        """
        if packets is None:
            packets = self.packets
        
        if not packets:
            logger.warning("No packets to extract flows from")
            return {}
        
        flows: Dict[FlowKey, NetworkFlow] = {}
        last_packet_time: Dict[FlowKey, datetime] = {}
        
        for pkt in packets:
            flow_key = FlowKey(
                src_ip=pkt.src_ip,
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port,
                dst_port=pkt.dst_port,
                protocol=pkt.protocol
            )
            
            if flow_key not in flows:
                flows[flow_key] = NetworkFlow(
                    flow_key=flow_key,
                    start_time=pkt.timestamp,
                    end_time=pkt.timestamp
                )
            
            flow = flows[flow_key]
            flow.end_time = pkt.timestamp
            
            # Determine direction (forward = src initiated)
            is_forward = (pkt.src_ip == flow_key.src_ip and pkt.src_port == flow_key.src_port)
            
            if is_forward:
                flow.packets_forward += 1
                flow.bytes_forward += pkt.length
            else:
                flow.packets_backward += 1
                flow.bytes_backward += pkt.length
            
            flow.packet_lengths.append(pkt.length)
            
            # Calculate inter-arrival time
            if flow_key in last_packet_time:
                iat = (pkt.timestamp - last_packet_time[flow_key]).total_seconds()
                flow.inter_arrival_times.append(iat)
            last_packet_time[flow_key] = pkt.timestamp
            
            # Count flags
            for flag_name, flag_set in pkt.flags.items():
                if flag_set:
                    flow.flags[flag_name] += 1
        
        logger.info(f"Extracted {len(flows)} flows from {len(packets)} packets")
        self.flows = flows
        return flows
    
    def get_flow_features(self) -> List[Dict[str, Any]]:
        """
        Get ML-ready features from all extracted flows.
        
        Returns:
            List of feature dictionaries
        """
        if not self.flows:
            self.extract_flows()
        
        features = []
        for flow in self.flows.values():
            features.append(flow.to_features())
        
        return features
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall PCAP statistics."""
        if not self.packets:
            return {}
        
        protocols = defaultdict(int)
        unique_src_ips = set()
        unique_dst_ips = set()
        unique_src_ports = set()
        unique_dst_ports = set()
        total_bytes = 0
        
        for pkt in self.packets:
            protocols[pkt.protocol] += 1
            unique_src_ips.add(pkt.src_ip)
            unique_dst_ips.add(pkt.dst_ip)
            unique_src_ports.add(pkt.src_port)
            unique_dst_ports.add(pkt.dst_port)
            total_bytes += pkt.length
        
        duration = (self.packets[-1].timestamp - self.packets[0].timestamp).total_seconds()
        
        return {
            'total_packets': len(self.packets),
            'total_bytes': total_bytes,
            'duration_seconds': duration,
            'packets_per_second': len(self.packets) / max(duration, 0.001),
            'bytes_per_second': total_bytes / max(duration, 0.001),
            'unique_source_ips': len(unique_src_ips),
            'unique_destination_ips': len(unique_dst_ips),
            'unique_source_ports': len(unique_src_ports),
            'unique_destination_ports': len(unique_dst_ports),
            'protocol_distribution': dict(protocols),
            'start_time': self.packets[0].timestamp.isoformat(),
            'end_time': self.packets[-1].timestamp.isoformat()
        }
    
    def to_dataframe(self):
        """Convert packets to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame conversion")
        
        data = [pkt.to_dict() for pkt in self.packets]
        return pd.DataFrame(data)
    
    def flows_to_dataframe(self):
        """Convert flows to pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for DataFrame conversion")
        
        features = self.get_flow_features()
        return pd.DataFrame(features)


class PCAPBatchProcessor:
    """Process multiple PCAP files in batch."""
    
    def __init__(self, backend: str = 'auto'):
        self.handler = PCAPHandler(backend=backend)
        self.results = []
    
    def process_directory(self, directory: str, pattern: str = '*.pcap') -> List[Dict[str, Any]]:
        """
        Process all PCAP files in a directory.
        
        Args:
            directory: Path to directory containing PCAP files
            pattern: Glob pattern for matching files
            
        Returns:
            List of processing results
        """
        import glob
        
        files = glob.glob(os.path.join(directory, pattern))
        files.extend(glob.glob(os.path.join(directory, '*.pcapng')))
        
        results = []
        
        for filepath in files:
            try:
                logger.info(f"Processing: {filepath}")
                packets = self.handler.read_pcap(filepath)
                flows = self.handler.extract_flows(packets)
                stats = self.handler.get_statistics()
                
                results.append({
                    'file': filepath,
                    'status': 'success',
                    'packets': len(packets),
                    'flows': len(flows),
                    'statistics': stats
                })
                
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                results.append({
                    'file': filepath,
                    'status': 'error',
                    'error': str(e)
                })
        
        self.results = results
        return results
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """Get combined features from all processed files."""
        return self.handler.get_flow_features()


# Convenience function
def analyze_pcap(filepath: str, max_packets: Optional[int] = None) -> Dict[str, Any]:
    """
    Quick analysis of a PCAP file.
    
    Args:
        filepath: Path to PCAP file
        max_packets: Maximum packets to analyze
        
    Returns:
        Analysis results dictionary
    """
    handler = PCAPHandler()
    packets = handler.read_pcap(filepath, max_packets)
    flows = handler.extract_flows()
    stats = handler.get_statistics()
    
    return {
        'statistics': stats,
        'flows': len(flows),
        'features': handler.get_flow_features()
    }


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pcap_file = sys.argv[1]
        print(f"Analyzing: {pcap_file}")
        
        result = analyze_pcap(pcap_file)
        print(f"\nStatistics:")
        for key, value in result['statistics'].items():
            print(f"  {key}: {value}")
        
        print(f"\nTotal flows: {result['flows']}")
        print(f"Feature vectors: {len(result['features'])}")
    else:
        print("Usage: python pcap_handler.py <pcap_file>")
