"""
Log Collectors Package for AI-NIDS

This package provides comprehensive network traffic collection capabilities:
- Suricata: Signature-based IDS alert parsing
- Zeek: Network metadata and flow log parsing
- PCAP: Offline packet capture file processing
- Live Capture: Real-time packet sniffing with Scapy
"""

from .suricata_parser import SuricataParser, SuricataFlow, create_suricata_parser
from .zeek_parser import ZeekParser, ZeekConn, create_zeek_parser
from .pcap_handler import PCAPHandler, create_pcap_handler
from .live_capture import LiveCapture, IntrusionDetector, create_live_capture

__all__ = [
    # Suricata Parser
    'SuricataParser', 'SuricataFlow', 'create_suricata_parser',
    # Zeek Parser
    'ZeekParser', 'ZeekConn', 'create_zeek_parser',
    # PCAP Handler
    'PCAPHandler', 'create_pcap_handler',
    # Live Capture
    'LiveCapture', 'IntrusionDetector', 'create_live_capture'
]
