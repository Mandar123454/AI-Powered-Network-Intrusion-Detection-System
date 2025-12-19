"""
IOC (Indicators of Compromise) Feeds for AI-NIDS

This module provides interfaces to multiple threat intelligence feeds:
- AlienVault OTX (Open Threat Exchange)
- AbuseIPDB
- VirusTotal
- FireHOL IP Lists
- Emerging Threats
- Spamhaus
- CISA Known Exploited Vulnerabilities

Each feed provides:
- IP addresses
- Domains
- File hashes
- URLs
- Threat scores

Author: AI-NIDS Team
"""

import asyncio
import aiohttp
import hashlib
import ipaddress
import json
import logging
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class IndicatorType(Enum):
    """Types of indicators of compromise."""
    IP = "ip"
    DOMAIN = "domain"
    URL = "url"
    HASH_MD5 = "hash_md5"
    HASH_SHA1 = "hash_sha1"
    HASH_SHA256 = "hash_sha256"
    EMAIL = "email"
    CVE = "cve"
    JA3 = "ja3"
    USER_AGENT = "user_agent"


class ThreatCategory(Enum):
    """Categories of threats."""
    MALWARE = "malware"
    BOTNET = "botnet"
    C2 = "command_and_control"
    PHISHING = "phishing"
    SPAM = "spam"
    SCANNER = "scanner"
    BRUTE_FORCE = "brute_force"
    EXPLOIT = "exploit"
    APT = "apt"
    RANSOMWARE = "ransomware"
    CRYPTOMINER = "cryptominer"
    TOR_EXIT = "tor_exit"
    PROXY = "proxy"
    VPN = "vpn"
    UNKNOWN = "unknown"


@dataclass
class IOCEntry:
    """Represents a single IOC entry."""
    indicator: str
    indicator_type: IndicatorType
    category: ThreatCategory
    confidence: float  # 0.0 - 1.0
    severity: float    # 0.0 - 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'indicator': self.indicator,
            'indicator_type': self.indicator_type.value,
            'category': self.category.value,
            'confidence': self.confidence,
            'severity': self.severity,
            'source': self.source,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'description': self.description,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IOCEntry':
        """Create from dictionary."""
        return cls(
            indicator=data['indicator'],
            indicator_type=IndicatorType(data['indicator_type']),
            category=ThreatCategory(data['category']),
            confidence=data['confidence'],
            severity=data['severity'],
            source=data['source'],
            first_seen=datetime.fromisoformat(data['first_seen']),
            last_seen=datetime.fromisoformat(data['last_seen']),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )


class IOCCache:
    """SQLite-based cache for IOC entries."""
    
    def __init__(self, db_path: str = "data/ioc_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ioc_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator TEXT NOT NULL,
                    indicator_type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity REAL NOT NULL,
                    source TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(indicator, source)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_indicator ON ioc_entries(indicator)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_indicator_type ON ioc_entries(indicator_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON ioc_entries(source)
            """)
            conn.commit()
    
    def add(self, entry: IOCEntry):
        """Add an IOC entry to cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ioc_entries
                (indicator, indicator_type, category, confidence, severity,
                 source, first_seen, last_seen, description, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.indicator,
                entry.indicator_type.value,
                entry.category.value,
                entry.confidence,
                entry.severity,
                entry.source,
                entry.first_seen.isoformat(),
                entry.last_seen.isoformat(),
                entry.description,
                json.dumps(entry.tags),
                json.dumps(entry.metadata)
            ))
            conn.commit()
    
    def add_batch(self, entries: List[IOCEntry]):
        """Add multiple IOC entries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO ioc_entries
                (indicator, indicator_type, category, confidence, severity,
                 source, first_seen, last_seen, description, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    e.indicator, e.indicator_type.value, e.category.value,
                    e.confidence, e.severity, e.source,
                    e.first_seen.isoformat(), e.last_seen.isoformat(),
                    e.description, json.dumps(e.tags), json.dumps(e.metadata)
                )
                for e in entries
            ])
            conn.commit()
    
    def lookup(self, indicator: str) -> List[IOCEntry]:
        """Look up an indicator."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ioc_entries WHERE indicator = ?
            """, (indicator,))
            rows = cursor.fetchall()
            return [self._row_to_entry(row) for row in rows]
    
    def lookup_by_type(self, indicator_type: IndicatorType) -> List[IOCEntry]:
        """Get all indicators of a specific type."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM ioc_entries WHERE indicator_type = ?
            """, (indicator_type.value,))
            rows = cursor.fetchall()
            return [self._row_to_entry(row) for row in rows]
    
    def get_all_ips(self) -> Set[str]:
        """Get all malicious IPs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT indicator FROM ioc_entries
                WHERE indicator_type = 'ip'
            """)
            return {row[0] for row in cursor.fetchall()}
    
    def get_all_domains(self) -> Set[str]:
        """Get all malicious domains."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT DISTINCT indicator FROM ioc_entries
                WHERE indicator_type = 'domain'
            """)
            return {row[0] for row in cursor.fetchall()}
    
    def _row_to_entry(self, row: sqlite3.Row) -> IOCEntry:
        """Convert a database row to IOCEntry."""
        return IOCEntry(
            indicator=row['indicator'],
            indicator_type=IndicatorType(row['indicator_type']),
            category=ThreatCategory(row['category']),
            confidence=row['confidence'],
            severity=row['severity'],
            source=row['source'],
            first_seen=datetime.fromisoformat(row['first_seen']),
            last_seen=datetime.fromisoformat(row['last_seen']),
            description=row['description'] or '',
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def clear_source(self, source: str):
        """Clear all entries from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ioc_entries WHERE source = ?", (source,))
            conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM ioc_entries").fetchone()[0]
            by_type = dict(conn.execute("""
                SELECT indicator_type, COUNT(*) FROM ioc_entries
                GROUP BY indicator_type
            """).fetchall())
            by_source = dict(conn.execute("""
                SELECT source, COUNT(*) FROM ioc_entries
                GROUP BY source
            """).fetchall())
            return {
                'total_entries': total,
                'by_type': by_type,
                'by_source': by_source
            }


class IOCFeed(ABC):
    """Abstract base class for IOC feeds."""
    
    def __init__(self, api_key: Optional[str] = None, cache: Optional[IOCCache] = None):
        self.api_key = api_key
        self.cache = cache or IOCCache()
        self.name = self.__class__.__name__
        self.last_update: Optional[datetime] = None
        self.update_interval = timedelta(hours=1)
        self._session: Optional[aiohttp.ClientSession] = None
    
    @abstractmethod
    async def fetch(self) -> List[IOCEntry]:
        """Fetch IOCs from the feed."""
        pass
    
    @abstractmethod
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Look up a specific indicator."""
        pass
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def needs_update(self) -> bool:
        """Check if the feed needs updating."""
        if self.last_update is None:
            return True
        return datetime.now() - self.last_update > self.update_interval
    
    async def update(self) -> int:
        """Update the feed and cache results."""
        try:
            entries = await self.fetch()
            if entries:
                self.cache.add_batch(entries)
                self.last_update = datetime.now()
                logger.info(f"{self.name}: Updated {len(entries)} IOCs")
            return len(entries)
        except Exception as e:
            logger.error(f"{self.name}: Update failed - {e}")
            return 0


class AlienVaultOTX(IOCFeed):
    """AlienVault Open Threat Exchange feed."""
    
    BASE_URL = "https://otx.alienvault.com/api/v1"
    
    def __init__(self, api_key: str, cache: Optional[IOCCache] = None):
        super().__init__(api_key, cache)
        self.update_interval = timedelta(hours=1)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch pulses from OTX."""
        entries = []
        session = await self._get_session()
        
        headers = {"X-OTX-API-KEY": self.api_key}
        
        try:
            # Get subscribed pulses
            async with session.get(
                f"{self.BASE_URL}/pulses/subscribed",
                headers=headers,
                params={"modified_since": (datetime.now() - timedelta(days=7)).isoformat()}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for pulse in data.get('results', []):
                        entries.extend(self._parse_pulse(pulse))
        except Exception as e:
            logger.error(f"OTX fetch error: {e}")
        
        return entries
    
    def _parse_pulse(self, pulse: Dict[str, Any]) -> List[IOCEntry]:
        """Parse a pulse into IOC entries."""
        entries = []
        now = datetime.now()
        
        pulse_name = pulse.get('name', 'Unknown')
        pulse_tags = pulse.get('tags', [])
        
        for indicator in pulse.get('indicators', []):
            ioc_type = self._map_indicator_type(indicator.get('type', ''))
            if ioc_type is None:
                continue
            
            entries.append(IOCEntry(
                indicator=indicator.get('indicator', ''),
                indicator_type=ioc_type,
                category=self._infer_category(pulse_tags),
                confidence=0.8,
                severity=0.7,
                source="AlienVault_OTX",
                first_seen=datetime.fromisoformat(
                    indicator.get('created', now.isoformat()).replace('Z', '+00:00')
                ) if indicator.get('created') else now,
                last_seen=now,
                description=pulse_name,
                tags=pulse_tags,
                metadata={'pulse_id': pulse.get('id')}
            ))
        
        return entries
    
    def _map_indicator_type(self, otx_type: str) -> Optional[IndicatorType]:
        """Map OTX indicator type to our type."""
        mapping = {
            'IPv4': IndicatorType.IP,
            'IPv6': IndicatorType.IP,
            'domain': IndicatorType.DOMAIN,
            'hostname': IndicatorType.DOMAIN,
            'URL': IndicatorType.URL,
            'FileHash-MD5': IndicatorType.HASH_MD5,
            'FileHash-SHA1': IndicatorType.HASH_SHA1,
            'FileHash-SHA256': IndicatorType.HASH_SHA256,
            'email': IndicatorType.EMAIL,
            'CVE': IndicatorType.CVE,
            'JA3': IndicatorType.JA3
        }
        return mapping.get(otx_type)
    
    def _infer_category(self, tags: List[str]) -> ThreatCategory:
        """Infer threat category from tags."""
        tag_str = ' '.join(tags).lower()
        
        if 'malware' in tag_str:
            return ThreatCategory.MALWARE
        elif 'botnet' in tag_str:
            return ThreatCategory.BOTNET
        elif 'c2' in tag_str or 'command' in tag_str:
            return ThreatCategory.C2
        elif 'phishing' in tag_str:
            return ThreatCategory.PHISHING
        elif 'ransomware' in tag_str:
            return ThreatCategory.RANSOMWARE
        elif 'apt' in tag_str:
            return ThreatCategory.APT
        elif 'exploit' in tag_str:
            return ThreatCategory.EXPLOIT
        else:
            return ThreatCategory.UNKNOWN
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Look up an indicator in OTX."""
        session = await self._get_session()
        headers = {"X-OTX-API-KEY": self.api_key}
        
        # Determine indicator type
        indicator_type = self._detect_indicator_type(indicator)
        if indicator_type is None:
            return None
        
        endpoint_map = {
            IndicatorType.IP: f"/indicators/IPv4/{indicator}/general",
            IndicatorType.DOMAIN: f"/indicators/domain/{indicator}/general",
            IndicatorType.HASH_MD5: f"/indicators/file/{indicator}/general",
            IndicatorType.HASH_SHA1: f"/indicators/file/{indicator}/general",
            IndicatorType.HASH_SHA256: f"/indicators/file/{indicator}/general",
        }
        
        endpoint = endpoint_map.get(indicator_type)
        if not endpoint:
            return None
        
        try:
            async with session.get(
                f"{self.BASE_URL}{endpoint}",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    pulse_count = data.get('pulse_info', {}).get('count', 0)
                    
                    if pulse_count > 0:
                        return IOCEntry(
                            indicator=indicator,
                            indicator_type=indicator_type,
                            category=ThreatCategory.UNKNOWN,
                            confidence=min(0.5 + (pulse_count * 0.1), 1.0),
                            severity=min(0.3 + (pulse_count * 0.1), 1.0),
                            source="AlienVault_OTX",
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            description=f"Found in {pulse_count} OTX pulses",
                            metadata={'pulse_count': pulse_count}
                        )
        except Exception as e:
            logger.error(f"OTX lookup error: {e}")
        
        return None
    
    def _detect_indicator_type(self, indicator: str) -> Optional[IndicatorType]:
        """Detect the type of indicator."""
        # IP address
        try:
            ipaddress.ip_address(indicator)
            return IndicatorType.IP
        except ValueError:
            pass
        
        # Hash
        if re.match(r'^[a-fA-F0-9]{32}$', indicator):
            return IndicatorType.HASH_MD5
        elif re.match(r'^[a-fA-F0-9]{40}$', indicator):
            return IndicatorType.HASH_SHA1
        elif re.match(r'^[a-fA-F0-9]{64}$', indicator):
            return IndicatorType.HASH_SHA256
        
        # Domain
        if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$', indicator):
            return IndicatorType.DOMAIN
        
        return None


class AbuseIPDB(IOCFeed):
    """AbuseIPDB threat intelligence feed."""
    
    BASE_URL = "https://api.abuseipdb.com/api/v2"
    
    def __init__(self, api_key: str, cache: Optional[IOCCache] = None):
        super().__init__(api_key, cache)
        self.update_interval = timedelta(hours=24)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch blacklisted IPs."""
        entries = []
        session = await self._get_session()
        
        headers = {
            "Key": self.api_key,
            "Accept": "application/json"
        }
        
        try:
            async with session.get(
                f"{self.BASE_URL}/blacklist",
                headers=headers,
                params={"confidenceMinimum": 75}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    now = datetime.now()
                    
                    for item in data.get('data', []):
                        entries.append(IOCEntry(
                            indicator=item['ipAddress'],
                            indicator_type=IndicatorType.IP,
                            category=self._map_category(item.get('abuseCategories', [])),
                            confidence=item.get('abuseConfidenceScore', 0) / 100.0,
                            severity=item.get('abuseConfidenceScore', 0) / 100.0,
                            source="AbuseIPDB",
                            first_seen=now,
                            last_seen=now,
                            description=f"Country: {item.get('countryCode', 'Unknown')}",
                            metadata={
                                'country': item.get('countryCode'),
                                'total_reports': item.get('totalReports', 0)
                            }
                        ))
        except Exception as e:
            logger.error(f"AbuseIPDB fetch error: {e}")
        
        return entries
    
    def _map_category(self, categories: List[int]) -> ThreatCategory:
        """Map AbuseIPDB categories."""
        category_map = {
            3: ThreatCategory.BRUTE_FORCE,  # Fraud Orders
            4: ThreatCategory.SCANNER,       # DDoS Attack
            5: ThreatCategory.BRUTE_FORCE,   # FTP Brute-Force
            6: ThreatCategory.SCANNER,       # Ping of Death
            7: ThreatCategory.PHISHING,      # Phishing
            9: ThreatCategory.PROXY,         # Open Proxy
            10: ThreatCategory.SPAM,         # Web Spam
            11: ThreatCategory.SPAM,         # Email Spam
            14: ThreatCategory.SCANNER,      # Port Scan
            15: ThreatCategory.EXPLOIT,      # Hacking
            18: ThreatCategory.BRUTE_FORCE,  # Brute-Force
            19: ThreatCategory.EXPLOIT,      # Bad Web Bot
            20: ThreatCategory.EXPLOIT,      # Exploited Host
            21: ThreatCategory.EXPLOIT,      # Web App Attack
            22: ThreatCategory.BRUTE_FORCE,  # SSH
            23: ThreatCategory.EXPLOIT,      # IoT Targeted
        }
        
        for cat in categories:
            if cat in category_map:
                return category_map[cat]
        return ThreatCategory.UNKNOWN
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Check an IP against AbuseIPDB."""
        try:
            ipaddress.ip_address(indicator)
        except ValueError:
            return None  # Not an IP
        
        session = await self._get_session()
        headers = {
            "Key": self.api_key,
            "Accept": "application/json"
        }
        
        try:
            async with session.get(
                f"{self.BASE_URL}/check",
                headers=headers,
                params={"ipAddress": indicator, "maxAgeInDays": 90}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get('data', {})
                    
                    score = result.get('abuseConfidenceScore', 0)
                    if score > 0:
                        return IOCEntry(
                            indicator=indicator,
                            indicator_type=IndicatorType.IP,
                            category=ThreatCategory.UNKNOWN,
                            confidence=score / 100.0,
                            severity=score / 100.0,
                            source="AbuseIPDB",
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            description=f"Abuse score: {score}%, Reports: {result.get('totalReports', 0)}",
                            metadata={
                                'country': result.get('countryCode'),
                                'isp': result.get('isp'),
                                'domain': result.get('domain'),
                                'total_reports': result.get('totalReports'),
                                'is_tor': result.get('isTor', False),
                                'is_public': result.get('isPublic', False)
                            }
                        )
        except Exception as e:
            logger.error(f"AbuseIPDB lookup error: {e}")
        
        return None


class VirusTotal(IOCFeed):
    """VirusTotal threat intelligence feed."""
    
    BASE_URL = "https://www.virustotal.com/api/v3"
    
    def __init__(self, api_key: str, cache: Optional[IOCCache] = None):
        super().__init__(api_key, cache)
        self.update_interval = timedelta(hours=6)
        self._rate_limit_delay = 15  # Free API limit: 4 requests/min
    
    async def fetch(self) -> List[IOCEntry]:
        """VirusTotal doesn't have a bulk feed - use lookup instead."""
        return []
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Look up an indicator in VirusTotal."""
        session = await self._get_session()
        headers = {"x-apikey": self.api_key}
        
        indicator_type = self._detect_type(indicator)
        if indicator_type is None:
            return None
        
        endpoint_map = {
            IndicatorType.IP: f"/ip_addresses/{indicator}",
            IndicatorType.DOMAIN: f"/domains/{indicator}",
            IndicatorType.HASH_MD5: f"/files/{indicator}",
            IndicatorType.HASH_SHA1: f"/files/{indicator}",
            IndicatorType.HASH_SHA256: f"/files/{indicator}",
            IndicatorType.URL: f"/urls/{self._url_id(indicator)}"
        }
        
        endpoint = endpoint_map.get(indicator_type)
        if not endpoint:
            return None
        
        try:
            async with session.get(
                f"{self.BASE_URL}{endpoint}",
                headers=headers
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    attrs = data.get('data', {}).get('attributes', {})
                    stats = attrs.get('last_analysis_stats', {})
                    
                    malicious = stats.get('malicious', 0)
                    total = sum(stats.values()) or 1
                    detection_ratio = malicious / total
                    
                    if malicious > 0:
                        return IOCEntry(
                            indicator=indicator,
                            indicator_type=indicator_type,
                            category=ThreatCategory.MALWARE if detection_ratio > 0.3 else ThreatCategory.UNKNOWN,
                            confidence=min(detection_ratio * 2, 1.0),
                            severity=detection_ratio,
                            source="VirusTotal",
                            first_seen=datetime.now(),
                            last_seen=datetime.now(),
                            description=f"{malicious}/{total} vendors flagged as malicious",
                            metadata={
                                'malicious': malicious,
                                'suspicious': stats.get('suspicious', 0),
                                'harmless': stats.get('harmless', 0),
                                'undetected': stats.get('undetected', 0),
                                'total_engines': total
                            }
                        )
        except Exception as e:
            logger.error(f"VirusTotal lookup error: {e}")
        
        return None
    
    def _detect_type(self, indicator: str) -> Optional[IndicatorType]:
        """Detect indicator type."""
        try:
            ipaddress.ip_address(indicator)
            return IndicatorType.IP
        except ValueError:
            pass
        
        if re.match(r'^[a-fA-F0-9]{32}$', indicator):
            return IndicatorType.HASH_MD5
        elif re.match(r'^[a-fA-F0-9]{40}$', indicator):
            return IndicatorType.HASH_SHA1
        elif re.match(r'^[a-fA-F0-9]{64}$', indicator):
            return IndicatorType.HASH_SHA256
        elif indicator.startswith(('http://', 'https://')):
            return IndicatorType.URL
        elif re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$', indicator):
            return IndicatorType.DOMAIN
        
        return None
    
    def _url_id(self, url: str) -> str:
        """Generate VirusTotal URL ID."""
        import base64
        return base64.urlsafe_b64encode(url.encode()).decode().rstrip('=')


class FireHOL(IOCFeed):
    """FireHOL IP blocklists."""
    
    LISTS = {
        'level1': 'https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level1.netset',
        'level2': 'https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level2.netset',
        'level3': 'https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/firehol_level3.netset',
        'spamhaus_drop': 'https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/spamhaus_drop.netset',
        'dshield': 'https://raw.githubusercontent.com/firehol/blocklist-ipsets/master/dshield.netset',
    }
    
    def __init__(self, cache: Optional[IOCCache] = None):
        super().__init__(None, cache)
        self.update_interval = timedelta(hours=6)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch FireHOL blocklists."""
        entries = []
        session = await self._get_session()
        now = datetime.now()
        
        for list_name, url in self.LISTS.items():
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith('#'):
                                # Handle CIDR notation
                                ip = line.split('/')[0]
                                try:
                                    ipaddress.ip_address(ip)
                                    entries.append(IOCEntry(
                                        indicator=line,
                                        indicator_type=IndicatorType.IP,
                                        category=ThreatCategory.MALWARE,
                                        confidence=0.9,
                                        severity=0.8,
                                        source=f"FireHOL_{list_name}",
                                        first_seen=now,
                                        last_seen=now,
                                        description=f"FireHOL {list_name} blocklist"
                                    ))
                                except ValueError:
                                    continue
            except Exception as e:
                logger.error(f"FireHOL {list_name} fetch error: {e}")
        
        return entries
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Check if IP is in cached blocklists."""
        cached = self.cache.lookup(indicator)
        if cached:
            return cached[0]
        return None


class EmergingThreats(IOCFeed):
    """Emerging Threats IP blocklist."""
    
    URL = "https://rules.emergingthreats.net/fwrules/emerging-Block-IPs.txt"
    
    def __init__(self, cache: Optional[IOCCache] = None):
        super().__init__(None, cache)
        self.update_interval = timedelta(hours=6)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch Emerging Threats blocklist."""
        entries = []
        session = await self._get_session()
        now = datetime.now()
        
        try:
            async with session.get(self.URL) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                ipaddress.ip_address(line)
                                entries.append(IOCEntry(
                                    indicator=line,
                                    indicator_type=IndicatorType.IP,
                                    category=ThreatCategory.MALWARE,
                                    confidence=0.85,
                                    severity=0.75,
                                    source="EmergingThreats",
                                    first_seen=now,
                                    last_seen=now,
                                    description="Emerging Threats blocklist"
                                ))
                            except ValueError:
                                continue
        except Exception as e:
            logger.error(f"Emerging Threats fetch error: {e}")
        
        return entries
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Check if IP is in cached blocklist."""
        cached = self.cache.lookup(indicator)
        for entry in cached:
            if entry.source == "EmergingThreats":
                return entry
        return None


class Spamhaus(IOCFeed):
    """Spamhaus DROP/EDROP lists."""
    
    LISTS = {
        'drop': 'https://www.spamhaus.org/drop/drop.txt',
        'edrop': 'https://www.spamhaus.org/drop/edrop.txt',
    }
    
    def __init__(self, cache: Optional[IOCCache] = None):
        super().__init__(None, cache)
        self.update_interval = timedelta(hours=12)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch Spamhaus DROP lists."""
        entries = []
        session = await self._get_session()
        now = datetime.now()
        
        for list_name, url in self.LISTS.items():
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.text()
                        for line in content.split('\n'):
                            line = line.strip()
                            if line and not line.startswith(';'):
                                # Format: "CIDR ; SBL reference"
                                parts = line.split(';')
                                if parts:
                                    cidr = parts[0].strip()
                                    entries.append(IOCEntry(
                                        indicator=cidr,
                                        indicator_type=IndicatorType.IP,
                                        category=ThreatCategory.SPAM,
                                        confidence=0.95,
                                        severity=0.9,
                                        source=f"Spamhaus_{list_name.upper()}",
                                        first_seen=now,
                                        last_seen=now,
                                        description=f"Spamhaus {list_name.upper()} list"
                                    ))
            except Exception as e:
                logger.error(f"Spamhaus {list_name} fetch error: {e}")
        
        return entries
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Check if IP is in Spamhaus lists."""
        cached = self.cache.lookup(indicator)
        for entry in cached:
            if 'Spamhaus' in entry.source:
                return entry
        return None


class CISAFeed(IOCFeed):
    """CISA Known Exploited Vulnerabilities feed."""
    
    URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
    
    def __init__(self, cache: Optional[IOCCache] = None):
        super().__init__(None, cache)
        self.update_interval = timedelta(hours=24)
    
    async def fetch(self) -> List[IOCEntry]:
        """Fetch CISA KEV catalog."""
        entries = []
        session = await self._get_session()
        now = datetime.now()
        
        try:
            async with session.get(self.URL) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    for vuln in data.get('vulnerabilities', []):
                        entries.append(IOCEntry(
                            indicator=vuln.get('cveID', ''),
                            indicator_type=IndicatorType.CVE,
                            category=ThreatCategory.EXPLOIT,
                            confidence=1.0,  # CISA verified
                            severity=0.9,
                            source="CISA_KEV",
                            first_seen=datetime.strptime(
                                vuln.get('dateAdded', now.strftime('%Y-%m-%d')),
                                '%Y-%m-%d'
                            ),
                            last_seen=now,
                            description=vuln.get('shortDescription', ''),
                            tags=[vuln.get('vendorProject', ''), vuln.get('product', '')],
                            metadata={
                                'vendor': vuln.get('vendorProject'),
                                'product': vuln.get('product'),
                                'vulnerability_name': vuln.get('vulnerabilityName'),
                                'due_date': vuln.get('dueDate'),
                                'required_action': vuln.get('requiredAction')
                            }
                        ))
        except Exception as e:
            logger.error(f"CISA KEV fetch error: {e}")
        
        return entries
    
    async def lookup(self, indicator: str) -> Optional[IOCEntry]:
        """Look up a CVE in CISA KEV."""
        cached = self.cache.lookup(indicator)
        for entry in cached:
            if entry.source == "CISA_KEV":
                return entry
        return None


class FeedManager:
    """Manages multiple IOC feeds."""
    
    def __init__(self, cache: Optional[IOCCache] = None):
        self.cache = cache or IOCCache()
        self.feeds: Dict[str, IOCFeed] = {}
    
    def add_feed(self, name: str, feed: IOCFeed):
        """Add a feed to the manager."""
        self.feeds[name] = feed
    
    def remove_feed(self, name: str):
        """Remove a feed."""
        if name in self.feeds:
            del self.feeds[name]
    
    async def update_all(self) -> Dict[str, int]:
        """Update all feeds."""
        results = {}
        for name, feed in self.feeds.items():
            if feed.needs_update():
                count = await feed.update()
                results[name] = count
        return results
    
    async def lookup(self, indicator: str) -> List[IOCEntry]:
        """Look up an indicator across all feeds."""
        # First check cache
        cached = self.cache.lookup(indicator)
        if cached:
            return cached
        
        # Query active feeds
        results = []
        for feed in self.feeds.values():
            result = await feed.lookup(indicator)
            if result:
                results.append(result)
                self.cache.add(result)
        
        return results
    
    async def is_malicious(self, indicator: str, threshold: float = 0.5) -> Tuple[bool, float]:
        """Check if an indicator is malicious."""
        entries = await self.lookup(indicator)
        if not entries:
            return False, 0.0
        
        # Calculate aggregate score
        max_confidence = max(e.confidence for e in entries)
        max_severity = max(e.severity for e in entries)
        score = (max_confidence + max_severity) / 2
        
        return score >= threshold, score
    
    async def close_all(self):
        """Close all feed sessions."""
        for feed in self.feeds.values():
            await feed.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all feeds."""
        return {
            'cache_stats': self.cache.get_stats(),
            'feeds': list(self.feeds.keys()),
            'feed_status': {
                name: {
                    'last_update': feed.last_update.isoformat() if feed.last_update else None,
                    'needs_update': feed.needs_update()
                }
                for name, feed in self.feeds.items()
            }
        }


def create_feed_manager(
    otx_api_key: Optional[str] = None,
    abuseipdb_api_key: Optional[str] = None,
    virustotal_api_key: Optional[str] = None,
    cache_path: str = "data/ioc_cache.db"
) -> FeedManager:
    """Create a feed manager with available feeds."""
    cache = IOCCache(cache_path)
    manager = FeedManager(cache)
    
    # Add API-based feeds if keys provided
    if otx_api_key:
        manager.add_feed('otx', AlienVaultOTX(otx_api_key, cache))
    
    if abuseipdb_api_key:
        manager.add_feed('abuseipdb', AbuseIPDB(abuseipdb_api_key, cache))
    
    if virustotal_api_key:
        manager.add_feed('virustotal', VirusTotal(virustotal_api_key, cache))
    
    # Add free feeds (no API key needed)
    manager.add_feed('firehol', FireHOL(cache))
    manager.add_feed('emerging_threats', EmergingThreats(cache))
    manager.add_feed('spamhaus', Spamhaus(cache))
    manager.add_feed('cisa', CISAFeed(cache))
    
    return manager


# Convenience functions
async def check_ip(ip: str, manager: FeedManager) -> Dict[str, Any]:
    """Quick IP reputation check."""
    is_bad, score = await manager.is_malicious(ip)
    entries = await manager.lookup(ip)
    
    return {
        'ip': ip,
        'is_malicious': is_bad,
        'threat_score': score,
        'sources': [e.source for e in entries],
        'categories': list(set(e.category.value for e in entries)),
        'details': [e.to_dict() for e in entries]
    }


async def check_domain(domain: str, manager: FeedManager) -> Dict[str, Any]:
    """Quick domain reputation check."""
    is_bad, score = await manager.is_malicious(domain)
    entries = await manager.lookup(domain)
    
    return {
        'domain': domain,
        'is_malicious': is_bad,
        'threat_score': score,
        'sources': [e.source for e in entries],
        'categories': list(set(e.category.value for e in entries)),
        'details': [e.to_dict() for e in entries]
    }
