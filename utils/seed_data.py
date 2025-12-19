"""
Data Seeder for AI-NIDS
======================
Generates realistic sample data for testing and demonstration.
This populates the database with:
- Network flows (normal and malicious)
- Security alerts (various attack types)
- System metrics

Run with: python -m utils.seed_data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
import random
import json
from app import create_app, db
from app.models.database import Alert, NetworkFlow, User, APIKey
from werkzeug.security import generate_password_hash

# Attack types with descriptions and MITRE ATT&CK mappings
ATTACK_TYPES = [
    {'name': 'DDoS', 'severity': 'critical', 'mitre': 'T1498', 'confidence_range': (0.92, 0.99)},
    {'name': 'Port Scan', 'severity': 'medium', 'mitre': 'T1046', 'confidence_range': (0.85, 0.95)},
    {'name': 'SQL Injection', 'severity': 'high', 'mitre': 'T1190', 'confidence_range': (0.88, 0.98)},
    {'name': 'Brute Force', 'severity': 'high', 'mitre': 'T1110', 'confidence_range': (0.90, 0.97)},
    {'name': 'Malware', 'severity': 'critical', 'mitre': 'T1204', 'confidence_range': (0.94, 0.99)},
    {'name': 'Phishing', 'severity': 'high', 'mitre': 'T1566', 'confidence_range': (0.87, 0.96)},
    {'name': 'Data Exfiltration', 'severity': 'critical', 'mitre': 'T1048', 'confidence_range': (0.91, 0.98)},
    {'name': 'C2 Communication', 'severity': 'critical', 'mitre': 'T1071', 'confidence_range': (0.93, 0.99)},
    {'name': 'Privilege Escalation', 'severity': 'high', 'mitre': 'T1068', 'confidence_range': (0.89, 0.97)},
    {'name': 'Lateral Movement', 'severity': 'high', 'mitre': 'T1021', 'confidence_range': (0.86, 0.95)},
    {'name': 'Ransomware', 'severity': 'critical', 'mitre': 'T1486', 'confidence_range': (0.95, 0.99)},
    {'name': 'Cryptomining', 'severity': 'medium', 'mitre': 'T1496', 'confidence_range': (0.88, 0.96)},
    {'name': 'DNS Tunneling', 'severity': 'medium', 'mitre': 'T1071.004', 'confidence_range': (0.84, 0.94)},
    {'name': 'XSS Attack', 'severity': 'medium', 'mitre': 'T1059.007', 'confidence_range': (0.86, 0.95)},
    {'name': 'Zero-Day Exploit', 'severity': 'critical', 'mitre': 'T1203', 'confidence_range': (0.96, 0.99)},
]

# Malicious IP addresses (fake)
MALICIOUS_IPS = [
    '185.220.101.42', '45.33.32.156', '89.248.165.12', '104.244.72.115',
    '192.42.116.16', '171.25.193.20', '95.211.230.211', '91.219.236.222',
    '185.129.61.1', '45.155.205.233', '194.26.29.113', '185.100.86.74',
    '162.247.74.27', '198.96.155.3', '77.247.181.163', '46.182.21.248',
    '199.249.230.89', '51.77.135.89', '185.220.102.8', '209.127.17.242'
]

# Internal network IPs
INTERNAL_IPS = [
    '192.168.1.10', '192.168.1.25', '192.168.1.50', '192.168.1.100',
    '192.168.1.105', '192.168.1.150', '192.168.1.200', '192.168.1.220',
    '10.0.0.5', '10.0.0.15', '10.0.0.25', '10.0.0.50',
    '172.16.0.10', '172.16.0.20', '172.16.0.30', '172.16.0.100'
]

# Protocols
PROTOCOLS = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP', 'SMTP']

# Common ports
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3306, 3389, 5432, 8080, 8443]


def generate_network_flows(app, count=5000, days=7):
    """Generate sample network flow data."""
    print(f"🔄 Generating {count} network flows over {days} days...")
    
    with app.app_context():
        flows_created = 0
        now = datetime.utcnow()
        
        for i in range(count):
            # Random timestamp within the time range
            random_hours = random.uniform(0, days * 24)
            timestamp = now - timedelta(hours=random_hours)
            
            # 85% normal traffic, 15% suspicious
            is_suspicious = random.random() < 0.15
            
            if is_suspicious:
                src_ip = random.choice(MALICIOUS_IPS)
                dst_ip = random.choice(INTERNAL_IPS)
                bytes_sent = random.randint(5000, 500000)
                bytes_recv = random.randint(100, 10000)
                duration = random.uniform(0.1, 30)
            else:
                src_ip = random.choice(INTERNAL_IPS)
                dst_ip = random.choice(['8.8.8.8', '1.1.1.1', '208.67.222.222', '9.9.9.9'] + INTERNAL_IPS)
                bytes_sent = random.randint(100, 50000)
                bytes_recv = random.randint(100, 100000)
                duration = random.uniform(0.01, 10)
            
            flow = NetworkFlow(
                source_ip=src_ip,
                destination_ip=dst_ip,
                source_port=random.randint(1024, 65535),
                destination_port=random.choice(COMMON_PORTS),
                protocol=random.choice(PROTOCOLS),
                duration=round(duration, 3),
                bytes_sent=bytes_sent,
                bytes_recv=bytes_recv,
                packets_sent=random.randint(1, 1000),
                packets_recv=random.randint(1, 1000),
                total_bytes=bytes_sent + bytes_recv,
                timestamp=timestamp,
                is_anomaly=is_suspicious
            )
            
            db.session.add(flow)
            flows_created += 1
            
            if flows_created % 500 == 0:
                db.session.commit()
                print(f"   Created {flows_created}/{count} flows...")
        
        db.session.commit()
        print(f"✅ Created {flows_created} network flows")
        return flows_created


def generate_alerts(app, count=500, days=7):
    """Generate sample security alerts."""
    print(f"🔄 Generating {count} security alerts over {days} days...")
    
    with app.app_context():
        alerts_created = 0
        now = datetime.utcnow()
        
        for i in range(count):
            # Random timestamp within the time range (more recent = more alerts)
            weight = random.betavariate(2, 5)  # More recent timestamps
            random_hours = weight * days * 24
            timestamp = now - timedelta(hours=random_hours)
            
            # Select attack type
            attack = random.choice(ATTACK_TYPES)
            
            # Generate confidence score
            min_conf, max_conf = attack['confidence_range']
            confidence = round(random.uniform(min_conf, max_conf), 4)
            
            # Some alerts are acknowledged/resolved
            is_acknowledged = random.random() < 0.4
            is_resolved = random.random() < 0.25 if is_acknowledged else False
            
            alert = Alert(
                source_ip=random.choice(MALICIOUS_IPS),
                destination_ip=random.choice(INTERNAL_IPS),
                source_port=random.randint(1024, 65535),
                destination_port=random.choice(COMMON_PORTS),
                protocol=random.choice(PROTOCOLS),
                attack_type=attack['name'],
                severity=attack['severity'],
                confidence=confidence,
                timestamp=timestamp,
                description=f"{attack['name']} attack detected from external source. MITRE ATT&CK: {attack['mitre']}",
                acknowledged=is_acknowledged,
                resolved=is_resolved,
                acknowledged_at=timestamp + timedelta(minutes=random.randint(5, 60)) if is_acknowledged else None,
                resolved_at=timestamp + timedelta(hours=random.randint(1, 12)) if is_resolved else None
            )
            
            db.session.add(alert)
            alerts_created += 1
            
            if alerts_created % 100 == 0:
                db.session.commit()
                print(f"   Created {alerts_created}/{count} alerts...")
        
        db.session.commit()
        print(f"✅ Created {alerts_created} security alerts")
        return alerts_created


def create_demo_user(app):
    """Create demo user if not exists."""
    print("🔄 Checking demo user...")
    
    with app.app_context():
        demo = User.query.filter_by(username='demo').first()
        if not demo:
            demo = User(
                username='demo',
                email='demo@ai-nids.local',
                password_hash=generate_password_hash('demo123'),
                role='admin',
                is_active=True
            )
            db.session.add(demo)
            db.session.commit()
            print("✅ Created demo user (username: demo, password: demo123)")
        else:
            print("✅ Demo user already exists")
        
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@ai-nids.local',
                password_hash=generate_password_hash('admin123'),
                role='admin',
                is_active=True
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Created admin user (username: admin, password: admin123)")
        else:
            print("✅ Admin user already exists")


def generate_api_keys(app):
    """Generate sample API keys for users."""
    print("🔑 Generating API keys...")
    
    with app.app_context():
        users = User.query.all()
        
        for user in users:
            # Check if user already has API keys
            existing_keys = APIKey.query.filter_by(user_id=user.id).count()
            if existing_keys > 0:
                print(f"   • User '{user.username}' already has {existing_keys} API key(s)")
                continue
            
            # Create 2-3 API keys per user
            key_configs = [
                {'name': 'Production API Key', 'expires_days': 365},
                {'name': 'Development Key', 'expires_days': 90},
                {'name': 'Testing Key', 'expires_days': 30},
            ]
            
            for config in key_configs[:random.randint(2, 3)]:
                api_key = APIKey.generate_key(
                    user_id=user.id,
                    name=config['name'],
                    expires_days=config['expires_days']
                )
                db.session.add(api_key)
            
            db.session.commit()
            print(f"   • Created API keys for user '{user.username}'")
        
        print("✅ API keys generated")


def clear_existing_data(app):
    """Clear existing sample data."""
    print("🔄 Clearing existing data...")
    
    with app.app_context():
        Alert.query.delete()
        NetworkFlow.query.delete()
        db.session.commit()
        print("✅ Cleared existing alerts and flows")


def seed_all(clear=False, flows=5000, alerts=500, days=7):
    """Seed all data."""
    print("\n" + "="*60)
    print("🛡️  AI-NIDS DATA SEEDER")
    print("="*60 + "\n")
    
    app = create_app()
    
    with app.app_context():
        db.create_all()
    
    if clear:
        clear_existing_data(app)
    
    create_demo_user(app)
    generate_api_keys(app)
    generate_network_flows(app, count=flows, days=days)
    generate_alerts(app, count=alerts, days=days)
    
    print("\n" + "="*60)
    print("✅ DATA SEEDING COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   • Network Flows: {flows}")
    print(f"   • Security Alerts: {alerts}")
    print(f"   • Time Range: Last {days} days")
    print(f"\n🔐 Demo Credentials:")
    print(f"   • Username: demo | Password: demo123")
    print(f"   • Username: admin | Password: admin123")
    print(f"\n🚀 Start the server with: python run.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Seed AI-NIDS database with sample data')
    parser.add_argument('--clear', action='store_true', help='Clear existing data before seeding')
    parser.add_argument('--flows', type=int, default=5000, help='Number of network flows to generate')
    parser.add_argument('--alerts', type=int, default=500, help='Number of alerts to generate')
    parser.add_argument('--days', type=int, default=7, help='Days of historical data to generate')
    
    args = parser.parse_args()
    
    seed_all(
        clear=args.clear,
        flows=args.flows,
        alerts=args.alerts,
        days=args.days
    )
