"""
AI-NIDS Test Configuration
==========================
Pytest fixtures and configuration
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from app.models.database import db, User


@pytest.fixture(scope='session')
def app():
    """Create application for testing."""
    # Use testing configuration
    os.environ['FLASK_ENV'] = 'testing'
    
    app = create_app('testing')
    
    # Create test database
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Create test client."""
    return app.test_client()


@pytest.fixture(scope='function')
def runner(app):
    """Create CLI test runner."""
    return app.test_cli_runner()


@pytest.fixture(scope='function')
def auth_client(app, client):
    """Create authenticated test client."""
    with app.app_context():
        # Create test user
        user = User(
            username='testuser',
            email='test@example.com',
            role='analyst'
        )
        user.set_password('testpassword123')
        db.session.add(user)
        db.session.commit()
        
        # Login
        client.post('/auth/login', data={
            'username': 'testuser',
            'password': 'testpassword123'
        }, follow_redirects=True)
        
        yield client
        
        # Cleanup
        db.session.delete(user)
        db.session.commit()


@pytest.fixture(scope='function')
def admin_client(app, client):
    """Create admin authenticated test client."""
    with app.app_context():
        # Create admin user
        admin = User(
            username='admin',
            email='admin@example.com',
            role='admin'
        )
        admin.set_password('adminpassword123')
        db.session.add(admin)
        db.session.commit()
        
        # Login
        client.post('/auth/login', data={
            'username': 'admin',
            'password': 'adminpassword123'
        }, follow_redirects=True)
        
        yield client
        
        # Cleanup
        db.session.delete(admin)
        db.session.commit()


@pytest.fixture
def sample_network_flow():
    """Create sample network flow data."""
    return {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.50',
        'src_port': 54321,
        'dst_port': 443,
        'protocol': 'TCP',
        'bytes_sent': 1500,
        'bytes_recv': 45000,
        'duration': 5.2,
        'packets': 120
    }


@pytest.fixture
def sample_alert_data():
    """Create sample alert data."""
    return {
        'source_ip': '192.168.1.100',
        'dest_ip': '10.0.0.50',
        'source_port': 54321,
        'dest_port': 22,
        'attack_type': 'brute_force',
        'severity': 'high',
        'confidence': 0.92,
        'description': 'SSH brute force attack detected'
    }


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for models."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
