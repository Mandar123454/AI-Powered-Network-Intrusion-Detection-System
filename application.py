"""
Azure App Service Entry Point
==============================
This file is auto-detected by Azure App Service for Python apps.
"""

import os
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set production environment
os.environ.setdefault('FLASK_ENV', 'production')
os.environ.setdefault('SECRET_KEY', 'azure-production-secret-key-2024')

from app import create_app

# Create the Flask application
app = create_app()
application = app  # WSGI standard name

if __name__ == '__main__':
    app.run()
