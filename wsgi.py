"""
WSGI Entry Point
================
Production WSGI entry point for Gunicorn/uWSGI.

Usage with Gunicorn:
    gunicorn -c deployment/gunicorn.conf.py wsgi:application
"""

import os
from app import create_app

# Set production environment
os.environ.setdefault('FLASK_ENV', 'production')

# Create application instance
application = create_app()

# Alias for some WSGI servers
app = application

if __name__ == '__main__':
    application.run()
