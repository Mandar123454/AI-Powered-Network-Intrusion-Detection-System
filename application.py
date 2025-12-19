"""
AI-NIDS Application Entry Point
================================
Main application entry point for running the Flask server.
"""

from app import create_app
from config import config

# Create the Flask application
app = create_app()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
