"""
Routes Package
==============
Blueprint registrations for all routes.
"""

from app.routes.dashboard import dashboard_bp
from app.routes.auth import auth_bp
from app.routes.api import api_bp
from app.routes.alerts import alerts_bp
from app.routes.analytics import analytics_bp
from app.routes.ai_models import ai_models_bp

__all__ = ['dashboard_bp', 'auth_bp', 'api_bp', 'alerts_bp', 'analytics_bp', 'ai_models_bp']

