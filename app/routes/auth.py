"""
Authentication Routes
=====================
User authentication, registration, and session management.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, Response
from flask_login import login_user, logout_user, login_required, current_user
from urllib.parse import urlparse as url_parse
import json
from datetime import datetime

from app import db, login_manager
from app.models.database import User, Alert, APIKey
from app.routes.forms import LoginForm, RegistrationForm, ChangePasswordForm

auth_bp = Blueprint('auth', __name__)


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    return User.query.get(int(user_id))


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    form = LoginForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
        
        if not user.is_active:
            flash('Your account has been deactivated', 'warning')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        user.last_login = db.func.now()
        db.session.commit()
        
        flash(f'Welcome back, {user.username}!', 'success')
        
        # Redirect to next page or dashboard
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('dashboard.dashboard')
        
        return redirect(next_page)
    
    return render_template('login.html', form=form)


@auth_bp.route('/logout')
@login_required
def logout():
    """User logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration (admin only in production)."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.dashboard'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            role='analyst'  # Default role
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('register.html', form=form)


@auth_bp.route('/profile')
@login_required
def profile():
    """User profile page."""
    from app.models.database import APIKey
    form = ChangePasswordForm()
    api_keys = APIKey.query.filter_by(user_id=current_user.id).order_by(APIKey.created_at.desc()).all()
    return render_template('profile.html', user=current_user, form=form, api_keys=api_keys)


@auth_bp.route('/profile/generate-api-key', methods=['POST'])
@login_required
def generate_api_key():
    """Generate a new API key for the current user."""
    from app.models.database import APIKey
    import secrets
    
    key_name = request.form.get('key_name', 'Default Key')
    
    # Generate a secure API key
    api_key = secrets.token_urlsafe(32)
    
    new_key = APIKey(
        key=api_key,
        name=key_name,
        user_id=current_user.id,
        is_active=True
    )
    
    db.session.add(new_key)
    db.session.commit()
    
    flash(f'API Key generated successfully! Key: {api_key}', 'success')
    return redirect(url_for('auth.profile'))


@auth_bp.route('/profile/revoke-api-key/<int:key_id>', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    """Revoke an API key."""
    from app.models.database import APIKey
    
    key = APIKey.query.filter_by(id=key_id, user_id=current_user.id).first()
    if key:
        db.session.delete(key)
        db.session.commit()
        flash('API Key revoked successfully.', 'success')
    else:
        flash('API Key not found.', 'danger')
    
    return redirect(url_for('auth.profile'))


@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password."""
    form = ChangePasswordForm()
    
    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash('Current password is incorrect', 'danger')
            return redirect(url_for('auth.change_password'))
        
        current_user.set_password(form.new_password.data)
        db.session.commit()
        
        flash('Password changed successfully!', 'success')
        return redirect(url_for('auth.profile'))
    
    return render_template('change_password.html', form=form)


@auth_bp.route('/users')
@login_required
def user_list():
    """List all users (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    users = User.query.all()
    return render_template('users.html', users=users)


@auth_bp.route('/users/<int:user_id>/toggle-active', methods=['POST'])
@login_required
def toggle_user_active(user_id):
    """Toggle user active status (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        flash('You cannot deactivate yourself.', 'warning')
        return redirect(url_for('auth.user_list'))
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.username} has been {status}.', 'success')
    
    return redirect(url_for('auth.user_list'))


@auth_bp.route('/users/<int:user_id>/change-role', methods=['POST'])
@login_required
def change_user_role(user_id):
    """Change user role (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    user = User.query.get_or_404(user_id)
    new_role = request.form.get('role')
    
    if new_role not in ['admin', 'analyst', 'viewer']:
        flash('Invalid role.', 'danger')
        return redirect(url_for('auth.user_list'))
    
    if user.id == current_user.id and new_role != 'admin':
        flash('You cannot demote yourself.', 'warning')
        return redirect(url_for('auth.user_list'))
    
    user.role = new_role
    db.session.commit()
    
    flash(f'User {user.username} role changed to {new_role}.', 'success')
    return redirect(url_for('auth.user_list'))


@auth_bp.route('/export-my-data')
@login_required
def export_my_data():
    """Export all user data as JSON."""
    # Collect user profile data
    user_data = {
        'export_date': datetime.utcnow().isoformat(),
        'user': {
            'username': current_user.username,
            'email': current_user.email,
            'role': current_user.role,
            'created_at': current_user.created_at.isoformat() if current_user.created_at else None,
            'last_login': current_user.last_login.isoformat() if current_user.last_login else None,
        },
        'api_keys': [],
        'alerts': []
    }
    
    # Get user's API keys (without exposing full keys)
    api_keys = APIKey.query.filter_by(user_id=current_user.id).all()
    for key in api_keys:
        user_data['api_keys'].append({
            'name': key.name,
            'created_at': key.created_at.isoformat() if key.created_at else None,
            'last_used': key.last_used.isoformat() if key.last_used else None,
            'expires_at': key.expires_at.isoformat() if key.expires_at else None,
            'is_active': key.is_active,
            'key_preview': f"{key.key[:8]}...{key.key[-4:]}" if key.key else None
        })
    
    # Get recent alerts (last 100)
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(100).all()
    for alert in alerts:
        user_data['alerts'].append({
            'id': alert.id,
            'attack_type': alert.attack_type,
            'severity': alert.severity,
            'source_ip': alert.source_ip,
            'destination_ip': alert.destination_ip,
            'timestamp': alert.timestamp.isoformat() if alert.timestamp else None,
            'acknowledged': alert.acknowledged,
            'resolved': alert.resolved,
            'description': alert.description
        })
    
    # Create downloadable JSON response
    response = Response(
        json.dumps(user_data, indent=2),
        mimetype='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=ai-nids-export-{current_user.username}-{datetime.utcnow().strftime("%Y%m%d")}.json'
        }
    )
    
    return response


@auth_bp.route('/settings')
@login_required
def settings():
    """System settings page (admin only)."""
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard.dashboard'))
    
    return render_template('settings.html')
