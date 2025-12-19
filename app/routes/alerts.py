"""
Alerts Routes
=============
Alert management views and operations.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timedelta
from sqlalchemy import func

from app import db
from app.models.database import Alert

alerts_bp = Blueprint('alerts', __name__)


@alerts_bp.route('/')
@login_required
def alert_list():
    """Display all alerts with filtering."""
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    
    # Filters
    severity = request.args.get('severity')
    attack_type = request.args.get('attack_type')
    status = request.args.get('status')  # all, unacknowledged, acknowledged, resolved
    search = request.args.get('search')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    # Build query
    query = Alert.query
    
    if severity:
        query = query.filter(Alert.severity == severity)
    
    if attack_type:
        query = query.filter(Alert.attack_type == attack_type)
    
    if status == 'unacknowledged':
        query = query.filter(Alert.acknowledged == False)
    elif status == 'acknowledged':
        query = query.filter(Alert.acknowledged == True, Alert.resolved == False)
    elif status == 'resolved':
        query = query.filter(Alert.resolved == True)
    
    if search:
        search_term = f'%{search}%'
        query = query.filter(
            db.or_(
                Alert.source_ip.like(search_term),
                Alert.destination_ip.like(search_term),
                Alert.description.like(search_term)
            )
        )
    
    if date_from:
        query = query.filter(Alert.timestamp >= datetime.strptime(date_from, '%Y-%m-%d'))
    
    if date_to:
        query = query.filter(Alert.timestamp <= datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1))
    
    # Order and paginate
    pagination = query.order_by(Alert.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Get filter options
    severities = ['critical', 'high', 'medium', 'low', 'info']
    attack_types = db.session.query(Alert.attack_type).distinct().all()
    attack_types = [at[0] for at in attack_types if at[0]]
    
    return render_template(
        'alerts.html',
        alerts=pagination.items,
        pagination=pagination,
        severities=severities,
        attack_types=attack_types,
        filters={
            'severity': severity,
            'attack_type': attack_type,
            'status': status,
            'search': search,
            'date_from': date_from,
            'date_to': date_to
        }
    )


@alerts_bp.route('/<int:alert_id>')
@login_required
def alert_detail(alert_id):
    """Display single alert details."""
    alert = Alert.query.get_or_404(alert_id)
    
    # Get SHAP explanation if available
    explanation = None
    if alert.shap_values:
        import json
        explanation = json.loads(alert.shap_values)
    
    # Get related alerts (same source IP in last 24 hours)
    related = Alert.query.filter(
        Alert.source_ip == alert.source_ip,
        Alert.id != alert.id,
        Alert.timestamp >= alert.timestamp - timedelta(hours=24)
    ).order_by(Alert.timestamp.desc()).limit(10).all()
    
    return render_template(
        'alert_detail.html',
        alert=alert,
        explanation=explanation,
        related_alerts=related
    )


@alerts_bp.route('/<int:alert_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge(alert_id):
    """Acknowledge an alert."""
    alert = Alert.query.get_or_404(alert_id)
    
    alert.acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    db.session.commit()
    
    flash(f'Alert #{alert.id} acknowledged.', 'success')
    
    # Return JSON for AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})
    
    return redirect(request.referrer or url_for('alerts.alert_list'))


@alerts_bp.route('/<int:alert_id>/resolve', methods=['GET', 'POST'])
@login_required
def resolve(alert_id):
    """Resolve an alert."""
    alert = Alert.query.get_or_404(alert_id)
    
    if request.method == 'POST':
        notes = request.form.get('notes', '')
        
        alert.resolved = True
        alert.resolved_by = current_user.id
        alert.resolved_at = datetime.utcnow()
        alert.resolution_notes = notes
        
        if not alert.acknowledged:
            alert.acknowledged = True
            alert.acknowledged_by = current_user.id
            alert.acknowledged_at = datetime.utcnow()
        
        db.session.commit()
        
        flash(f'Alert #{alert.id} resolved.', 'success')
        return redirect(url_for('alerts.alert_list'))
    
    return render_template('resolve_alert.html', alert=alert)


@alerts_bp.route('/<int:alert_id>/add-note', methods=['POST'])
@login_required
def add_note(alert_id):
    """Add a note to an alert."""
    alert = Alert.query.get_or_404(alert_id)
    
    note = request.form.get('note', '').strip()
    if note:
        # Append note to resolution_notes or create new
        existing_notes = alert.resolution_notes or ''
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
        new_note = f"[{timestamp}] {current_user.username}: {note}"
        
        if existing_notes:
            alert.resolution_notes = existing_notes + '\n' + new_note
        else:
            alert.resolution_notes = new_note
        
        db.session.commit()
        flash('Note added successfully.', 'success')
    else:
        flash('Note cannot be empty.', 'warning')
    
    return redirect(url_for('alerts.alert_detail', alert_id=alert_id))


@alerts_bp.route('/<int:alert_id>/delete', methods=['POST'])
@login_required
def delete_alert(alert_id):
    """Delete an alert (admin only)."""
    if current_user.role != 'admin':
        flash('Permission denied. Admin access required.', 'danger')
        return redirect(url_for('alerts.alert_list'))
    
    alert = Alert.query.get_or_404(alert_id)
    db.session.delete(alert)
    db.session.commit()
    
    flash(f'Alert #{alert_id} deleted.', 'success')
    return redirect(url_for('alerts.alert_list'))


@alerts_bp.route('/bulk-action', methods=['POST'])
@login_required
def bulk_action():
    """Perform bulk actions on alerts."""
    action = request.form.get('action')
    alert_ids = request.form.getlist('alert_ids')
    
    if not alert_ids:
        flash('No alerts selected.', 'warning')
        return redirect(url_for('alerts.alert_list'))
    
    alerts = Alert.query.filter(Alert.id.in_(alert_ids)).all()
    
    if action == 'acknowledge':
        for alert in alerts:
            if not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_by = current_user.id
                alert.acknowledged_at = datetime.utcnow()
        flash(f'{len(alerts)} alerts acknowledged.', 'success')
    
    elif action == 'resolve':
        for alert in alerts:
            if not alert.resolved:
                alert.resolved = True
                alert.resolved_by = current_user.id
                alert.resolved_at = datetime.utcnow()
                if not alert.acknowledged:
                    alert.acknowledged = True
                    alert.acknowledged_by = current_user.id
                    alert.acknowledged_at = datetime.utcnow()
        flash(f'{len(alerts)} alerts resolved.', 'success')
    
    elif action == 'delete' and current_user.role == 'admin':
        for alert in alerts:
            db.session.delete(alert)
        flash(f'{len(alerts)} alerts deleted.', 'success')
    
    db.session.commit()
    
    return redirect(url_for('alerts.alert_list'))


@alerts_bp.route('/export')
@login_required
def export_alerts():
    """Export alerts to CSV."""
    import csv
    from io import StringIO
    from flask import Response
    
    # Get filters from query params
    severity = request.args.get('severity')
    attack_type = request.args.get('attack_type')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    query = Alert.query
    
    if severity:
        query = query.filter(Alert.severity == severity)
    if attack_type:
        query = query.filter(Alert.attack_type == attack_type)
    if date_from:
        query = query.filter(Alert.timestamp >= datetime.strptime(date_from, '%Y-%m-%d'))
    if date_to:
        query = query.filter(Alert.timestamp <= datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1))
    
    alerts = query.order_by(Alert.timestamp.desc()).all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'ID', 'Timestamp', 'Source IP', 'Destination IP', 'Source Port',
        'Destination Port', 'Protocol', 'Attack Type', 'Severity',
        'Confidence', 'Description', 'Acknowledged', 'Resolved'
    ])
    
    # Data
    for alert in alerts:
        writer.writerow([
            alert.id,
            alert.timestamp.isoformat(),
            alert.source_ip,
            alert.destination_ip,
            alert.source_port,
            alert.destination_port,
            alert.protocol,
            alert.attack_type,
            alert.severity,
            alert.confidence,
            alert.description,
            alert.acknowledged,
            alert.resolved
        ])
    
    output.seek(0)
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        }
    )


@alerts_bp.route('/summary')
@login_required
def alert_summary():
    """Alert summary view."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    month_start = today_start - timedelta(days=30)
    
    # Check if we have data for today, if not use latest data available
    today_count = Alert.query.filter(Alert.timestamp >= today_start).count()
    
    if today_count == 0:
        # Find the latest alert and use that date instead
        latest_alert = Alert.query.order_by(Alert.timestamp.desc()).first()
        if latest_alert:
            now = latest_alert.timestamp
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_start = today_start - timedelta(days=7)
            month_start = today_start - timedelta(days=30)
    
    # Today's stats (or latest day with data)
    today_stats = {
        'total': Alert.query.filter(Alert.timestamp >= today_start).count(),
        'critical': Alert.query.filter(Alert.timestamp >= today_start, Alert.severity == 'critical').count(),
        'high': Alert.query.filter(Alert.timestamp >= today_start, Alert.severity == 'high').count(),
        'unacknowledged': Alert.query.filter(Alert.timestamp >= today_start, Alert.acknowledged == False).count()
    }
    
    # Weekly trend
    weekly_data = db.session.query(
        func.date(Alert.timestamp).label('date'),
        func.count().label('count')
    ).filter(
        Alert.timestamp >= week_start
    ).group_by(func.date(Alert.timestamp)).all()
    
    # If no weekly data, get all data
    if not weekly_data:
        weekly_data = db.session.query(
            func.date(Alert.timestamp).label('date'),
            func.count().label('count')
        ).group_by(func.date(Alert.timestamp)).order_by(func.date(Alert.timestamp).desc()).limit(7).all()
    
    # Attack type breakdown
    attack_breakdown = db.session.query(
        Alert.attack_type,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= month_start
    ).group_by(Alert.attack_type).order_by(func.count().desc()).all()
    
    # If no attack breakdown data, get all
    if not attack_breakdown:
        attack_breakdown = db.session.query(
            Alert.attack_type,
            func.count().label('count')
        ).group_by(Alert.attack_type).order_by(func.count().desc()).all()
    
    # Top source IPs
    top_sources = db.session.query(
        Alert.source_ip,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= week_start
    ).group_by(Alert.source_ip).order_by(func.count().desc()).limit(10).all()
    
    # If no top sources data, get all
    if not top_sources:
        top_sources = db.session.query(
            Alert.source_ip,
            func.count().label('count')
        ).group_by(Alert.source_ip).order_by(func.count().desc()).limit(10).all()
    
    return render_template(
        'alert_summary.html',
        today_stats=today_stats,
        weekly_data=[{'date': str(d.date), 'count': d.count} for d in weekly_data],
        attack_breakdown=[{'type': ab.attack_type or 'Unknown', 'count': ab.count} for ab in attack_breakdown],
        top_sources=[{'ip': ts.source_ip, 'count': ts.count} for ts in top_sources]
    )

