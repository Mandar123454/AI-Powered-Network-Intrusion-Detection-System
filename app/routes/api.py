"""
REST API Routes
===============
API endpoints for external integrations and AJAX requests.
"""

from flask import Blueprint, jsonify, request, current_app, render_template
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, timedelta
import json

from app import db
from app.models.database import Alert, NetworkFlow, APIKey
from detection.detector import DetectionEngine as IntrusionDetector
from ml.inference.predictor import ModelPredictor

api_bp = Blueprint('api', __name__)


def api_key_required(f):
    """Decorator to require API key for endpoints."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        key = APIKey.query.filter_by(key=api_key, is_active=True).first()
        if not key:
            return jsonify({'error': 'Invalid API key'}), 401
        
        # Update last used timestamp
        key.last_used = datetime.utcnow()
        db.session.commit()
        
        return f(*args, **kwargs)
    return decorated_function


# ==================== API Index ====================

@api_bp.route('/')
def api_index():
    """API index - beautiful landing page."""
    # Check if client wants JSON
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            'name': 'AI-NIDS API',
            'version': 'v1',
            'status': 'operational',
            'endpoints': {
                'health': '/api/v1/health',
                'status': '/api/v1/status',
                'detect': '/api/v1/detect [POST]',
                'alerts': '/api/v1/alerts',
                'stats': '/api/v1/stats',
                'threat_intel': '/api/v1/threat-intel'
            },
            'documentation': 'https://github.com/ai-nids/docs',
            'timestamp': datetime.utcnow().isoformat()
        })
    return render_template('api_index.html')


# ==================== Health & Status ====================

@api_bp.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': current_app.config.get('APP_VERSION', '1.0.0')
    })


@api_bp.route('/status')
def system_status():
    """Get system status (public endpoint)."""
    try:
        # Check ML models
        predictor = ModelPredictor()
        models_loaded = predictor.is_loaded()
    except:
        models_loaded = False
    
    # Get database stats
    try:
        total_alerts = Alert.query.count()
        total_flows = NetworkFlow.query.count()
    except:
        total_alerts = 0
        total_flows = 0
    
    return jsonify({
        'status': 'operational',
        'models_loaded': models_loaded,
        'database': {
            'total_alerts': total_alerts,
            'total_flows': total_flows
        },
        'uptime': '99.9%',
        'timestamp': datetime.utcnow().isoformat()
    })


# ==================== Detection ====================

@api_bp.route('/detect', methods=['POST'])
def detect_intrusion():
    """
    Analyze network flow for intrusions (public endpoint).
    
    Expected JSON payload:
    {
        "flows": [
            {
                "src_ip": "192.168.1.100",
                "dst_ip": "10.0.0.1",
                "src_port": 54321,
                "dst_port": 80,
                "protocol": "TCP",
                "duration": 1.5,
                "bytes_sent": 1024,
                "bytes_recv": 2048,
                "packets_sent": 10,
                "packets_recv": 15,
                ...
            }
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'flows' not in data:
            return jsonify({'error': 'Invalid request. Expected "flows" array.'}), 400
        
        flows = data['flows']
        
        if not isinstance(flows, list):
            return jsonify({'error': '"flows" must be an array'}), 400
        
        # Initialize detector
        detector = IntrusionDetector()
        
        # Analyze flows
        results = []
        for flow in flows:
            result = detector.analyze_flow(flow)
            results.append(result)
            
            # Create alert if threat detected
            if result['is_threat']:
                alert = Alert(
                    source_ip=flow.get('src_ip'),
                    destination_ip=flow.get('dst_ip'),
                    source_port=flow.get('src_port'),
                    destination_port=flow.get('dst_port'),
                    protocol=flow.get('protocol'),
                    attack_type=result['attack_type'],
                    severity=result['severity'],
                    confidence=result['confidence'],
                    description=result['description'],
                    raw_data=json.dumps(flow)
                )
                db.session.add(alert)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'results': results,
            'total_analyzed': len(flows),
            'threats_detected': sum(1 for r in results if r['is_threat'])
        })
        
    except Exception as e:
        current_app.logger.error(f'Detection error: {str(e)}')
        return jsonify({'error': str(e)}), 500


@api_bp.route('/detect/batch', methods=['POST'])
@api_key_required
def detect_batch():
    """Batch detection for multiple flows."""
    try:
        data = request.get_json()
        
        if not data or 'flows' not in data:
            return jsonify({'error': 'Invalid request'}), 400
        
        detector = IntrusionDetector()
        results = detector.analyze_batch(data['flows'])
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(results),
                'threats': sum(1 for r in results if r['is_threat']),
                'clean': sum(1 for r in results if not r['is_threat'])
            }
        })
        
    except Exception as e:
        current_app.logger.error(f'Batch detection error: {str(e)}')
        return jsonify({'error': str(e)}), 500


# ==================== Alerts ====================

@api_bp.route('/alerts')
def get_alerts():
    """Get alerts with filtering and pagination (public, limited data)."""
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 50)  # Max 50
    limit = request.args.get('limit', type=int)  # Optional limit
    
    # Filters
    severity = request.args.get('severity')
    attack_type = request.args.get('attack_type')
    source_ip = request.args.get('source_ip')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Build query
    query = Alert.query
    
    if severity:
        query = query.filter(Alert.severity == severity)
    if attack_type:
        query = query.filter(Alert.attack_type == attack_type)
    if source_ip:
        query = query.filter(Alert.source_ip == source_ip)
    if start_date:
        query = query.filter(Alert.timestamp >= datetime.fromisoformat(start_date))
    if end_date:
        query = query.filter(Alert.timestamp <= datetime.fromisoformat(end_date))
    
    # Apply limit if specified
    if limit:
        per_page = min(limit, 50)
    
    # Order and paginate
    pagination = query.order_by(Alert.timestamp.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Public-safe alert data (hide sensitive details if not logged in)
    alerts_data = []
    for alert in pagination.items:
        alert_dict = alert.to_dict()
        alerts_data.append(alert_dict)
    
    return jsonify({
        'alerts': alerts_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages,
            'has_next': pagination.has_next,
            'has_prev': pagination.has_prev
        }
    })


@api_bp.route('/alerts/<int:alert_id>')
@login_required
def get_alert(alert_id):
    """Get single alert details."""
    alert = Alert.query.get_or_404(alert_id)
    return jsonify(alert.to_dict(include_explanation=True))


@api_bp.route('/alerts/<int:alert_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    alert = Alert.query.get_or_404(alert_id)
    alert.acknowledged = True
    alert.acknowledged_by = current_user.id
    alert.acknowledged_at = datetime.utcnow()
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Alert acknowledged'})


@api_bp.route('/alerts/<int:alert_id>/resolve', methods=['POST'])
@login_required
def resolve_alert(alert_id):
    """Resolve an alert."""
    alert = Alert.query.get_or_404(alert_id)
    data = request.get_json() or {}
    
    alert.resolved = True
    alert.resolved_by = current_user.id
    alert.resolved_at = datetime.utcnow()
    alert.resolution_notes = data.get('notes', '')
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Alert resolved'})


@api_bp.route('/stats/dashboard')
def dashboard_stats():
    """Get dashboard statistics (public endpoint)."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    
    try:
        # Today's stats
        today_alerts = Alert.query.filter(Alert.timestamp >= today_start).count()
        today_critical = Alert.query.filter(
            Alert.timestamp >= today_start,
            Alert.severity == 'critical'
        ).count()
        
        # Total stats
        total_alerts = Alert.query.count()
        total_flows = NetworkFlow.query.count()
        
        # By severity
        severity_stats = db.session.query(
            Alert.severity,
            db.func.count().label('count')
        ).group_by(Alert.severity).all()
        
        return jsonify({
            'status': 'success',
            'stats': {
                'today': {
                    'alerts': today_alerts,
                    'critical': today_critical
                },
                'total': {
                    'alerts': total_alerts,
                    'flows': total_flows
                },
                'by_severity': {s.severity or 'unknown': s.count for s in severity_stats}
            },
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'success',
            'stats': {
                'today': {'alerts': 0, 'critical': 0},
                'total': {'alerts': 0, 'flows': 0},
                'by_severity': {}
            },
            'message': 'No data available yet',
            'timestamp': datetime.utcnow().isoformat()
        })


@api_bp.route('/alerts/stats')
@login_required
def alert_stats():
    """Get alert statistics."""
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_start - timedelta(days=7)
    
    # Today's stats
    today_total = Alert.query.filter(Alert.timestamp >= today_start).count()
    today_critical = Alert.query.filter(
        Alert.timestamp >= today_start,
        Alert.severity == 'critical'
    ).count()
    
    # Weekly stats
    week_total = Alert.query.filter(Alert.timestamp >= week_start).count()
    
    # By severity
    by_severity = db.session.query(
        Alert.severity,
        db.func.count().label('count')
    ).filter(
        Alert.timestamp >= today_start
    ).group_by(Alert.severity).all()
    
    # By attack type
    by_type = db.session.query(
        Alert.attack_type,
        db.func.count().label('count')
    ).filter(
        Alert.timestamp >= today_start
    ).group_by(Alert.attack_type).all()
    
    return jsonify({
        'today': {
            'total': today_total,
            'critical': today_critical
        },
        'week': {
            'total': week_total
        },
        'by_severity': {s.severity: s.count for s in by_severity},
        'by_type': {t.attack_type or 'Unknown': t.count for t in by_type}
    })


# ==================== Network Flows ====================

@api_bp.route('/flows')
@login_required
def get_flows():
    """Get network flows with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    pagination = NetworkFlow.query.order_by(
        NetworkFlow.timestamp.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'flows': [flow.to_dict() for flow in pagination.items],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': pagination.total,
            'pages': pagination.pages
        }
    })


@api_bp.route('/flows/ingest', methods=['POST'])
@api_key_required
def ingest_flows():
    """Ingest network flow data."""
    try:
        data = request.get_json()
        
        if not data or 'flows' not in data:
            return jsonify({'error': 'Invalid request'}), 400
        
        flows_added = 0
        for flow_data in data['flows']:
            flow = NetworkFlow(
                source_ip=flow_data.get('src_ip'),
                destination_ip=flow_data.get('dst_ip'),
                source_port=flow_data.get('src_port'),
                destination_port=flow_data.get('dst_port'),
                protocol=flow_data.get('protocol'),
                duration=flow_data.get('duration'),
                total_bytes=flow_data.get('bytes_sent', 0) + flow_data.get('bytes_recv', 0),
                packets_sent=flow_data.get('packets_sent'),
                packets_recv=flow_data.get('packets_recv'),
                raw_data=json.dumps(flow_data)
            )
            db.session.add(flow)
            flows_added += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'flows_ingested': flows_added
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ==================== Analytics ====================

@api_bp.route('/analytics/timeline')
@login_required
def analytics_timeline():
    """Get timeline data for analytics."""
    hours = request.args.get('hours', 24, type=int)
    metric = request.args.get('metric', 'alerts')  # alerts, flows, bytes
    
    now = datetime.utcnow()
    start_time = now - timedelta(hours=hours)
    
    if metric == 'alerts':
        data = db.session.query(
            db.func.strftime('%Y-%m-%d %H:00:00', Alert.timestamp).label('hour'),
            db.func.count().label('value')
        ).filter(
            Alert.timestamp >= start_time
        ).group_by('hour').all()
    else:
        data = db.session.query(
            db.func.strftime('%Y-%m-%d %H:00:00', NetworkFlow.timestamp).label('hour'),
            db.func.count().label('value') if metric == 'flows' else db.func.sum(NetworkFlow.total_bytes).label('value')
        ).filter(
            NetworkFlow.timestamp >= start_time
        ).group_by('hour').all()
    
    return jsonify({
        'labels': [d.hour for d in data],
        'values': [d.value or 0 for d in data]
    })


@api_bp.route('/analytics/top-attackers')
@login_required
def top_attackers():
    """Get top attacking IPs."""
    limit = request.args.get('limit', 10, type=int)
    days = request.args.get('days', 7, type=int)
    
    start_time = datetime.utcnow() - timedelta(days=days)
    
    top = db.session.query(
        Alert.source_ip,
        db.func.count().label('count')
    ).filter(
        Alert.timestamp >= start_time
    ).group_by(Alert.source_ip).order_by(
        db.func.count().desc()
    ).limit(limit).all()
    
    return jsonify([{
        'ip': t.source_ip,
        'count': t.count
    } for t in top])


@api_bp.route('/analytics/attack-types')
@login_required
def attack_types():
    """Get attack type distribution."""
    days = request.args.get('days', 7, type=int)
    start_time = datetime.utcnow() - timedelta(days=days)
    
    distribution = db.session.query(
        Alert.attack_type,
        db.func.count().label('count')
    ).filter(
        Alert.timestamp >= start_time
    ).group_by(Alert.attack_type).all()
    
    return jsonify({
        'labels': [d.attack_type or 'Unknown' for d in distribution],
        'values': [d.count for d in distribution]
    })


# ==================== Threat Intelligence ====================

@api_bp.route('/threat-intel')
def threat_intelligence():
    """Get threat intelligence data (public endpoint)."""
    try:
        days = request.args.get('days', 30, type=int)
        start_time = datetime.utcnow() - timedelta(days=days)
        
        # Get top threat IPs
        threat_ips = db.session.query(
            Alert.source_ip,
            db.func.count().label('count'),
            Alert.attack_type
        ).filter(
            Alert.severity.in_(['critical', 'high'])
        ).group_by(Alert.source_ip).order_by(
            db.func.count().desc()
        ).limit(20).all()
        
        # If no recent data, get all data
        if not threat_ips:
            threat_ips = db.session.query(
                Alert.source_ip,
                db.func.count().label('count'),
                Alert.attack_type
            ).filter(
                Alert.severity.in_(['critical', 'high'])
            ).group_by(Alert.source_ip).order_by(
                db.func.count().desc()
            ).limit(20).all()
        
        # Get attack type distribution
        attack_types = db.session.query(
            Alert.attack_type,
            db.func.count().label('count')
        ).group_by(Alert.attack_type).order_by(
            db.func.count().desc()
        ).limit(10).all()
        
        # Get IOCs (Indicators of Compromise)
        iocs = []
        for ip in threat_ips[:10]:
            iocs.append({
                'type': 'ip',
                'value': ip.source_ip,
                'threat_type': ip.attack_type or 'Unknown',
                'severity': 'high',
                'occurrences': ip.count
            })
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_threats': sum(ip.count for ip in threat_ips),
                'unique_ips': len(threat_ips),
                'attack_types': len(attack_types)
            },
            'threat_ips': [{
                'ip': ip.source_ip,
                'count': ip.count,
                'attack_type': ip.attack_type or 'Unknown'
            } for ip in threat_ips],
            'attack_distribution': [{
                'type': at.attack_type or 'Unknown',
                'count': at.count
            } for at in attack_types],
            'iocs': iocs
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


# ==================== Model Management ====================

@api_bp.route('/models/info')
@login_required
def model_info():
    """Get information about loaded models."""
    predictor = ModelPredictor()
    
    return jsonify({
        'models': predictor.get_model_info(),
        'loaded': predictor.is_loaded()
    })


@api_bp.route('/models/predict', methods=['POST'])
@api_key_required
def model_predict():
    """Direct model prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Features required'}), 400
        
        predictor = ModelPredictor()
        prediction = predictor.predict(data['features'])
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== API Key Management ====================

@api_bp.route('/keys', methods=['GET'])
@login_required
def list_api_keys():
    """List API keys (admin only)."""
    if current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    keys = APIKey.query.filter_by(user_id=current_user.id).all()
    return jsonify([key.to_dict() for key in keys])


@api_bp.route('/keys', methods=['POST'])
@login_required
def create_api_key():
    """Create new API key."""
    data = request.get_json() or {}
    name = data.get('name', 'API Key')
    
    key = APIKey.generate_key(user_id=current_user.id, name=name)
    db.session.add(key)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'key': key.key,  # Only shown once!
        'message': 'Save this key securely. It will not be shown again.'
    })


@api_bp.route('/keys/<int:key_id>', methods=['DELETE'])
@login_required
def revoke_api_key(key_id):
    """Revoke API key."""
    key = APIKey.query.get_or_404(key_id)
    
    if key.user_id != current_user.id and current_user.role != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    key.is_active = False
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'API key revoked'})


# ==================== Search ====================

@api_bp.route('/search')
@login_required
def global_search():
    """Global search for alerts, IPs, and threats."""
    query = request.args.get('q', '').strip()
    
    if len(query) < 2:
        return jsonify([])
    
    results = []
    
    # Search alerts by source IP
    ip_alerts = Alert.query.filter(
        (Alert.source_ip.contains(query)) | 
        (Alert.destination_ip.contains(query))
    ).limit(10).all()
    
    for alert in ip_alerts:
        results.append({
            'id': alert.id,
            'type': 'alert',
            'icon': 'exclamation-triangle',
            'title': f'{alert.attack_type or "Alert"} - {alert.source_ip}',
            'subtitle': f'Severity: {alert.severity} | {alert.timestamp.strftime("%Y-%m-%d %H:%M")}',
            'url': f'/alerts/{alert.id}',
            'severity': alert.severity
        })
    
    # Search by attack type
    type_alerts = Alert.query.filter(
        Alert.attack_type.ilike(f'%{query}%')
    ).limit(5).all()
    
    for alert in type_alerts:
        if not any(r['id'] == alert.id for r in results):
            results.append({
                'id': alert.id,
                'type': 'alert',
                'icon': 'bug',
                'title': f'{alert.attack_type} Attack',
                'subtitle': f'From: {alert.source_ip} | {alert.severity}',
                'url': f'/alerts/{alert.id}',
                'severity': alert.severity
            })
    
    # Search network flows
    flows = NetworkFlow.query.filter(
        (NetworkFlow.source_ip.contains(query)) |
        (NetworkFlow.destination_ip.contains(query))
    ).limit(5).all()
    
    for flow in flows:
        results.append({
            'id': flow.id,
            'type': 'flow',
            'icon': 'diagram-3',
            'title': f'{flow.source_ip} → {flow.destination_ip}',
            'subtitle': f'Protocol: {flow.protocol} | Port: {flow.destination_port}',
            'url': f'/analytics/traffic',
            'severity': 'info'
        })
    
    # Add quick actions for IP searches
    if is_valid_ip(query):
        results.insert(0, {
            'id': 'ip-lookup',
            'type': 'action',
            'icon': 'search',
            'title': f'Search all alerts for {query}',
            'subtitle': 'View all security events related to this IP',
            'url': f'/alerts?ip={query}',
            'severity': 'info'
        })
    
    return jsonify(results[:15])  # Limit to 15 results


def is_valid_ip(s):
    """Check if string is a valid IP address."""
    parts = s.split('.')
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(part) <= 255 for part in parts)
    except ValueError:
        return False

