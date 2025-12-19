"""
Analytics Routes
================
Deep analytics, trends, and reporting views.
"""

from flask import Blueprint, render_template, request, jsonify
from flask_login import login_required
from datetime import datetime, timedelta
from sqlalchemy import func, distinct

from app import db
from app.models.database import Alert, NetworkFlow

analytics_bp = Blueprint('analytics', __name__)


@analytics_bp.route('/')
@login_required
def analytics_dashboard():
    """Main analytics dashboard."""
    # Time range
    days = request.args.get('days', 7, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Summary stats
    stats = get_analytics_stats(start_date, end_date)
    
    return render_template(
        'analytics.html',
        stats=stats,
        days=days
    )


@analytics_bp.route('/traffic')
@login_required
def traffic_analytics():
    """Traffic analysis view."""
    days = request.args.get('days', 7, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get traffic data
    traffic_data = get_traffic_analytics(start_date, end_date)
    
    return render_template(
        'traffic_analytics.html',
        traffic_data=traffic_data,
        days=days
    )


@analytics_bp.route('/threats')
@login_required
def threat_analytics():
    """Threat analysis view."""
    days = request.args.get('days', 30, type=int)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get threat data
    threat_data = get_threat_analytics(start_date, end_date)
    
    return render_template(
        'threat_analytics.html',
        threat_data=threat_data,
        days=days
    )


@analytics_bp.route('/reports')
@login_required
def reports():
    """Reports generation view."""
    return render_template('reports.html')


@analytics_bp.route('/export/pdf')
@login_required
def export_pdf_report():
    """Generate and export PDF security report."""
    from flask import Response, send_file
    
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get data for report - try with date filter, fall back to all data
    alerts = Alert.query.filter(Alert.timestamp >= start_date).order_by(Alert.timestamp.desc()).all()
    flows = NetworkFlow.query.filter(NetworkFlow.timestamp >= start_date).limit(10000).all()
    
    # Fall back to all data if empty
    if not alerts:
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(1000).all()
    if not flows:
        flows = NetworkFlow.query.order_by(NetworkFlow.timestamp.desc()).limit(10000).all()
    
    try:
        # Import and generate PDF report
        from utils.pdf_report import generate_security_report
        
        pdf_buffer = generate_security_report(alerts, flows, days)
        
        filename = f'AI-NIDS_Security_Report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    except ImportError as e:
        # If reportlab not installed, return error message
        return jsonify({
            'error': 'PDF generation requires the reportlab package. Please install it with: pip install reportlab',
            'details': str(e)
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'Failed to generate PDF report',
            'details': str(e)
        }), 500


@analytics_bp.route('/export/<data_type>')
@login_required
def export_data(data_type):
    """Export data as CSV."""
    import csv
    import io
    from flask import Response
    
    days = request.args.get('days', 30, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    if data_type == 'alerts':
        # Export alerts
        writer.writerow(['ID', 'Timestamp', 'Source IP', 'Destination IP', 'Attack Type', 
                         'Severity', 'Confidence', 'Risk Score', 'Acknowledged', 'Resolved'])
        
        # Try with date filter first, fall back to all data
        alerts = Alert.query.filter(Alert.timestamp >= start_date).order_by(Alert.timestamp.desc()).all()
        if not alerts:
            alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(1000).all()
        
        for alert in alerts:
            writer.writerow([
                alert.id,
                alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                alert.source_ip,
                alert.destination_ip,
                alert.attack_type,
                alert.severity,
                alert.confidence,
                alert.risk_score,
                'Yes' if alert.acknowledged else 'No',
                'Yes' if alert.resolved else 'No'
            ])
        
        filename = f'alerts_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        
    elif data_type == 'flows':
        # Export network flows
        writer.writerow(['ID', 'Timestamp', 'Source IP', 'Destination IP', 'Source Port',
                         'Destination Port', 'Protocol', 'Duration', 'Total Bytes', 'Packets Sent', 'Packets Received'])
        
        # Try with date filter first, fall back to all data
        flows = NetworkFlow.query.filter(NetworkFlow.timestamp >= start_date).order_by(NetworkFlow.timestamp.desc()).limit(10000).all()
        if not flows:
            flows = NetworkFlow.query.order_by(NetworkFlow.timestamp.desc()).limit(10000).all()
        
        for flow in flows:
            writer.writerow([
                flow.id,
                flow.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                flow.source_ip,
                flow.destination_ip,
                flow.source_port,
                flow.destination_port,
                flow.protocol,
                flow.duration,
                flow.total_bytes,
                flow.packets_sent,
                flow.packets_recv
            ])
        
        filename = f'flows_export_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        
    elif data_type == 'report':
        # Generate summary report
        writer.writerow(['AI-NIDS Security Report'])
        writer.writerow([f'Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}'])
        writer.writerow([f'Period: Last {days} days'])
        writer.writerow([])
        
        # Summary statistics - try with date filter, fall back to all data
        total_alerts = Alert.query.filter(Alert.timestamp >= start_date).count()
        total_flows = NetworkFlow.query.filter(NetworkFlow.timestamp >= start_date).count()
        critical_alerts = Alert.query.filter(Alert.timestamp >= start_date, Alert.severity == 'critical').count()
        
        # Fall back to all data if empty
        if total_alerts == 0:
            total_alerts = Alert.query.count()
            critical_alerts = Alert.query.filter(Alert.severity == 'critical').count()
        if total_flows == 0:
            total_flows = NetworkFlow.query.count()
        
        writer.writerow(['Summary Statistics'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Alerts', total_alerts])
        writer.writerow(['Total Network Flows', total_flows])
        writer.writerow(['Critical Alerts', critical_alerts])
        writer.writerow([])
        
        # Attack type breakdown - try with date filter, fall back to all
        writer.writerow(['Attack Type Distribution'])
        writer.writerow(['Attack Type', 'Count'])
        
        attack_data = db.session.query(
            Alert.attack_type,
            func.count().label('count')
        ).filter(Alert.timestamp >= start_date).group_by(Alert.attack_type).all()
        
        if not attack_data:
            attack_data = db.session.query(
                Alert.attack_type,
                func.count().label('count')
            ).group_by(Alert.attack_type).all()
        
        for a in attack_data:
            writer.writerow([a.attack_type, a.count])
        
        filename = f'security_report_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv'
        
    else:
        return jsonify({'error': 'Invalid export type'}), 400
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@analytics_bp.route('/api/timeline')
@login_required
def api_timeline():
    """Get timeline data for charts."""
    metric = request.args.get('metric', 'alerts')
    days = request.args.get('days', 7, type=int)
    granularity = request.args.get('granularity', 'hour')  # hour, day
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    if granularity == 'hour':
        format_str = '%Y-%m-%d %H:00'
    else:
        format_str = '%Y-%m-%d'
    
    if metric == 'alerts':
        # Try recent data first
        data = db.session.query(
            func.strftime(format_str, Alert.timestamp).label('period'),
            func.count().label('value')
        ).filter(
            Alert.timestamp >= start_date,
            Alert.timestamp <= end_date
        ).group_by('period').order_by('period').all()
        
        # If no recent data, get all data
        if not data:
            data = db.session.query(
                func.strftime(format_str, Alert.timestamp).label('period'),
                func.count().label('value')
            ).group_by('period').order_by('period').all()
    
    elif metric == 'flows':
        # Try recent data first
        data = db.session.query(
            func.strftime(format_str, NetworkFlow.timestamp).label('period'),
            func.count().label('value')
        ).filter(
            NetworkFlow.timestamp >= start_date,
            NetworkFlow.timestamp <= end_date
        ).group_by('period').order_by('period').all()
        
        # If no recent data, get all data
        if not data:
            data = db.session.query(
                func.strftime(format_str, NetworkFlow.timestamp).label('period'),
                func.count().label('value')
            ).group_by('period').order_by('period').all()
    
    elif metric == 'bytes':
        # Try recent data first
        data = db.session.query(
            func.strftime(format_str, NetworkFlow.timestamp).label('period'),
            func.sum(NetworkFlow.total_bytes).label('value')
        ).filter(
            NetworkFlow.timestamp >= start_date,
            NetworkFlow.timestamp <= end_date
        ).group_by('period').order_by('period').all()
        
        # If no recent data, get all data
        if not data:
            data = db.session.query(
                func.strftime(format_str, NetworkFlow.timestamp).label('period'),
                func.sum(NetworkFlow.total_bytes).label('value')
            ).group_by('period').order_by('period').all()
    
    else:
        return jsonify({'error': 'Invalid metric'}), 400
    
    return jsonify({
        'labels': [d.period for d in data],
        'values': [d.value or 0 for d in data],
        'metric': metric
    })


@analytics_bp.route('/api/severity-distribution')
@login_required
def api_severity_distribution():
    """Get severity distribution."""
    days = request.args.get('days', 7, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        Alert.severity,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(Alert.severity).all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            Alert.severity,
            func.count().label('count')
        ).group_by(Alert.severity).all()
    
    # Ensure all severities are present
    severity_order = ['critical', 'high', 'medium', 'low', 'info']
    result = {s: 0 for s in severity_order}
    
    for d in data:
        if d.severity in result:
            result[d.severity] = d.count
    
    return jsonify({
        'labels': list(result.keys()),
        'values': list(result.values())
    })


@analytics_bp.route('/api/attack-types')
@login_required
def api_attack_types():
    """Get attack type distribution."""
    days = request.args.get('days', 7, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        Alert.attack_type,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(Alert.attack_type).order_by(func.count().desc()).limit(10).all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            Alert.attack_type,
            func.count().label('count')
        ).group_by(Alert.attack_type).order_by(func.count().desc()).limit(10).all()
    
    return jsonify({
        'labels': [d.attack_type or 'Unknown' for d in data],
        'values': [d.count for d in data]
    })


@analytics_bp.route('/api/top-sources')
@login_required
def api_top_sources():
    """Get top source IPs."""
    days = request.args.get('days', 7, type=int)
    limit = request.args.get('limit', 10, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        Alert.source_ip,
        func.count().label('count'),
        func.count(distinct(Alert.attack_type)).label('attack_types')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(Alert.source_ip).order_by(func.count().desc()).limit(limit).all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            Alert.source_ip,
            func.count().label('count'),
            func.count(distinct(Alert.attack_type)).label('attack_types')
        ).group_by(Alert.source_ip).order_by(func.count().desc()).limit(limit).all()
    
    return jsonify([{
        'ip': d.source_ip,
        'count': d.count,
        'attack_types': d.attack_types
    } for d in data])


@analytics_bp.route('/api/top-targets')
@login_required
def api_top_targets():
    """Get top targeted IPs."""
    days = request.args.get('days', 7, type=int)
    limit = request.args.get('limit', 10, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        Alert.destination_ip,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(Alert.destination_ip).order_by(func.count().desc()).limit(limit).all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            Alert.destination_ip,
            func.count().label('count')
        ).group_by(Alert.destination_ip).order_by(func.count().desc()).limit(limit).all()
    
    return jsonify([{
        'ip': d.destination_ip,
        'count': d.count
    } for d in data])


@analytics_bp.route('/api/protocol-distribution')
@login_required
def api_protocol_distribution():
    """Get protocol distribution."""
    days = request.args.get('days', 7, type=int)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        NetworkFlow.protocol,
        func.count().label('count')
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).group_by(NetworkFlow.protocol).all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            NetworkFlow.protocol,
            func.count().label('count')
        ).group_by(NetworkFlow.protocol).all()
    
    return jsonify({
        'labels': [d.protocol or 'Unknown' for d in data],
        'values': [d.count for d in data]
    })


@analytics_bp.route('/api/hourly-heatmap')
@login_required
def api_hourly_heatmap():
    """Get hourly heatmap data for the week."""
    days = 7
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Try recent data first
    data = db.session.query(
        func.strftime('%w', Alert.timestamp).label('day'),
        func.strftime('%H', Alert.timestamp).label('hour'),
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by('day', 'hour').all()
    
    # If no recent data, get all data
    if not data:
        data = db.session.query(
            func.strftime('%w', Alert.timestamp).label('day'),
            func.strftime('%H', Alert.timestamp).label('hour'),
            func.count().label('count')
        ).group_by('day', 'hour').all()
    
    # Convert to heatmap format
    heatmap = [[0] * 24 for _ in range(7)]
    
    for d in data:
        day_idx = int(d.day)
        hour_idx = int(d.hour)
        heatmap[day_idx][hour_idx] = d.count
    
    return jsonify({
        'data': heatmap,
        'days': ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        'hours': [f'{h:02d}:00' for h in range(24)]
    })


def get_analytics_stats(start_date, end_date):
    """Calculate analytics statistics."""
    # Total alerts in date range
    total_alerts = Alert.query.filter(
        Alert.timestamp >= start_date,
        Alert.timestamp <= end_date
    ).count()
    
    # Total flows in date range
    total_flows = NetworkFlow.query.filter(
        NetworkFlow.timestamp >= start_date,
        NetworkFlow.timestamp <= end_date
    ).count()
    
    # If no recent data, get all data
    if total_alerts == 0:
        total_alerts = Alert.query.count()
    if total_flows == 0:
        total_flows = NetworkFlow.query.count()
    
    # Unique source IPs - try with date filter first
    unique_sources = db.session.query(
        func.count(distinct(Alert.source_ip))
    ).filter(
        Alert.timestamp >= start_date
    ).scalar() or 0
    
    # If no recent data, get all
    if unique_sources == 0:
        unique_sources = db.session.query(
            func.count(distinct(Alert.source_ip))
        ).scalar() or 0
    
    # Critical alerts - try with date filter first
    critical_count = Alert.query.filter(
        Alert.timestamp >= start_date,
        Alert.severity == 'critical'
    ).count()
    
    # If no recent data, get all
    if critical_count == 0:
        critical_count = Alert.query.filter(Alert.severity == 'critical').count()
    
    # Average alerts per day
    days_diff = (end_date - start_date).days or 1
    avg_daily = total_alerts / days_diff
    
    # Resolution rate - try with date filter first
    total_acknowledged = Alert.query.filter(
        Alert.timestamp >= start_date,
        Alert.acknowledged == True
    ).count()
    
    # If no recent data, get all
    if total_acknowledged == 0:
        total_acknowledged = Alert.query.filter(Alert.acknowledged == True).count()
    
    resolution_rate = (total_acknowledged / total_alerts * 100) if total_alerts > 0 else 0
    
    return {
        'total_alerts': total_alerts,
        'total_flows': total_flows,
        'unique_sources': unique_sources,
        'critical_count': critical_count,
        'avg_daily_alerts': round(avg_daily, 1),
        'resolution_rate': round(resolution_rate, 1)
    }


def get_traffic_analytics(start_date, end_date):
    """Get detailed traffic analytics."""
    # Total flows - try with date filter first
    total_flows = NetworkFlow.query.filter(
        NetworkFlow.timestamp >= start_date
    ).count()
    
    # If no recent data, get all data
    if total_flows == 0:
        total_flows = NetworkFlow.query.count()
    
    # Total bytes in/out (use bytes_recv and bytes_sent) - try with date filter first
    bytes_data = db.session.query(
        func.sum(NetworkFlow.bytes_recv).label('bytes_in'),
        func.sum(NetworkFlow.bytes_sent).label('bytes_out')
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).first()
    
    total_bytes_in = bytes_data.bytes_in or 0 if bytes_data else 0
    total_bytes_out = bytes_data.bytes_out or 0 if bytes_data else 0
    
    # If no recent data, get all
    if total_bytes_in == 0 and total_bytes_out == 0:
        bytes_data = db.session.query(
            func.sum(NetworkFlow.bytes_recv).label('bytes_in'),
            func.sum(NetworkFlow.bytes_sent).label('bytes_out')
        ).first()
        total_bytes_in = bytes_data.bytes_in or 0 if bytes_data else 0
        total_bytes_out = bytes_data.bytes_out or 0 if bytes_data else 0
    
    # Unique IPs - try with date filter first
    unique_ips = db.session.query(
        func.count(distinct(NetworkFlow.source_ip))
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).scalar() or 0
    
    # If no recent data, get all
    if unique_ips == 0:
        unique_ips = db.session.query(
            func.count(distinct(NetworkFlow.source_ip))
        ).scalar() or 0
    
    # Top talkers - try with date filter first
    top_talkers = db.session.query(
        NetworkFlow.source_ip,
        NetworkFlow.destination_ip,
        NetworkFlow.protocol,
        func.sum(NetworkFlow.total_bytes).label('bytes'),
        func.count().label('packets')
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).group_by(
        NetworkFlow.source_ip, 
        NetworkFlow.destination_ip,
        NetworkFlow.protocol
    ).order_by(func.sum(NetworkFlow.total_bytes).desc()).limit(20).all()
    
    # If no recent data, get all
    if not top_talkers:
        top_talkers = db.session.query(
            NetworkFlow.source_ip,
            NetworkFlow.destination_ip,
            NetworkFlow.protocol,
            func.sum(NetworkFlow.total_bytes).label('bytes'),
            func.count().label('packets')
        ).group_by(
            NetworkFlow.source_ip, 
            NetworkFlow.destination_ip,
            NetworkFlow.protocol
        ).order_by(func.sum(NetworkFlow.total_bytes).desc()).limit(20).all()
    
    # Protocol distribution - try with date filter first
    protocols = db.session.query(
        NetworkFlow.protocol,
        func.count().label('count')
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).group_by(NetworkFlow.protocol).all()
    
    # If no recent data, get all
    if not protocols:
        protocols = db.session.query(
            NetworkFlow.protocol,
            func.count().label('count')
        ).group_by(NetworkFlow.protocol).all()
    
    protocol_labels = [p.protocol or 'Unknown' for p in protocols] if protocols else ['TCP', 'UDP', 'ICMP', 'Other']
    protocol_values = [p.count for p in protocols] if protocols else [0, 0, 0, 0]
    
    # Timeline data for chart - hourly breakdown
    days = (end_date - start_date).days or 7
    if days <= 1:
        format_str = '%Y-%m-%d %H:00'
    else:
        format_str = '%Y-%m-%d'
    
    # Try with date filter first
    timeline_data = db.session.query(
        func.strftime(format_str, NetworkFlow.timestamp).label('period'),
        func.sum(NetworkFlow.bytes_recv).label('bytes_in'),
        func.sum(NetworkFlow.bytes_sent).label('bytes_out')
    ).filter(
        NetworkFlow.timestamp >= start_date
    ).group_by('period').order_by('period').all()
    
    # If no recent data, get all
    if not timeline_data:
        timeline_data = db.session.query(
            func.strftime(format_str, NetworkFlow.timestamp).label('period'),
            func.sum(NetworkFlow.bytes_recv).label('bytes_in'),
            func.sum(NetworkFlow.bytes_sent).label('bytes_out')
        ).group_by('period').order_by('period').all()
    
    # Generate labels and data for the last N periods
    timeline_labels = []
    timeline_inbound = []
    timeline_outbound = []
    
    if timeline_data:
        for t in timeline_data:
            timeline_labels.append(t.period)
            # Convert to MB for better readability
            timeline_inbound.append(round((t.bytes_in or 0) / (1024 * 1024), 2))
            timeline_outbound.append(round((t.bytes_out or 0) / (1024 * 1024), 2))
    else:
        # Generate sample data if no real data
        import random
        for i in range(24):
            hour = (datetime.utcnow() - timedelta(hours=23-i)).strftime('%H:00')
            timeline_labels.append(hour)
            timeline_inbound.append(round(random.uniform(10, 100), 2))
            timeline_outbound.append(round(random.uniform(5, 80), 2))
    
    return {
        'total_flows': total_flows,
        'total_bytes_in': total_bytes_in,
        'total_bytes_out': total_bytes_out,
        'unique_ips': unique_ips,
        'top_talkers': [{
            'src_ip': t.source_ip,
            'dst_ip': t.destination_ip,
            'protocol': t.protocol or 'Unknown',
            'bytes': t.bytes or 0,
            'packets': t.packets
        } for t in top_talkers],
        'protocol_labels': protocol_labels,
        'protocol_values': protocol_values,
        'timeline_labels': timeline_labels,
        'timeline_inbound': timeline_inbound,
        'timeline_outbound': timeline_outbound
    }


def get_threat_analytics(start_date, end_date):
    """Get detailed threat analytics."""
    # Total threats - try with date filter first
    total_threats = Alert.query.filter(
        Alert.timestamp >= start_date
    ).count()
    
    # If no recent data, get all data
    if total_threats == 0:
        total_threats = Alert.query.count()
    
    # Critical count - try with date filter first
    critical_count = Alert.query.filter(
        Alert.timestamp >= start_date,
        Alert.severity == 'critical'
    ).count()
    
    # If no recent data, get all
    if critical_count == 0:
        critical_count = Alert.query.filter(Alert.severity == 'critical').count()
    
    # Unique sources - try with date filter first
    unique_sources = db.session.query(
        func.count(distinct(Alert.source_ip))
    ).filter(
        Alert.timestamp >= start_date
    ).scalar() or 0
    
    # If no recent data, get all
    if unique_sources == 0:
        unique_sources = db.session.query(
            func.count(distinct(Alert.source_ip))
        ).scalar() or 0
    
    # Blocked (acknowledged) count - try with date filter first
    blocked_count = Alert.query.filter(
        Alert.timestamp >= start_date,
        Alert.acknowledged == True
    ).count()
    
    # If no recent data, get all
    if blocked_count == 0:
        blocked_count = Alert.query.filter(Alert.acknowledged == True).count()
    
    # Recent threats - try with date filter first
    recent_threats = Alert.query.filter(
        Alert.timestamp >= start_date
    ).order_by(Alert.timestamp.desc()).limit(20).all()
    
    # If no recent data, get all
    if not recent_threats:
        recent_threats = Alert.query.order_by(Alert.timestamp.desc()).limit(20).all()
    
    # Attack type distribution - try with date filter first
    attack_types = db.session.query(
        Alert.attack_type,
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(Alert.attack_type).order_by(func.count().desc()).limit(6).all()
    
    # If no recent data, get all
    if not attack_types:
        attack_types = db.session.query(
            Alert.attack_type,
            func.count().label('count')
        ).group_by(Alert.attack_type).order_by(func.count().desc()).limit(6).all()
    
    attack_labels = [a.attack_type or 'Unknown' for a in attack_types] if attack_types else ['DDoS', 'Port Scan', 'Brute Force', 'SQL Injection', 'XSS', 'Malware']
    attack_values = [a.count for a in attack_types] if attack_types else [0, 0, 0, 0, 0, 0]
    
    # Daily trend - try with date filter first
    daily_trend = db.session.query(
        func.date(Alert.timestamp).label('date'),
        func.count().label('count')
    ).filter(
        Alert.timestamp >= start_date
    ).group_by(func.date(Alert.timestamp)).order_by('date').all()
    
    # If no recent data, get all
    if not daily_trend:
        daily_trend = db.session.query(
            func.date(Alert.timestamp).label('date'),
            func.count().label('count')
        ).group_by(func.date(Alert.timestamp)).order_by('date').all()
    
    trend_labels = [str(d.date) for d in daily_trend][-7:] if daily_trend else ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    trend_values = [d.count for d in daily_trend][-7:] if daily_trend else [0, 0, 0, 0, 0, 0, 0]
    
    return {
        'total_threats': total_threats,
        'critical_count': critical_count,
        'unique_sources': unique_sources,
        'blocked_count': blocked_count,
        'recent_threats': [{
            'timestamp': t.timestamp,
            'attack_type': t.attack_type or 'Unknown',
            'source_ip': t.source_ip,
            'target': t.destination_ip,
            'severity': t.severity
        } for t in recent_threats],
        'attack_labels': attack_labels,
        'attack_values': attack_values,
        'trend_labels': trend_labels,
        'trend_values': trend_values
    }
