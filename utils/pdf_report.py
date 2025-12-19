"""
AI-NIDS PDF Report Generator
============================
Generates professional security reports in PDF format with charts and analysis.
"""

import io
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable, ListFlowable, ListItem
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics import renderPDF

# For chart generation
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class SecurityReportGenerator:
    """Generates comprehensive PDF security reports."""
    
    # Color scheme
    COLORS = {
        'primary': colors.HexColor('#6366f1'),
        'secondary': colors.HexColor('#8b5cf6'),
        'success': colors.HexColor('#10b981'),
        'warning': colors.HexColor('#f59e0b'),
        'danger': colors.HexColor('#ef4444'),
        'info': colors.HexColor('#06b6d4'),
        'dark': colors.HexColor('#1e293b'),
        'light': colors.HexColor('#f1f5f9'),
        'text': colors.HexColor('#334155'),
        'muted': colors.HexColor('#64748b'),
    }
    
    SEVERITY_COLORS = {
        'critical': colors.HexColor('#dc2626'),
        'high': colors.HexColor('#ea580c'),
        'medium': colors.HexColor('#d97706'),
        'low': colors.HexColor('#65a30d'),
        'info': colors.HexColor('#0891b2'),
    }
    
    def __init__(self, title: str = "AI-NIDS Security Report"):
        self.title = title
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.elements = []
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=28,
            textColor=self.COLORS['primary'],
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=self.COLORS['muted'],
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=self.COLORS['dark'],
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            borderPadding=(0, 0, 5, 0),
        ))
        
        # Subsection
        self.styles.add(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=self.COLORS['primary'],
            spaceBefore=15,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        # Body text - override existing BodyText style
        self.styles['BodyText'].fontSize = 10
        self.styles['BodyText'].textColor = self.COLORS['text']
        self.styles['BodyText'].spaceAfter = 8
        self.styles['BodyText'].alignment = TA_JUSTIFY
        self.styles['BodyText'].leading = 14
        
        # Stat value
        self.styles.add(ParagraphStyle(
            name='StatValue',
            parent=self.styles['Normal'],
            fontSize=24,
            textColor=self.COLORS['primary'],
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Stat label
        self.styles.add(ParagraphStyle(
            name='StatLabel',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=self.COLORS['muted'],
            alignment=TA_CENTER
        ))
        
        # Footer
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=self.COLORS['muted'],
            alignment=TA_CENTER
        ))
        
    def _add_header_section(self, report_data: Dict):
        """Add report header with logo and title."""
        # Title
        self.elements.append(Paragraph(
            "🛡️ AI-NIDS Security Report",
            self.styles['ReportTitle']
        ))
        
        # Subtitle with date range
        days = report_data.get('days', 30)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        self.elements.append(Paragraph(
            f"Analysis Period: {start_date.strftime('%B %d, %Y')} - {end_date.strftime('%B %d, %Y')}",
            self.styles['ReportSubtitle']
        ))
        
        self.elements.append(Paragraph(
            f"Generated: {end_date.strftime('%B %d, %Y at %H:%M UTC')}",
            self.styles['ReportSubtitle']
        ))
        
        # Horizontal line
        self.elements.append(HRFlowable(
            width="100%",
            thickness=2,
            color=self.COLORS['primary'],
            spaceBefore=10,
            spaceAfter=20
        ))
        
    def _add_executive_summary(self, report_data: Dict):
        """Add executive summary section."""
        self.elements.append(Paragraph(
            "📊 Executive Summary",
            self.styles['SectionHeader']
        ))
        
        summary = report_data.get('summary', {})
        
        # Summary text
        total_alerts = summary.get('total_alerts', 0)
        critical = summary.get('critical_alerts', 0)
        high = summary.get('high_alerts', 0)
        total_flows = summary.get('total_flows', 0)
        detection_rate = summary.get('detection_rate', 0)
        
        summary_text = f"""
        During the analysis period, the AI-NIDS system processed <b>{total_flows:,}</b> network flows 
        and detected <b>{total_alerts:,}</b> security alerts. Of these, <b>{critical}</b> were classified 
        as critical severity and <b>{high}</b> as high severity, requiring immediate attention.
        <br/><br/>
        The system maintained a detection rate of <b>{detection_rate:.1f}%</b>, demonstrating effective 
        threat identification capabilities. This report provides detailed analysis of attack patterns, 
        threat sources, and recommended security measures.
        """
        
        self.elements.append(Paragraph(summary_text, self.styles['BodyText']))
        self.elements.append(Spacer(1, 15))
        
        # Key metrics table
        self._add_key_metrics_table(summary)
        
    def _add_key_metrics_table(self, summary: Dict):
        """Add key metrics as a styled table."""
        metrics = [
            ('Total Alerts', f"{summary.get('total_alerts', 0):,}", self.COLORS['primary']),
            ('Critical', f"{summary.get('critical_alerts', 0):,}", self.COLORS['danger']),
            ('High', f"{summary.get('high_alerts', 0):,}", self.COLORS['warning']),
            ('Medium', f"{summary.get('medium_alerts', 0):,}", colors.HexColor('#f59e0b')),
            ('Low', f"{summary.get('low_alerts', 0):,}", self.COLORS['success']),
            ('Resolved', f"{summary.get('resolved_alerts', 0):,}", self.COLORS['info']),
        ]
        
        # Create table data
        table_data = [[
            Paragraph(f"<b>{m[0]}</b>", self.styles['StatLabel']),
            Paragraph(f"<font color='{m[2].hexval()}'><b>{m[1]}</b></font>", 
                     ParagraphStyle('metric', fontSize=16, alignment=TA_CENTER, fontName='Helvetica-Bold'))
        ] for m in metrics]
        
        # Transpose to horizontal layout
        headers = [[Paragraph(f"<b>{m[0]}</b>", self.styles['StatLabel']) for m in metrics]]
        values = [[Paragraph(m[1], ParagraphStyle('val', fontSize=18, alignment=TA_CENTER, fontName='Helvetica-Bold', textColor=m[2])) for m in metrics]]
        
        table = Table(headers + values, colWidths=[85] * 6)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['light']),
            ('BACKGROUND', (0, 1), (-1, 1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, self.COLORS['light']),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
        
    def _add_threat_analysis(self, report_data: Dict):
        """Add threat analysis section with charts."""
        self.elements.append(Paragraph(
            "🔍 Threat Analysis",
            self.styles['SectionHeader']
        ))
        
        attack_types = report_data.get('attack_types', {})
        
        if attack_types:
            self.elements.append(Paragraph(
                "Attack Type Distribution",
                self.styles['SubSection']
            ))
            
            # Create attack types bar chart
            chart_image = self._create_attack_chart(attack_types)
            if chart_image:
                self.elements.append(Image(chart_image, width=450, height=250))
            
            self.elements.append(Spacer(1, 10))
            
            # Attack types table
            self._add_attack_types_table(attack_types)
            
    def _create_attack_chart(self, attack_types: Dict) -> Optional[io.BytesIO]:
        """Create attack types bar chart as image."""
        try:
            plt.figure(figsize=(8, 4.5))
            
            # Sort by count
            sorted_attacks = sorted(attack_types.items(), key=lambda x: x[1], reverse=True)[:10]
            labels = [a[0].replace('_', ' ').title() for a in sorted_attacks]
            values = [a[1] for a in sorted_attacks]
            
            # Color gradient
            cmap = plt.cm.get_cmap('RdYlGn_r')
            colors_list = [cmap(i / len(values)) for i in range(len(values))]
            
            bars = plt.barh(labels[::-1], values[::-1], color=colors_list[::-1], edgecolor='white', linewidth=0.5)
            
            plt.xlabel('Number of Incidents', fontsize=10, color='#334155')
            plt.title('Top Attack Types Detected', fontsize=12, fontweight='bold', color='#1e293b', pad=15)
            
            # Add value labels
            for bar, val in zip(bars, values[::-1]):
                plt.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{val:,}', va='center', fontsize=9, color='#64748b')
            
            plt.tight_layout()
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_color('#e2e8f0')
            plt.gca().spines['bottom'].set_color('#e2e8f0')
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close()
            
            return buf
        except Exception as e:
            print(f"Error creating attack chart: {e}")
            return None
            
    def _add_attack_types_table(self, attack_types: Dict):
        """Add attack types breakdown table."""
        sorted_attacks = sorted(attack_types.items(), key=lambda x: x[1], reverse=True)[:10]
        total = sum(attack_types.values())
        
        table_data = [['Attack Type', 'Count', 'Percentage', 'Severity']]
        
        severity_map = {
            'dos': 'High', 'ddos': 'Critical', 'portscan': 'Medium',
            'probe': 'Medium', 'bruteforce': 'High', 'sql_injection': 'Critical',
            'xss': 'High', 'malware': 'Critical', 'backdoor': 'Critical',
            'rootkit': 'Critical', 'trojan': 'Critical', 'worm': 'High',
            'botnet': 'Critical', 'exploit': 'High', 'reconnaissance': 'Low',
            'normal': 'Info'
        }
        
        for attack, count in sorted_attacks:
            pct = (count / total * 100) if total > 0 else 0
            severity = severity_map.get(attack.lower(), 'Medium')
            table_data.append([
                attack.replace('_', ' ').title(),
                f"{count:,}",
                f"{pct:.1f}%",
                severity
            ])
        
        table = Table(table_data, colWidths=[180, 80, 80, 80])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light']]),
            ('BOX', (0, 0), (-1, -1), 1, self.COLORS['light']),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        self.elements.append(table)
        self.elements.append(Spacer(1, 20))
        
    def _add_severity_breakdown(self, report_data: Dict):
        """Add severity breakdown section."""
        self.elements.append(Paragraph(
            "⚠️ Severity Breakdown",
            self.styles['SectionHeader']
        ))
        
        severity_data = report_data.get('severity_breakdown', {})
        
        if severity_data:
            chart_image = self._create_severity_pie_chart(severity_data)
            if chart_image:
                self.elements.append(Image(chart_image, width=400, height=280))
                
        self.elements.append(Spacer(1, 20))
        
    def _create_severity_pie_chart(self, severity_data: Dict) -> Optional[io.BytesIO]:
        """Create severity pie chart."""
        try:
            plt.figure(figsize=(7, 5))
            
            labels = []
            sizes = []
            chart_colors = []
            
            color_map = {
                'critical': '#dc2626',
                'high': '#ea580c',
                'medium': '#d97706',
                'low': '#65a30d',
                'info': '#0891b2'
            }
            
            for severity, count in severity_data.items():
                if count > 0:
                    labels.append(f"{severity.title()} ({count:,})")
                    sizes.append(count)
                    chart_colors.append(color_map.get(severity.lower(), '#94a3b8'))
            
            if sizes:
                explode = [0.05 if i == 0 else 0 for i in range(len(sizes))]
                
                wedges, texts, autotexts = plt.pie(
                    sizes, explode=explode, labels=labels, colors=chart_colors,
                    autopct='%1.1f%%', shadow=False, startangle=90,
                    textprops={'fontsize': 9}
                )
                
                plt.setp(autotexts, size=9, weight='bold', color='white')
                plt.title('Alert Severity Distribution', fontsize=12, fontweight='bold', 
                         color='#1e293b', pad=15)
                
                plt.axis('equal')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                buf.seek(0)
                plt.close()
                
                return buf
        except Exception as e:
            print(f"Error creating severity chart: {e}")
            return None
            
    def _add_timeline_analysis(self, report_data: Dict):
        """Add threat timeline section."""
        self.elements.append(Paragraph(
            "📈 Threat Timeline",
            self.styles['SectionHeader']
        ))
        
        timeline = report_data.get('timeline', {})
        
        if timeline:
            chart_image = self._create_timeline_chart(timeline)
            if chart_image:
                self.elements.append(Image(chart_image, width=500, height=250))
                
        self.elements.append(Spacer(1, 10))
        
        # Add analysis text
        if timeline:
            peak_day = max(timeline.items(), key=lambda x: x[1]) if timeline else (None, 0)
            avg_daily = sum(timeline.values()) / len(timeline) if timeline else 0
            
            analysis_text = f"""
            The timeline analysis shows threat activity patterns over the reporting period.
            Peak activity was observed on <b>{peak_day[0]}</b> with <b>{peak_day[1]:,}</b> alerts detected.
            The average daily alert volume was <b>{avg_daily:.0f}</b> alerts.
            """
            self.elements.append(Paragraph(analysis_text, self.styles['BodyText']))
            
        self.elements.append(Spacer(1, 20))
        
    def _create_timeline_chart(self, timeline: Dict) -> Optional[io.BytesIO]:
        """Create timeline line chart."""
        try:
            plt.figure(figsize=(9, 4))
            
            dates = list(timeline.keys())
            values = list(timeline.values())
            
            # Create gradient fill
            plt.fill_between(range(len(dates)), values, alpha=0.3, color='#6366f1')
            plt.plot(range(len(dates)), values, color='#6366f1', linewidth=2, marker='o', 
                    markersize=4, markerfacecolor='white', markeredgecolor='#6366f1', markeredgewidth=2)
            
            # Customize
            plt.xticks(range(len(dates)), dates, rotation=45, ha='right', fontsize=8)
            plt.ylabel('Alerts', fontsize=10, color='#334155')
            plt.title('Daily Threat Activity', fontsize=12, fontweight='bold', color='#1e293b', pad=15)
            
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_color('#e2e8f0')
            plt.gca().spines['bottom'].set_color('#e2e8f0')
            plt.grid(axis='y', alpha=0.3, color='#e2e8f0')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            plt.close()
            
            return buf
        except Exception as e:
            print(f"Error creating timeline chart: {e}")
            return None
            
    def _add_top_threats(self, report_data: Dict):
        """Add top threats section."""
        self.elements.append(Paragraph(
            "🎯 Top Threat Sources",
            self.styles['SectionHeader']
        ))
        
        top_sources = report_data.get('top_sources', [])
        
        if top_sources:
            table_data = [['Rank', 'Source IP', 'Alert Count', 'Primary Attack', 'Risk Level']]
            
            for i, source in enumerate(top_sources[:10], 1):
                risk_level = 'Critical' if source.get('count', 0) > 50 else 'High' if source.get('count', 0) > 20 else 'Medium'
                table_data.append([
                    str(i),
                    source.get('ip', 'Unknown'),
                    f"{source.get('count', 0):,}",
                    source.get('primary_attack', 'Various').replace('_', ' ').title(),
                    risk_level
                ])
            
            table = Table(table_data, colWidths=[40, 120, 80, 120, 80])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.COLORS['danger']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('TOPPADDING', (0, 0), (-1, 0), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.COLORS['light']]),
                ('BOX', (0, 0), (-1, -1), 1, self.COLORS['light']),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, self.COLORS['light']),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            
            self.elements.append(table)
        else:
            self.elements.append(Paragraph(
                "No significant threat sources identified in this period.",
                self.styles['BodyText']
            ))
            
        self.elements.append(Spacer(1, 20))
        
    def _add_recommendations(self, report_data: Dict):
        """Add security recommendations section."""
        self.elements.append(PageBreak())
        self.elements.append(Paragraph(
            "💡 Security Recommendations",
            self.styles['SectionHeader']
        ))
        
        summary = report_data.get('summary', {})
        critical = summary.get('critical_alerts', 0)
        high = summary.get('high_alerts', 0)
        
        recommendations = []
        
        if critical > 0:
            recommendations.append(
                f"<b>URGENT:</b> Address {critical} critical alerts immediately. "
                "These represent the highest risk to your network infrastructure."
            )
        
        if high > 10:
            recommendations.append(
                f"<b>HIGH PRIORITY:</b> Investigate {high} high-severity alerts. "
                "Consider implementing additional access controls for affected systems."
            )
        
        recommendations.extend([
            "<b>Network Segmentation:</b> Review network segmentation policies to limit lateral movement potential.",
            "<b>Access Controls:</b> Audit and strengthen authentication mechanisms, especially for critical systems.",
            "<b>Monitoring Enhancement:</b> Consider expanding log collection and correlation capabilities.",
            "<b>Incident Response:</b> Ensure incident response procedures are up-to-date and team is trained.",
            "<b>Patch Management:</b> Verify all systems are patched against known vulnerabilities.",
            "<b>User Awareness:</b> Conduct security awareness training focused on recent attack patterns.",
        ])
        
        for rec in recommendations:
            self.elements.append(Paragraph(f"• {rec}", self.styles['BodyText']))
            self.elements.append(Spacer(1, 5))
            
        self.elements.append(Spacer(1, 20))
        
    def _add_footer(self, report_data: Dict):
        """Add report footer."""
        self.elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=self.COLORS['light'],
            spaceBefore=20,
            spaceAfter=10
        ))
        
        footer_text = f"""
        <i>This report was automatically generated by AI-NIDS (AI Network Intrusion Detection System).<br/>
        Report ID: RPT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}<br/>
        Classification: CONFIDENTIAL - For authorized personnel only.</i>
        """
        
        self.elements.append(Paragraph(footer_text, self.styles['Footer']))
        
    def generate_report(self, report_data: Dict) -> io.BytesIO:
        """Generate the complete PDF report."""
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
            title=self.title,
            author="AI-NIDS Security System"
        )
        
        self.elements = []
        
        # Build report sections
        self._add_header_section(report_data)
        self._add_executive_summary(report_data)
        self._add_threat_analysis(report_data)
        self._add_severity_breakdown(report_data)
        self._add_timeline_analysis(report_data)
        self._add_top_threats(report_data)
        self._add_recommendations(report_data)
        self._add_footer(report_data)
        
        # Build PDF
        doc.build(self.elements)
        
        buffer.seek(0)
        return buffer


def generate_security_report(
    alerts: list,
    flows: list,
    days: int = 30
) -> io.BytesIO:
    """
    Generate a security report from alert and flow data.
    
    Args:
        alerts: List of Alert objects from database
        flows: List of NetworkFlow objects from database
        days: Number of days in the report period
        
    Returns:
        BytesIO buffer containing the PDF report
    """
    from collections import Counter
    from datetime import datetime, timedelta
    
    # Calculate summary statistics
    total_alerts = len(alerts)
    severity_counts = Counter(a.severity for a in alerts)
    attack_counts = Counter(a.attack_type for a in alerts)
    
    # Build timeline (daily counts)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    timeline = {}
    for i in range(min(days, 14)):  # Last 14 days for timeline
        day = (end_date - timedelta(days=i)).strftime('%m/%d')
        day_start = (end_date - timedelta(days=i)).replace(hour=0, minute=0, second=0)
        day_end = (end_date - timedelta(days=i)).replace(hour=23, minute=59, second=59)
        count = sum(1 for a in alerts if day_start <= a.timestamp <= day_end)
        timeline[day] = count
    
    # Reverse timeline to chronological order
    timeline = dict(reversed(list(timeline.items())))
    
    # Top threat sources
    source_counter = Counter(a.source_ip for a in alerts)
    top_sources = []
    for ip, count in source_counter.most_common(10):
        primary_attack = Counter(
            a.attack_type for a in alerts if a.source_ip == ip
        ).most_common(1)
        top_sources.append({
            'ip': ip,
            'count': count,
            'primary_attack': primary_attack[0][0] if primary_attack else 'Unknown'
        })
    
    # Calculate detection rate (mock - in real system this would come from detector)
    detection_rate = 96.8  # Example value
    
    # Build report data
    report_data = {
        'days': days,
        'summary': {
            'total_alerts': total_alerts,
            'total_flows': len(flows),
            'critical_alerts': severity_counts.get('critical', 0),
            'high_alerts': severity_counts.get('high', 0),
            'medium_alerts': severity_counts.get('medium', 0),
            'low_alerts': severity_counts.get('low', 0),
            'resolved_alerts': sum(1 for a in alerts if a.resolved),
            'detection_rate': detection_rate
        },
        'attack_types': dict(attack_counts),
        'severity_breakdown': {
            'critical': severity_counts.get('critical', 0),
            'high': severity_counts.get('high', 0),
            'medium': severity_counts.get('medium', 0),
            'low': severity_counts.get('low', 0),
            'info': severity_counts.get('info', 0)
        },
        'timeline': timeline,
        'top_sources': top_sources
    }
    
    # Generate PDF
    generator = SecurityReportGenerator()
    return generator.generate_report(report_data)
