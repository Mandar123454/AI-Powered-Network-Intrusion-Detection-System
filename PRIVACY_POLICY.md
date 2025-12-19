# Privacy Policy

**Effective Date**: January 2025  
**Last Updated**: January 2025  
**Version**: 1.0

---

## 1. Introduction

This Privacy Policy ("Policy") describes how the **AI-NIDS** project (referred to as "System," "We," "Us," or "Our") collects, uses, stores, and protects data. AI-NIDS is an open-source Network Intrusion Detection System designed for security operations.

### Applicability
- This Policy applies to all users of AI-NIDS
- For self-hosted deployments: Users control their own data
- For official cloud instances: This Policy describes our data practices
- Users deploying AI-NIDS should adapt this Policy to their jurisdiction

---

## 2. Data We Collect

### 2.1 Network Traffic Data

**Types Collected:**
- Network flow data (source IP, destination IP, ports, protocols)
- Packet headers and metadata (not packet payloads)
- DNS queries
- HTTP request headers (URLs, user agents - not request bodies)
- Network statistics (bytes, packets, duration)

**Purpose:** Threat detection and anomaly analysis

**Legal Basis:** Legitimate interest in network security

**Retention:** Default 30 days (configurable)

**Note:** AI-NIDS does NOT capture packet payloads or decrypt encrypted traffic.

### 2.2 User Account Information

**Types Collected:**
- Username and email address
- Password (stored using bcrypt hashing)
- Full name (optional)
- Profile preferences and API keys
- Login history and activity logs

**Purpose:** User authentication and access control

**Legal Basis:** Contractual necessity

**Retention:** For duration of account + 90 days after deletion

### 2.3 System Events and Logs

**Types Collected:**
- Alert events and detections
- User actions (alerts viewed, reports generated)
- System events (model retraining, configuration changes)
- Error and debug logs
- API access logs

**Purpose:** Audit trail, debugging, performance analysis

**Legal Basis:** Legitimate interest in system security and reliability

**Retention:** 90 days by default (configurable)

### 2.4 ML Model Training Data

**Types Collected:**
- Labeled network flow features
- Attack pattern samples
- Baseline normal behavior

**Purpose:** Training and improving detection models

**Legal Basis:** Legitimate interest in security improvement

**Retention:** Until model is retired/retrained

**Data Anonymization:** Features are extracted from raw traffic, not identifying individuals

### 2.5 Optional Threat Intelligence

If enabled, user may provide:
- IOC (Indicator of Compromise) feeds
- External threat feeds from security vendors
- API keys for threat intelligence services

**Purpose:** Enhanced threat detection

**Legal Basis:** User consent and configuration choice

---

## 3. How We Use Your Data

### 3.1 Primary Uses

| Purpose | Data Type | Legal Basis |
|---------|-----------|------------|
| Network threat detection | Network flows | Legitimate interest |
| User authentication | User accounts | Contractual |
| System auditing | Logs and events | Legal obligation |
| Model improvement | Training data | Legitimate interest |
| Security analytics | Aggregated flows | Legitimate interest |
| Alert correlation | Network data | Legitimate interest |

### 3.2 Data Processing

- **No sale of data** - We never sell user data to third parties
- **No advertising** - AI-NIDS contains no advertising or tracking
- **No profiling** - We don't profile individual users for non-security purposes
- **Security-only use** - Data used only for detection and security

---

## 4. Data Sharing and Disclosure

### 4.1 Who We Share Data With

**We share data in these cases:**

1. **Law Enforcement** - When required by valid legal process (subpoena, warrant)
2. **Service Providers** - Cloud storage providers, if using cloud deployment
3. **Federated Learning** - When explicitly enabled, anonymized threat patterns with other nodes
4. **Security Researchers** - Aggregated, anonymized data for research (with consent)

### 4.2 Who We DON'T Share Data With

- ❌ Marketing companies
- ❌ Data brokers
- ❌ Advertisers
- ❌ Unvetted third parties
- ❌ Affiliates for profit

### 4.3 Subpoena Transparency

If we receive legal demands, we will:
1. Review the legal validity of the request
2. Notify the affected user when legally permitted
3. Provide only the minimum data required
4. Document all requests

---

## 5. Data Security

### 5.1 Encryption

- **At Rest**: AES-256 encryption for sensitive data
- **In Transit**: TLS 1.3 for all network communications
- **Database**: Encrypted password hashing using bcrypt
- **API Keys**: Encrypted storage and transmission

### 5.2 Access Controls

- Role-Based Access Control (RBAC)
  - Admin: Full access
  - Analyst: Read/manage alerts
  - Viewer: Read-only dashboard
- Multi-factor authentication (optional)
- Session timeout: 1 hour default
- IP whitelist capabilities

### 5.3 Infrastructure Security

- Network isolation (if using cloud)
- Firewall rules and security groups
- Regular security updates and patches
- Vulnerability scanning
- Secure configuration hardening

### 5.4 Backup & Disaster Recovery

- Automated daily backups
- Encryption of backups
- Off-site backup storage
- Tested recovery procedures

---

## 6. User Rights (GDPR/CCPA Compliance)

### 6.1 EU Users (GDPR)

You have the following rights:

**Right to Access** (Article 15)
- Request copy of your personal data
- Command: `PATCH /api/profile/data-export`

**Right to Correction** (Article 16)
- Update inaccurate data
- URL: Dashboard → Profile → Edit

**Right to Erasure** (Article 17)
- Request deletion of your account
- Command: `DELETE /api/profile`
- Data deleted within 30 days (except legal holds)

**Right to Restrict Processing** (Article 18)
- Limit how your data is used
- Contact: privacy@ai-nids.dev

**Right to Data Portability** (Article 20)
- Receive your data in portable format
- Command: `PATCH /api/profile/data-export?format=json`

**Right to Object** (Article 21)
- Object to certain processing
- Contact: privacy@ai-nids.dev

**Right to Withdraw Consent**
- Withdraw previous consents anytime
- Dashboard → Settings → Privacy Preferences

### 6.2 California Users (CCPA)

California residents have rights under CCPA:

**Right to Know** (1798.100)
- What personal information is collected
- Command: Refer to Section 2 above

**Right to Delete** (1798.105)
- Request deletion (with exceptions)
- Command: `DELETE /api/profile`

**Right to Opt-Out** (1798.120)
- Opt-out of data sales
- Note: We don't sell data, so this is moot
- But you may still send: privacy@ai-nids.dev

**Right to Non-Discrimination** (1798.125)
- We don't discriminate based on privacy choices

### 6.3 Other Jurisdictions

Users in other jurisdictions have similar rights. Contact **privacy@ai-nids.dev** for information.

---

## 7. Data Retention

| Data Type | Retention Period | Deletion Method |
|-----------|-----------------|-----------------|
| Network flows | 30 days (default) | Automatic purge |
| User activity logs | 90 days | Automatic purge |
| Alerts | 1 year | Configurable retention |
| User accounts | Until deletion + 90 days | Secure deletion |
| Audit logs | 2 years | Automatic archive |
| Backups | 30 days | Secure deletion |
| ML training data | Until model retrain | Automatic cleanup |

---

## 8. Cookies and Tracking

### 8.1 Session Cookies

- **Name**: `session_id`
- **Purpose**: User authentication
- **Duration**: Session (until logout)
- **Required**: Yes

### 8.2 Preferences

- **Name**: `theme`, `language`
- **Purpose**: UI preferences
- **Duration**: 1 year
- **Required**: No (optional)

### 8.3 Analytics

**We DON'T use**:
- Google Analytics
- Third-party tracking pixels
- Behavioral tracking
- Cross-site tracking cookies

---

## 9. Third-Party Services

### 9.1 Optional Integrations

If you enable these services, their privacy policies apply:

| Service | Purpose | Privacy Policy |
|---------|---------|---|
| Slack | Notifications | https://slack.com/privacy |
| VirusTotal | Malware scanning | https://www.virustotal.com/privacy |
| Shodan | IP device info | https://www.shodan.io/policy |
| Azure Storage | Cloud backup | https://privacy.microsoft.com |

### 9.2 Data Shared

Only essential data shared:
- File hashes (to VirusTotal)
- IP addresses (to Shodan)
- Alert summaries (to Slack)

---

## 10. Children's Privacy

AI-NIDS is **NOT intended for children** under 13. We don't knowingly collect data from children. If we discover such data, we will delete it immediately.

---

## 11. Security Incident Response

### 11.1 If Data is Breached

We will:
1. **Investigate** the incident within 24 hours
2. **Notify** affected users within 72 hours (GDPR requirement)
3. **Provide** details:
   - What data was accessed
   - When it occurred
   - What we're doing to fix it
   - Steps you should take
4. **Document** the incident for transparency

### 11.2 Notification Method

- Email to registered address
- In-app notification
- Public security advisory
- GitHub security advisory

---

## 12. Policy Changes

### 12.1 Updates

We may update this Policy. Changes take effect:
- Immediately for clarifications
- 30 days notice for material changes
- Continued use = acceptance of changes

### 12.2 Notification

Users will be notified via:
- Email to registered address
- Dashboard banner
- GitHub releases page
- Email newsletter (if subscribed)

---

## 13. Open-Source Considerations

### 13.1 Self-Hosted Deployment

If you self-host AI-NIDS:
- **You control your data** - We have no access
- **You create your own policy** - Adapt this template
- **You are the controller** - GDPR/CCPA apply to you
- **You are responsible** - For security and compliance

### 13.2 Community Data Sharing

If you participate in:
- **Federated learning**: Anonymized threat patterns only
- **GitHub discussions**: Public data visible to all
- **Issue reports**: May include non-sensitive logs
- **Threat intelligence sharing**: Only with explicit consent

---

## 14. Contact & Rights Requests

### 14.1 Privacy Inquiries

**Email**: privacy@ai-nids.dev  
**Response Time**: Within 30 days (GDPR requirement)

### 14.2 Data Subject Rights Requests

To exercise your rights, send:
- Your full name
- Email address
- Specific right being exercised (access, deletion, etc.)
- Any relevant details

Example:
```
Subject: GDPR Right to Access Request

Dear Privacy Team,

I request a copy of all personal data you hold about me 
per GDPR Article 15.

Name: John Doe
Email: john@example.com

Thank you.
```

### 14.3 Data Protection Officer

Not currently appointed, but available upon request.

---

## 15. Special Data Categories

### 15.1 Sensitive Data

AI-NIDS may inadvertently process:
- Health information (if in network data)
- Financial information (if in network data)
- Racial/ethnic data (if in network data)

**Our Approach:**
- We don't intentionally collect sensitive data
- If captured, we minimize and delete quickly
- Strict access controls on such data
- Never used for non-security purposes

### 15.2 Biometric Data

**We don't collect**: Facial recognition, fingerprints, voice data

---

## 16. Policy Compliance

### 16.1 Standards

This policy complies with:
- ✅ GDPR (EU)
- ✅ CCPA (California)
- ✅ LGPD (Brazil)
- ✅ PIPEDA (Canada)
- ✅ PDPA (Singapore)

### 16.2 Audit & Assessment

- Annual privacy impact assessments
- Compliance audits by third parties
- Security testing and penetration testing
- Transparent reporting

---

## 17. Questions?

For questions about this Privacy Policy:

📧 **Email**: privacy@ai-nids.dev  
🐛 **GitHub**: [Open an Issue](https://github.com/yourusername/AI-NIDS/issues)  
💬 **Discussions**: [Community Forum](https://github.com/yourusername/AI-NIDS/discussions)  
📋 **Mailing List**: Subscribe at ai-nids.dev

---

## 18. Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2025 | Initial policy |

---

<div align="center">

**© 2025 AI-NIDS Project. All rights reserved.**

[Back to README](README.md) | [Security Policy](SECURITY.md) | [Contributing](CONTRIBUTING.md)

</div>
