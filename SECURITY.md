# Security Policy

## Reporting a Vulnerability

The AI-NIDS team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose your findings.

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

📧 **security@ai-nids.org**

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include as much of the following information as possible:

- **Type of vulnerability** (e.g., SQL injection, XSS, authentication bypass)
- **Full paths of source file(s)** related to the vulnerability
- **Location of the affected source code** (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept or exploit code** (if possible)
- **Impact of the vulnerability** and how an attacker might exploit it
- **Suggested fix** (if you have one)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.

2. **Communication**: We will keep you informed of the progress towards a fix.

3. **Fix Development**: We will develop a fix and prepare a security release.

4. **Disclosure**: Once the fix is ready, we will:
   - Release the patched version
   - Publish a security advisory
   - Credit you for the discovery (unless you prefer to remain anonymous)

### Safe Harbor

We support safe harbor for security researchers who:

- Make a good faith effort to avoid privacy violations, destruction of data, and interruption or degradation of our services
- Only interact with accounts you own or with explicit permission of the account holder
- Do not exploit a security issue you discover for any reason other than for testing purposes
- Report any vulnerability you've discovered promptly
- Do not use or disclose the vulnerability for purposes other than to help us fix it

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

When deploying AI-NIDS in production:

### Authentication & Access Control

- [ ] Change default admin credentials immediately
- [ ] Use strong, unique passwords (min 12 characters)
- [ ] Enable two-factor authentication when available
- [ ] Implement role-based access control (RBAC)
- [ ] Regularly audit user accounts and permissions

### Network Security

- [ ] Deploy behind a reverse proxy (nginx, Traefik)
- [ ] Use HTTPS with valid SSL/TLS certificates
- [ ] Restrict access to trusted IP ranges
- [ ] Use a Web Application Firewall (WAF)
- [ ] Segment the network appropriately

### Data Protection

- [ ] Encrypt sensitive data at rest
- [ ] Use secure database connections
- [ ] Regularly backup data
- [ ] Implement data retention policies
- [ ] Sanitize logs of sensitive information

### Application Security

- [ ] Keep dependencies updated (`pip install --upgrade -r requirements.txt`)
- [ ] Run security scans regularly
- [ ] Use environment variables for secrets (never commit to git)
- [ ] Enable CSRF protection
- [ ] Set secure cookie flags

### Monitoring & Logging

- [ ] Enable comprehensive logging
- [ ] Monitor for unusual activity
- [ ] Set up alerts for security events
- [ ] Regularly review security logs
- [ ] Implement intrusion detection for AI-NIDS itself

## Configuration Security

### Environment Variables

Sensitive configuration should be stored in environment variables:

```bash
# Required security configurations
export SECRET_KEY="your-super-secret-key-min-32-chars"
export DATABASE_URL="postgresql://user:pass@localhost/db"
export FLASK_ENV="production"

# Optional but recommended
export WTF_CSRF_ENABLED="true"
export SESSION_COOKIE_SECURE="true"
export SESSION_COOKIE_HTTPONLY="true"
```

### Production Checklist

```python
# config.py production settings
DEBUG = False
TESTING = False
SECRET_KEY = os.environ.get('SECRET_KEY')  # Never hardcode!
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
PREFERRED_URL_SCHEME = 'https'
```

## Known Security Considerations

### API Key Security

- API keys are stored hashed in the database
- Keys are only shown once upon generation
- Implement key rotation policies
- Monitor API key usage for anomalies

### ML Model Security

- Models should be validated before loading
- Be cautious of adversarial inputs
- Regularly retrain models with new threat data
- Validate all input data before processing

## Security Updates

Security updates will be published as:

1. **GitHub Security Advisories**
2. **Release notes** with CVE references
3. **Email notifications** (subscribe to security mailing list)

## Acknowledgments

We thank the following individuals for responsibly disclosing vulnerabilities:

- *Your name could be here!*

---

## Contact

For security-related inquiries:

- 📧 Email: security@ai-nids.org
- 🔐 PGP Key: [Available upon request]

For general inquiries, please use [GitHub Discussions](https://github.com/ai-nids/discussions).
