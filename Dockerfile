# AI-NIDS - Network Intrusion Detection System
# Multi-stage Docker build for production deployment

# ========== Stage 1: Builder ==========
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== Stage 2: Production ==========
FROM python:3.11-slim as production

# Labels
LABEL maintainer="AI-NIDS Team"
LABEL version="1.0.0"
LABEL description="AI-Powered Network Intrusion Detection System"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    FLASK_ENV=production \
    FLASK_APP=wsgi:application \
    PORT=8000 \
    WORKERS=4

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -r nids \
    && mkdir -p /app/logs /app/models /app/data /app/instance \
    && chown -R nids:nids /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/nids/.local

# Add local bin to PATH
ENV PATH=/home/nids/.local/bin:$PATH

# Copy application code
COPY --chown=nids:nids . .

# Create necessary directories
RUN mkdir -p /app/instance /app/logs /app/data /app/models

# Switch to non-root user
USER nids

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WORKERS} --threads 2 --timeout 120 --keep-alive 5 --log-level info --access-logfile - --error-logfile - wsgi:application"]
