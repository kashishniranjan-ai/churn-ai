# ===========================================================================
# Dockerfile — Customer Churn Predictor
# ===========================================================================
# Multi-stage build optimized for production. Installs only runtime deps,
# copies serialized model artifacts, and exposes both the FastAPI backend
# and the Streamlit frontend via a supervisor process.
# ===========================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# 1. Install system dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        supervisor \
        curl && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Copy application code
# ---------------------------------------------------------------------------
COPY model/ ./model/
COPY app/ ./app/
COPY frontend/ ./frontend/

# ---------------------------------------------------------------------------
# 4. Supervisor config (runs both FastAPI + Streamlit)
# ---------------------------------------------------------------------------
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# ---------------------------------------------------------------------------
# 5. Expose ports
# ---------------------------------------------------------------------------
EXPOSE 8000 8501

# ---------------------------------------------------------------------------
# 6. Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# 7. Run
# ---------------------------------------------------------------------------
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
