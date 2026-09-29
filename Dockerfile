# syntax=docker/dockerfile:1
# ===========================================================================
# Repeat-Purchase Prediction — container build
# ===========================================================================
# Two targets share one dependency layer:
#
#   api        FastAPI prediction service  <- default (last stage, deployable)
#   dashboard  Streamlit UI pointed at the API
#
#   docker build -t churn-api .                        # API only
#   docker build --target dashboard -t churn-ui .      # dashboard only
#   docker compose up --build                          # both, wired together
#
# The trained artifacts in model/ are copied into the image, so rebuild the
# image after retraining (python src/train_model.py). They are never rebuilt
# at container start — startup stays fast and deterministic.
# ===========================================================================

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer is reused across code-only rebuilds.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt


# ---------------------------------------------------------------------------
# Streamlit dashboard
# ---------------------------------------------------------------------------
FROM base AS dashboard

COPY frontend/ ./frontend/

ENV PORT=8501 \
    API_URL=http://api:8000

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

CMD streamlit run frontend/dashboard.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false


# ---------------------------------------------------------------------------
# FastAPI prediction service (default target)
# ---------------------------------------------------------------------------
FROM base AS api

# app/ + model/ is the whole runtime surface: inference unpickles a complete
# sklearn Pipeline, so it needs no training code or raw data.
COPY app/ ./app/
COPY model/ ./model/

ENV PORT=8000 \
    MODEL_DIR=/app/model

RUN useradd --create-home --uid 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# No curl in the slim image, so probe /health with the stdlib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=4)" || exit 1

# Shell form so ${PORT} is expanded — Render injects PORT at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
