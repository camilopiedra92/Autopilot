# syntax=docker/dockerfile:1
# ── Build stage ───────────────────────────────────────────────────────
# Pin exact Python version for reproducible builds
FROM python:3.13.2-slim AS builder

WORKDIR /app

# Ensure apt doesn't throw warnings about lack of UI
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps for HTTP/2 (h2 + hpack) and curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and source code
COPY pyproject.toml ./
COPY autopilot/ ./autopilot/

# Install Python dependencies (no-cache-dir for smaller image & Cloud Build compatibility)
# Since we use Kaniko, the layer caching is handled externally, so we can copy source first.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# ── Runtime stage ────────────────────────────────────────────────────
FROM python:3.13.2-slim AS runtime

WORKDIR /app

# Ensure apt doesn't throw warnings about lack of UI
ENV DEBIAN_FRONTEND=noninteractive

# Install curl for health checks in runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy ONLY runtime-required application code (allowlist pattern)
# Secrets (.env, credentials.json, token.json) excluded by omission
# Docs, scripts, tests, config files excluded by omission
COPY app.py ./
COPY autopilot/ ./autopilot/
COPY workflows/ ./workflows/

# Security: run as non-root
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

# ── Environment ──────────────────────────────────────────────────────
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Suppress gRPC C-level fork warnings (false positive from async threads)
ENV GRPC_ENABLE_FORK_SUPPORT=0
ENV GRPC_VERBOSITY=ERROR

# Cloud Run injects these automatically:
#   GOOGLE_CLOUD_PROJECT — GCP project ID
#   K_SERVICE           — Cloud Run service name
#   K_REVISION          — Cloud Run revision name
#   K_CONFIGURATION     — Cloud Run configuration name
# App-level env vars (set via gcloud/cloudbuild.yaml):
#   EVENTBUS_BACKEND              — "memory" (default) or "pubsub" (production)
#   SESSION_BACKEND               — "memory" (default) or "firestore" (production)
#   MEMORY_BACKEND                — "memory" (default) or "vertexai" (production)
#   MEMORY_AGENT_ENGINE_ID        — Vertex AI Agent Engine ID (required when MEMORY_BACKEND=vertexai)
#   ARTIFACT_BACKEND              — "memory" (default) or "gcs" (production)
#   ARTIFACT_GCS_BUCKET           — GCS bucket name (required when ARTIFACT_BACKEND=gcs)
#   CONTEXT_CACHE_MIN_TOKENS      — Min tokens to trigger caching (default: 2048)
#   CONTEXT_CACHE_TTL_SECONDS     — Cache TTL in seconds (default: 1800)
#   CONTEXT_CACHE_INTERVALS       — Max cache reuses before refresh (default: 10)
# Secrets are injected via Secret Manager → env vars (see cloudbuild.yaml)

EXPOSE ${PORT}

# ── Health check ─────────────────────────────────────────────────────
# Uses curl instead of spawning a Python interpreter (10x faster)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ── Cloud Run startup probe (annotation for documentation) ──────────
# Cloud Run uses TCP startup probe by default on the container port.
# For custom startup probes, configure via gcloud or cloudbuild.yaml:
#   --startup-cpu-boost
#   --cpu-throttling (disabled for faster cold starts)

# ── Start ────────────────────────────────────────────────────────────
# Cloud Run sets PORT env var; uvicorn listens on 0.0.0.0:$PORT
#
# Worker strategy (Cloud Run --cpu=1):
#   Default: 1 worker — single async event loop handles concurrency=80
#   Tunable: set WEB_CONCURRENCY env var (Uvicorn reads it natively)
#   Why: I/O-bound app; multiple workers waste memory with zero CPU gain
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
