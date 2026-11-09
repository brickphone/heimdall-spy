# syntax=docker/dockerfile:1
# Multi-stage build: dependencies are resolved in a builder stage; the runtime stage
# contains only Python, the locked virtualenv and the API code. Model weights, the
# gallery index and reference images are NOT in the image: mount them at /app/data.

FROM python:3.12.8-slim-bookworm AS builder
# Optional extras, e.g. --build-arg EXTRAS=heic for iPhone HEIC uploads. pillow-heif
# wheels are GPL-2.0 (bundled x265); the default image leaves them out.
ARG EXTRAS=""
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.9.5
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PYTHON=/usr/local/bin/python3.12
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY swespy ./swespy
COPY scripts/serve.py ./scripts/serve.py
RUN uv sync --frozen --no-dev ${EXTRAS:+--extra $EXTRAS}

FROM python:3.12.8-slim-bookworm AS runtime
# OpenCV (RANSAC for re-ranking) needs these shared libraries; curl is for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
# FAISS_OPT_LEVEL: faiss-cpu 1.9's ARM64 loader probes SVE via numpy.distutils, which
# does not exist on Python 3.12; "generic" skips the probe (exact flat search either way).
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 FAISS_OPT_LEVEL=generic \
    HF_HOME=/app/data/huggingface HF_HUB_OFFLINE=1 TORCH_HOME=/app/data/torch \
    MEGALOC_DEVICE=cpu RERANK_DEVICE=cpu RERANK_ENABLED=true TESTING=false
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s \
    CMD curl -fsS http://127.0.0.1:8000/ready || exit 1
ENTRYPOINT ["python", "scripts/serve.py"]
