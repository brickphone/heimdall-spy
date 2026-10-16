FROM python:3.12.8-slim-bookworm

WORKDIR /app

COPY pyproject.toml uv.lock README.md /app/
COPY swespy /app/swespy
COPY scripts /app/scripts

RUN pip install --no-cache-dir uv==0.9.5 && uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/data/huggingface

EXPOSE 8000

ENTRYPOINT ["python", "scripts/serve.py"]
