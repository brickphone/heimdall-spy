FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY swespy /app/swespy
COPY scripts /app/scripts

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "swespy.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
