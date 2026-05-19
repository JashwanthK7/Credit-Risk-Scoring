FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api_requirements.txt .
RUN pip install --no-cache-dir -r api_requirements.txt

COPY config/ ./config/
COPY src/ ./src/
COPY artifacts/best_model.pkl ./artifacts/
COPY serve_api.py .

EXPOSE 8000

CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000"]