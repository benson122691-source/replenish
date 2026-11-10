# ReplenishAI: one container = API + dashboard page (docs/04-sad.md, docs/deploy-railway.md).
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY replenishai ./replenishai
COPY scripts ./scripts
COPY static ./static
COPY docs ./docs

# artifacts/ is git-ignored, so build it here. With no files in data/raw/ the script generates
# SYNTHETIC M5-format data; every API response and the page then say data_source = synthetic.
RUN python scripts/run_experiment.py

EXPOSE 8000
# Railway (and most PaaS) set $PORT; locally it falls back to 8000.
CMD ["sh", "-c", "uvicorn replenishai.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
