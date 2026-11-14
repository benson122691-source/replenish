# ReplenishAI: one container serving the API + dashboard.
# artifacts/ is git-ignored, so the build runs the experiment on SYNTHETIC data to produce it.
# The API never trains at request time (docs/04-sad.md); everything it serves is baked in here.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    REPLENISHAI_ARTIFACTS=/app/artifacts

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY replenishai/ replenishai/
COPY scripts/run_experiment.py scripts/
COPY static/ static/
COPY docs/ docs/

# No data/raw in the image -> run_experiment.py generates synthetic M5-format data (data_source: synthetic)
RUN python scripts/run_experiment.py \
 && test -f artifacts/results.json \
 && test -f artifacts/holdout_forecasts.csv \
 && test -f artifacts/series_stats.csv

RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)"

# Railway (and similar hosts) inject PORT; default to 8000 locally
CMD ["sh", "-c", "uvicorn replenishai.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
