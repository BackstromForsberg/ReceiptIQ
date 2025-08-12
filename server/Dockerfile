# Dockerfile (development)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    FLASK_DEBUG=1

WORKDIR /app

# Install deps (great for caching)
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source (compose will bind-mount over this in dev)
COPY . .

EXPOSE 5000

# Launch Flask dev server (debug + auto-reload)
CMD ["python", "-m", "flask", "run"]