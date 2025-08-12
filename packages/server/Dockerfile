# Dockerfile (development)
FROM python:3.12-slim

# Faster logs & no .pyc files
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dev deps
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy your app (compose will mount the live code over this)
COPY . .

EXPOSE 5000

# Run Flask’s built-in server with debug & reload
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]