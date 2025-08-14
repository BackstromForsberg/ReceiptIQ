# Dockerfile (development)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    FLASK_DEBUG=1 \
    DEBIAN_FRONTEND=noninteractive
    
# Install system dependencies for Tesseract and OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install deps (great for caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (compose will bind-mount over this in dev)
COPY . .

EXPOSE 5000

# Launch Flask dev server (debug + auto-reload)
# CMD ["python", "-m", "flask", "run"]
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000", "--debug"]