# Backend Dockerfile for CarPriceML
# FastAPI backend with ML model for car price prediction

FROM python:3.11-slim

# Set metadata
LABEL maintainer="CarPriceML Team"
LABEL description="FastAPI backend for used car price prediction"
LABEL version="3.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with retries and timeout
RUN pip install --default-timeout=1000 --retries 5 --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY schemas.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/models

# Expose port 8000
EXPOSE 8000

# Health check - use curl instead of requests for lighter container
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
