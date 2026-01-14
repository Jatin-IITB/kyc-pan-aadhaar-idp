# Use an official Python runtime as a parent image (Slim variant for size)
FROM python:3.9-slim

# 1. Install System Dependencies
# FIX: Replaced 'libgl1-mesa-glx' with 'libgl1' and 'libgl1-mesa-dri'
# This fixes the "Package has no installation candidate" error on newer Debian Bookworm/Trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libgomp1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 2. Set Working Directory
WORKDIR /app

# 3. Install Python Dependencies
COPY requirements.txt .
# Using --default-timeout to prevent connection reset on slow networks
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# 4. Copy Your Code
COPY . .

# 5. Set Environment Variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV REDIS_URL=redis://redis:6379/0
ENV KYC_OLLAMA_TIMEOUT_S=20
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "apps/api/main.py"]
