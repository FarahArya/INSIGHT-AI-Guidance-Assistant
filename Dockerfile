# Base image with Python, OpenCV, and pip
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all source files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir ultralytics opencv-python-headless numpy

# Set the entrypoint
CMD ["python3", "Insight/insight_deploy/insight_infer.py"]
