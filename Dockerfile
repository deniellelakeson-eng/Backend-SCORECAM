# HerbaScan Backend Dockerfile
# For deployment to Railway, Render, or other cloud platforms

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV and TensorFlow)
# Note: libgl1-mesa-glx is replaced by libgl1 in newer Debian versions
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check (using curl - Railway sets PORT dynamically, use env var)
# Note: Railway also has its own healthcheck, but Docker healthcheck helps with readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/health || exit 1"

# Run the application (Railway sets PORT environment variable)
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"

