# HerbaScan Backend Dockerfile
# For deployment to Railway, Render, or other cloud platforms

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Health check (will use PORT env var if available)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, requests; port = os.environ.get('PORT', '8000'); requests.get(f'http://localhost:{port}/health')"

# Run the application (PORT env var will be read by main.py)
CMD ["python", "main.py"]

