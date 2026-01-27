# ================================================================================
# DOCKERFILE FOR HUGGING FACE SPACES DEPLOYMENT
# ================================================================================
# 
# This Dockerfile deploys the GeoTemporalFusion FastAPI backend to Hugging Face 
# Spaces using the Docker SDK. Hugging Face Spaces provides FREE hosting with:
#   - 16GB RAM
#   - 2 vCPU
#   - Persistent storage
#   - Custom domains
#
# Deployment Steps:
# 1. Create a new Space on huggingface.co/spaces
# 2. Select "Docker" as the SDK
# 3. Upload this project to the Space repository
# 4. The Space will automatically build and deploy
#
# ================================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY step4_model_architecture.py .
COPY step5_train.py .
COPY config.py .

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed/weather /app/data/processed/masks

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
