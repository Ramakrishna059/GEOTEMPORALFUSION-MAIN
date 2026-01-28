#!/bin/bash
# ============================================================================
# RENDER START SCRIPT
# ============================================================================
# Starts the FastAPI application with proper settings for production
# ============================================================================

echo "🚀 Starting GeoTemporalFusion API..."
echo "📦 Python Version: $(python --version)"
echo "📍 Working Directory: $(pwd)"
echo "🔧 Port: $PORT"

# Start FastAPI with Uvicorn
exec uvicorn main_geospatial_v3:app \
    --host 0.0.0.0 \
    --port ${PORT:-10000} \
    --workers 1 \
    --timeout-keep-alive 30 \
    --log-level info
