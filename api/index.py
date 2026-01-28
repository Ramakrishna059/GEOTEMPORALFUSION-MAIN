"""
================================================================================
VERCEL SERVERLESS FUNCTION - FULL FASTAPI APP
================================================================================
This file wraps the FastAPI app for Vercel's serverless function runtime.
All routes from app.main will be available under /api/*
================================================================================
"""

from app.main import app

# This is the handler that Vercel will call
# The app is already configured in app/main.py
handler = app
