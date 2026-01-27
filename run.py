"""
================================================================================
🔥 GEOTEMPORAL FUSION - RUN SERVER
================================================================================

Start the FastAPI server for wildfire prediction API.

Usage:
    python run.py                    # Start on default port 8000
    python run.py --port 7860        # Start on custom port (Hugging Face)
    python run.py --reload           # Development mode with auto-reload

Production:
    uvicorn app.main:app --host 0.0.0.0 --port 7860
================================================================================
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Start the GeoTemporalFusion API server"""
    parser = argparse.ArgumentParser(description='Run GeoTemporalFusion API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔥 GEOTEMPORAL FUSION API SERVER")
    print("=" * 60)
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   URL: http://localhost:{args.port}")
    print("=" * 60)
    
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
