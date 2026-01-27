"""
Simple directory creation script - no dependencies required
Run this first to ensure all directories exist
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

directories = [
    os.path.join(BASE_DIR, "data"),
    os.path.join(BASE_DIR, "data", "raw"),
    os.path.join(BASE_DIR, "data", "raw", "images"),
    os.path.join(BASE_DIR, "data", "processed"),
    os.path.join(BASE_DIR, "data", "processed", "weather"),
    os.path.join(BASE_DIR, "data", "processed", "masks"),
]

print("Creating required directories...")
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"  ✓ {os.path.relpath(directory, BASE_DIR)}")

print("\n✓ All directories ready!")
