"""
Setup script to initialize the project
Run this after cloning to prepare the environment
"""

import os
import sys
import subprocess
from config import create_directories

def setup_project():
    """Initialize the project by creating directories and installing dependencies"""
    
    print("=" * 60)
    print("🔥 GEO-TEMPORAL FUSION - PROJECT SETUP 🔥")
    print("=" * 60)
    
    # Step 1: Create directories
    print("\n📁 Creating required directories...")
    try:
        create_directories()
        print("✓ All directories created successfully!")
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
        return False
    
    # Step 2: Check Python version
    print(f"\n🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required!")
        return False
    print("✓ Python version is compatible")
    
    # Step 3: Install requirements (optional prompt)
    print("\n📦 Install Python packages?")
    print("   Run: pip install -r requirements.txt")
    response = input("   Install now? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing packages: {e}")
            print("   Try manually: pip install -r requirements.txt")
            return False
    
    # Step 4: Generate sample data
    print("\n🎯 Generate sample fire data for demo?")
    response = input("   Run step1_bypass.py? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            subprocess.check_call([sys.executable, "step1_bypass.py"])
            print("✓ Sample data generated!")
        except Exception as e:
            print(f"⚠ Could not run step1_bypass.py: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✓ PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("\n📖 Next steps:")
    print("   1. python step1_bypass.py         # Generate sample fire data")
    print("   2. python step2_get_images.py     # Download satellite images")
    print("   3. python step3_process_data.py   # Process weather data")
    print("   4. python step5_train.py          # Train the model")
    print("   5. python step6_visualize.py      # Visualize predictions")
    print("   6. uvicorn main:app --reload      # Launch web interface")
    print("\n📚 Read README.md for detailed documentation")
    print("=" * 60 + "\n")
    
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
