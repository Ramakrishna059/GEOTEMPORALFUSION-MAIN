"""
================================================================================
🔍 PRE-FLIGHT DEPLOYMENT CHECKLIST - GEOTEMPORAL FUSION
================================================================================

Automated verification script to ensure project is ready for deployment.

Checks:
1. ✅ requirements.txt exists with required packages (torch, fastapi, uvicorn)
2. ✅ Model file exists at correct path
3. ✅ No hardcoded local paths (C:/Users/..., /home/..., etc.)
4. ✅ All required files present
5. ✅ Configuration files valid
6. ✅ API endpoints importable

Run: python verify_deploy.py
================================================================================
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent
REQUIRED_PACKAGES = ['torch', 'fastapi', 'uvicorn']
MODEL_PATH = BASE_DIR / "models" / "simple_fire_model.pth"
ALT_MODEL_PATH = BASE_DIR / "models" / "best_fire_model.pth"

# Patterns for hardcoded paths (should NOT appear in code)
# Using raw strings with escape sequences to avoid false positives in this file
HARDCODED_PATH_PATTERNS = [
    r'[A-Z]:' + r'\\' + r'Users' + r'\\',           # Windows paths
    r'[A-Z]:' + r'/Users/',                          # Windows paths with forward slashes
    r'/home/[a-zA-Z0-9_]+/',                         # Linux home directories
    r'/Users/[a-zA-Z0-9_]+/',                        # macOS home directories
]

# Files to skip during hardcoded path check (self-reference)
SKIP_FILES = ['verify_deploy.py']

# Required files for deployment
REQUIRED_FILES = [
    'requirements.txt',
    'Dockerfile',
    'app/main.py',
    'step4_model_architecture.py',
    'step5_train.py',
    'config.py',
]

# Files to scan for hardcoded paths
CODE_FILE_EXTENSIONS = ['.py', '.js', '.jsx', '.ts', '.tsx', '.json', '.yaml', '.yml']


# ============================================================
# CHECK FUNCTIONS
# ============================================================
class DeploymentChecker:
    """Pre-flight deployment verification"""
    
    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.warnings: List[str] = []
        
    def add_result(self, check_name: str, passed: bool, message: str):
        """Add a check result"""
        self.results.append((check_name, passed, message))
    
    def add_warning(self, message: str):
        """Add a warning"""
        self.warnings.append(message)
    
    def check_requirements_file(self) -> bool:
        """Check if requirements.txt exists and has required packages"""
        req_path = BASE_DIR / "requirements.txt"
        
        if not req_path.exists():
            self.add_result(
                "requirements.txt exists",
                False,
                f"File not found: {req_path}"
            )
            return False
        
        self.add_result(
            "requirements.txt exists",
            True,
            f"Found: {req_path}"
        )
        
        # Check for required packages
        with open(req_path, 'r') as f:
            content = f.read().lower()
        
        missing_packages = []
        for package in REQUIRED_PACKAGES:
            if package.lower() not in content:
                missing_packages.append(package)
        
        if missing_packages:
            self.add_result(
                "Required packages in requirements.txt",
                False,
                f"Missing packages: {', '.join(missing_packages)}"
            )
            return False
        
        self.add_result(
            "Required packages in requirements.txt",
            True,
            f"All required packages found: {', '.join(REQUIRED_PACKAGES)}"
        )
        return True
    
    def check_model_file(self) -> bool:
        """Check if trained model file exists"""
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            self.add_result(
                "Model file exists",
                True,
                f"Found: {MODEL_PATH} ({size_mb:.2f} MB)"
            )
            return True
        
        if ALT_MODEL_PATH.exists():
            size_mb = ALT_MODEL_PATH.stat().st_size / (1024 * 1024)
            self.add_result(
                "Model file exists",
                True,
                f"Found alternative: {ALT_MODEL_PATH} ({size_mb:.2f} MB)"
            )
            self.add_warning(f"Using alternative model path: {ALT_MODEL_PATH}")
            return True
        
        self.add_result(
            "Model file exists",
            False,
            f"Model not found at: {MODEL_PATH} or {ALT_MODEL_PATH}"
        )
        return False
    
    def check_hardcoded_paths(self) -> bool:
        """Scan code files for hardcoded local paths"""
        violations: Dict[str, List[str]] = {}
        
        for file_path in BASE_DIR.rglob('*'):
            # Skip directories, hidden files, and non-code files
            if file_path.is_dir():
                continue
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'node_modules' in file_path.parts or '.venv' in file_path.parts:
                continue
            if '__pycache__' in file_path.parts:
                continue
            if file_path.suffix not in CODE_FILE_EXTENSIONS:
                continue
            # Skip this verification script itself
            if file_path.name in SKIP_FILES:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                file_violations = []
                for pattern in HARDCODED_PATH_PATTERNS:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        file_violations.extend(matches)
                
                if file_violations:
                    rel_path = file_path.relative_to(BASE_DIR)
                    violations[str(rel_path)] = list(set(file_violations))
                    
            except Exception as e:
                self.add_warning(f"Could not read {file_path}: {e}")
        
        if violations:
            violation_details = "; ".join([
                f"{path}: {', '.join(paths)}" 
                for path, paths in list(violations.items())[:3]
            ])
            self.add_result(
                "No hardcoded local paths",
                False,
                f"Found {len(violations)} file(s) with hardcoded paths: {violation_details}"
            )
            return False
        
        self.add_result(
            "No hardcoded local paths",
            True,
            "No hardcoded paths found in code files"
        )
        return True
    
    def check_required_files(self) -> bool:
        """Check if all required files exist"""
        missing_files = []
        
        for file_path in REQUIRED_FILES:
            full_path = BASE_DIR / file_path
            if not full_path.exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.add_result(
                "Required files exist",
                False,
                f"Missing files: {', '.join(missing_files)}"
            )
            return False
        
        self.add_result(
            "Required files exist",
            True,
            f"All {len(REQUIRED_FILES)} required files found"
        )
        return True
    
    def check_dockerfile(self) -> bool:
        """Verify Dockerfile is valid"""
        dockerfile_path = BASE_DIR / "Dockerfile"
        
        if not dockerfile_path.exists():
            self.add_result(
                "Dockerfile valid",
                False,
                "Dockerfile not found"
            )
            return False
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        # Check for essential Dockerfile commands
        required_commands = ['FROM', 'COPY', 'RUN', 'CMD']
        missing_commands = [cmd for cmd in required_commands if cmd not in content]
        
        if missing_commands:
            self.add_result(
                "Dockerfile valid",
                False,
                f"Dockerfile missing commands: {', '.join(missing_commands)}"
            )
            return False
        
        # Check for port 7860 (Hugging Face Spaces default)
        if '7860' not in content:
            self.add_warning("Dockerfile does not expose port 7860 (Hugging Face Spaces default)")
        
        self.add_result(
            "Dockerfile valid",
            True,
            "Dockerfile contains all required commands"
        )
        return True
    
    def check_api_importable(self) -> bool:
        """Check if FastAPI app is importable"""
        try:
            sys.path.insert(0, str(BASE_DIR))
            from app.main import app
            
            # Check app has required endpoints
            routes = [route.path for route in app.routes]
            required_routes = ['/health', '/predict']
            missing_routes = [r for r in required_routes if r not in routes]
            
            if missing_routes:
                self.add_warning(f"API missing routes: {', '.join(missing_routes)}")
            
            self.add_result(
                "FastAPI app importable",
                True,
                f"App imported successfully with {len(routes)} routes"
            )
            return True
            
        except ImportError as e:
            self.add_result(
                "FastAPI app importable",
                False,
                f"Import failed: {e}"
            )
            return False
        except Exception as e:
            self.add_result(
                "FastAPI app importable",
                False,
                f"Error: {e}"
            )
            return False
    
    def check_model_architecture(self) -> bool:
        """Verify model architecture file is valid"""
        try:
            sys.path.insert(0, str(BASE_DIR))
            from step4_model_architecture import GeoTemporalFusionNet
            
            # Test model instantiation
            model = GeoTemporalFusionNet()
            params = sum(p.numel() for p in model.parameters())
            
            self.add_result(
                "Model architecture valid",
                True,
                f"GeoTemporalFusionNet instantiated ({params:,} parameters)"
            )
            return True
            
        except ImportError as e:
            self.add_result(
                "Model architecture valid",
                False,
                f"Import failed: {e}"
            )
            return False
        except Exception as e:
            self.add_result(
                "Model architecture valid",
                False,
                f"Error: {e}"
            )
            return False
    
    def run_all_checks(self) -> bool:
        """Run all deployment checks"""
        print("=" * 70)
        print("🔍 PRE-FLIGHT DEPLOYMENT CHECKLIST")
        print("=" * 70)
        print(f"   Project: {BASE_DIR.name}")
        print(f"   Path: {BASE_DIR}")
        print("=" * 70)
        
        checks = [
            self.check_requirements_file,
            self.check_model_file,
            self.check_required_files,
            self.check_dockerfile,
            self.check_hardcoded_paths,
            self.check_api_importable,
            self.check_model_architecture,
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.add_result(
                    check.__name__.replace('check_', '').replace('_', ' ').title(),
                    False,
                    f"Check failed with error: {e}"
                )
        
        # Print results
        print("\n📋 CHECK RESULTS:")
        print("-" * 70)
        
        passed_count = 0
        failed_count = 0
        
        for check_name, passed, message in self.results:
            status = "✅" if passed else "❌"
            print(f"   {status} {check_name}")
            print(f"      {message}")
            if passed:
                passed_count += 1
            else:
                failed_count += 1
        
        # Print warnings
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            print("-" * 70)
            for warning in self.warnings:
                print(f"   ⚠️ {warning}")
        
        # Summary
        print("\n" + "=" * 70)
        all_passed = failed_count == 0
        
        if all_passed:
            print("🎉 ALL CHECKS PASSED! Project is ready for deployment.")
        else:
            print(f"❌ {failed_count} CHECK(S) FAILED. Please fix issues before deployment.")
        
        print(f"   Passed: {passed_count}/{len(self.results)}")
        print("=" * 70)
        
        return all_passed


# ============================================================
# DEPLOYMENT INSTRUCTIONS
# ============================================================
def print_deployment_instructions():
    """Print deployment instructions"""
    print("""
================================================================================
📦 DEPLOYMENT INSTRUCTIONS
================================================================================

🚀 BACKEND DEPLOYMENT (Hugging Face Spaces)
─────────────────────────────────────────────
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Choose:
   - Space name: geotemporal-api
   - SDK: Docker
   - Hardware: CPU (Free) or GPU if available
4. Clone the Space repository:
   git clone https://huggingface.co/spaces/<username>/geotemporal-api
5. Copy these files to the repository:
   - Dockerfile
   - requirements.txt
   - app/main.py
   - models/simple_fire_model.pth
   - step4_model_architecture.py
   - step5_train.py
   - config.py
6. Push to Hugging Face:
   git add .
   git commit -m "Deploy GeoTemporalFusion API"
   git push
7. Wait for build to complete (~5 minutes)
8. Your API will be live at:
   https://<username>-geotemporal-api.hf.space

🌐 FRONTEND DEPLOYMENT (Vercel)
─────────────────────────────────────────────
1. Update frontend API URL to point to Hugging Face backend:
   const API_URL = "https://<username>-geotemporal-api.hf.space"
2. Go to https://vercel.com
3. Import your frontend repository
4. Deploy with one click
5. Your frontend will be live at:
   https://geotemporal-fusion.vercel.app

🧪 TESTING THE DEPLOYMENT
─────────────────────────────────────────────
# Health check
curl https://<username>-geotemporal-api.hf.space/health

# Model info
curl https://<username>-geotemporal-api.hf.space/model/info

================================================================================
""")


# ============================================================
# MAIN
# ============================================================
def main():
    """Main entry point"""
    checker = DeploymentChecker()
    success = checker.run_all_checks()
    
    if success:
        print_deployment_instructions()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
