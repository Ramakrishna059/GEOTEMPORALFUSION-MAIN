# ================================================================================
# DEPLOY BACKEND TO HUGGING FACE SPACES
# ================================================================================
# This script helps deploy your FastAPI backend to Hugging Face Spaces
#
# PREREQUISITES:
# 1. Create a Hugging Face account at https://huggingface.co
# 2. Create a new Space:
#    - Name: geotemporal-wildfire-api
#    - SDK: Docker (IMPORTANT!)
#    - Hardware: CPU basic (free)
# 
# USAGE:
# Replace YOUR_HF_USERNAME below with your actual Hugging Face username
# Then run: .\deploy_to_huggingface.ps1
# ================================================================================

param(
    [Parameter(Mandatory=$false)]
    [string]$HFUsername = "YOUR_HF_USERNAME"
)

Write-Host "`n🔥 GeoTemporalFusion - Hugging Face Deployment" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

if ($HFUsername -eq "YOUR_HF_USERNAME") {
    Write-Host "`n❌ ERROR: Please provide your Hugging Face username!" -ForegroundColor Red
    Write-Host "`nUsage:" -ForegroundColor Yellow
    Write-Host "  .\deploy_to_huggingface.ps1 -HFUsername your_username" -ForegroundColor White
    Write-Host "`nOR edit this file and replace YOUR_HF_USERNAME with your actual username" -ForegroundColor Yellow
    exit 1
}

$spaceName = "geotemporal-wildfire-api"
$hfRepoUrl = "https://huggingface.co/spaces/$HFUsername/$spaceName"
$tempDir = Join-Path $env:TEMP "hf-space-temp"

Write-Host "`n📦 Deployment Configuration:" -ForegroundColor Green
Write-Host "   Username: $HFUsername"
Write-Host "   Space: $spaceName"
Write-Host "   URL: $hfRepoUrl"

Write-Host "`n⚠️  IMPORTANT: Make sure you have:" -ForegroundColor Yellow
Write-Host "   1. Created the Space on Hugging Face" -ForegroundColor White
Write-Host "   2. Selected 'Docker' as SDK" -ForegroundColor White
Write-Host "   3. Logged in with: huggingface-cli login" -ForegroundColor White

$response = Read-Host "`nHave you completed the above steps? (y/n)"
if ($response -ne "y") {
    Write-Host "`n📋 Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Go to: https://huggingface.co/spaces" -ForegroundColor White
    Write-Host "   2. Click 'Create new Space'" -ForegroundColor White
    Write-Host "   3. Name: $spaceName" -ForegroundColor White
    Write-Host "   4. SDK: Docker (IMPORTANT!)" -ForegroundColor White
    Write-Host "   5. Hardware: CPU basic" -ForegroundColor White
    Write-Host "   6. Run: pip install huggingface_hub" -ForegroundColor White
    Write-Host "   7. Run: huggingface-cli login" -ForegroundColor White
    Write-Host "   8. Run this script again" -ForegroundColor White
    exit 0
}

Write-Host "`n🚀 Starting deployment..." -ForegroundColor Green

# Clone or update HF space
if (Test-Path $tempDir) {
    Write-Host "   Cleaning temp directory..." -ForegroundColor Gray
    Remove-Item -Path $tempDir -Recurse -Force
}

Write-Host "   Cloning Hugging Face Space..." -ForegroundColor Gray
git clone $hfRepoUrl $tempDir

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Failed to clone Space. Make sure:" -ForegroundColor Red
    Write-Host "   1. The Space exists: $hfRepoUrl" -ForegroundColor White
    Write-Host "   2. You're logged in: huggingface-cli login" -ForegroundColor White
    exit 1
}

# Copy necessary files
Write-Host "   Copying backend files..." -ForegroundColor Gray
$currentDir = $PSScriptRoot

Copy-Item -Path "$currentDir\app" -Destination $tempDir -Recurse -Force
Copy-Item -Path "$currentDir\models" -Destination $tempDir -Recurse -Force
Copy-Item -Path "$currentDir\Dockerfile" -Destination $tempDir -Force
Copy-Item -Path "$currentDir\requirements.txt" -Destination $tempDir -Force
Copy-Item -Path "$currentDir\config.py" -Destination $tempDir -Force
Copy-Item -Path "$currentDir\step4_model_architecture.py" -Destination $tempDir -Force

# Create README for Hugging Face
$readmeContent = @"
---
title: GeoTemporalFusion Wildfire Prediction API
emoji: 🔥
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
license: mit
---

# GeoTemporalFusion - AI Wildfire Prediction System

🔥 **Live API Backend** for wildfire risk prediction using deep learning.

## 🌐 API Endpoints

- **GET /**: API Documentation (Swagger UI)
- **GET /health**: Health check
- **GET /model/info**: Model information
- **POST /predict**: Fire risk prediction

## 🚀 Usage

Visit this Space's URL to access the interactive API documentation (Swagger UI).

## 🛠️ Tech Stack

- FastAPI
- PyTorch
- Docker
- Python 3.10

## 📊 Model

- Architecture: GeoTemporal Fusion Network
- Input: 128x128 satellite images + 24h weather data
- Output: Fire risk probability (0-1)

Built by GeoTemporalFusion Team
"@

Set-Content -Path "$tempDir\README.md" -Value $readmeContent

# Commit and push
Write-Host "   Committing changes..." -ForegroundColor Gray
Set-Location $tempDir
git add .
git commit -m "Deploy FastAPI backend to Hugging Face Spaces"

Write-Host "   Pushing to Hugging Face..." -ForegroundColor Gray
git push

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "`n🎉 Your FastAPI backend is now deploying on Hugging Face!" -ForegroundColor Cyan
    Write-Host "`n📍 Access your API at:" -ForegroundColor Yellow
    Write-Host "   $hfRepoUrl" -ForegroundColor White
    Write-Host "`n⏱️  Build time: 3-5 minutes" -ForegroundColor Gray
    Write-Host "`n🔗 Once deployed, you'll see the Swagger UI interface" -ForegroundColor Green
    Write-Host "   (just like when you run uvicorn locally!)" -ForegroundColor Green
} else {
    Write-Host "`n❌ Push failed!" -ForegroundColor Red
    Write-Host "   Make sure you're authenticated with huggingface-cli" -ForegroundColor Yellow
}

Set-Location $currentDir
Write-Host "`n✨ Done!" -ForegroundColor Cyan
