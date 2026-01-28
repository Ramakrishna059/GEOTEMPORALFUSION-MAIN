# 🚀 VERCEL DEPLOYMENT GUIDE - FULL-STACK APP
# ============================================================================

## ✅ WHAT'S BEEN SET UP:

### 1. API Serverless Function
- Created: api/index.py (wraps FastAPI app)
- All backend routes now available under /api/*

### 2. Configuration Files
- vercel.json: Configured for Python API + Static frontend
- .vercelignore: Optimized to include API code, exclude heavy files

### 3. Frontend Updated
- index.html now calls /api/* endpoints
- Auto-tests API connection on page load

## 📋 DEPLOYMENT STEPS:

### Step 1: Login to Vercel
```powershell
vercel login
```
Choose your preferred login method (GitHub, GitLab, Email, etc.)

### Step 2: Deploy (Preview)
```powershell
vercel
```
This will:
- Link/create project
- Deploy to preview URL
- Test everything works

### Step 3: Deploy to Production
```powershell
vercel --prod
```
This deploys to your production domain

## 🔧 DEPLOYMENT PROMPTS:

When you run `vercel`, answer:
```
? Set up and deploy? [Y/n] Y
? Which scope? [Select your account]
? Link to existing project? [y/N] N
? What's your project's name? geotemporal-fusion
? In which directory is your code located? ./
```

## 📊 EXPECTED RESULTS:

✅ Frontend: Your-Project.vercel.app
✅ API Health: Your-Project.vercel.app/api/health
✅ API Docs: Your-Project.vercel.app/api
✅ Model Info: Your-Project.vercel.app/api/model/info

## 🐛 TROUBLESHOOTING:

### Issue: API not responding
**Solution**: Vercel Python functions have cold starts (~5-10 seconds first load)
Wait and refresh the page

### Issue: Module not found
**Solution**: Check requirements.txt includes all dependencies

### Issue: Model file too large
**Solution**: Vercel has 250MB limit. The simple_fire_model.pth should be OK.
If too large, consider using .vercelignore to exclude it and load from external URL

### Issue: Memory limit exceeded
**Solution**: Vercel free tier has 1GB memory limit
Consider optimizing model size or upgrading plan

## 📦 FILE STRUCTURE FOR VERCEL:

```
GeoTemporalFusion/
├── api/
│   └── index.py              # Serverless function entry
├── app/
│   ├── __init__.py
│   └── main.py               # FastAPI app
├── models/
│   └── simple_fire_model.pth # Model weights
├── data/
│   └── processed/weather/    # Small data files
├── index.html                # Frontend
├── vercel.json               # Vercel config
├── .vercelignore             # Exclude files
└── requirements.txt          # Python dependencies

```

## 🎯 NEXT STEPS AFTER DEPLOYMENT:

1. Test all API endpoints
2. Check console for errors
3. Monitor cold start times
4. Set up custom domain (optional)
5. Configure environment variables if needed

## 📝 USEFUL COMMANDS:

```powershell
# View deployment logs
vercel logs

# List all deployments
vercel ls

# Open project in browser
vercel open

# View domains
vercel domains ls

# Remove deployment
vercel rm [deployment-url]

# Check project info
vercel inspect [deployment-url]
```

## 🔐 ENVIRONMENT VARIABLES (If Needed):

If your app needs environment variables:
```powershell
# Add via CLI
vercel env add VARIABLE_NAME

# Or add via Vercel Dashboard:
# Project Settings → Environment Variables
```

## ✨ THAT'S IT!

Your full-stack application is ready to deploy!
Run `vercel` to start the deployment process.
