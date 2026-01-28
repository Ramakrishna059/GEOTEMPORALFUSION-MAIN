# 🚀 RENDER.COM DEPLOYMENT GUIDE - COMPLETE SETUP
## Full-Stack GeoTemporalFusion Application
================================================================================

## ✅ WHAT'S BEEN CONFIGURED:

### 1. Backend API Updates ✅
- All API routes now use `/api/*` prefix
- Static frontend served at `/` (root)
- API docs moved to `/api/docs`
- CORS configured for all origins

### 2. Configuration Files ✅
- **render.yaml**: Complete Render service configuration
- **start.sh**: Production startup script
- Routes configured for frontend + backend

### 3. File Structure ✅
```
GeoTemporalFusion/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI app (serves API + frontend)
├── models/
│   └── simple_fire_model.pth # Trained model
├── index.html               # Frontend page
├── render.yaml              # Render configuration
├── start.sh                 # Startup script
└── requirements.txt         # Python dependencies
```

================================================================================
## 📋 DEPLOYMENT STEPS:
================================================================================

### STEP 1: Commit and Push to GitHub

```powershell
# Add all files
git add .

# Commit changes
git commit -m "Configure for Render deployment"

# Push to GitHub
git push origin main
```

### STEP 2: Create Render Account

1. Go to: **https://render.com**
2. Click **"Get Started for Free"**
3. Sign up with GitHub (easiest option)
4. Authorize Render to access your GitHub repositories

### STEP 3: Create New Web Service

1. **Dashboard**: Click **"New +"** → **"Web Service"**
2. **Connect Repository**: Select `Ramakrishna059/GEOTEMPORALFUSION-MAIN`
3. **Configure Service**:

   ```
   Name: geotemporal-fusion
   Region: Oregon (US West)
   Branch: main
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

4. **Instance Type**: Select **"Free"**
   - 512 MB RAM
   - Shared CPU
   - Sleeps after 15 mins of inactivity
   - Perfect for demos!

5. **Environment Variables** (Optional):
   ```
   PYTHON_VERSION = 3.11.0
   ```

6. **Click**: **"Create Web Service"**

### STEP 4: Wait for Deployment

- Render will:
  1. Clone your repository ✅
  2. Install dependencies (2-3 minutes) ✅
  3. Build the service ✅
  4. Start the application ✅

- You'll see live logs in the dashboard
- Wait for: **"Your service is live 🎉"**

### STEP 5: Test Your Deployment

Your app will be available at:
```
https://geotemporal-fusion.onrender.com
```

**Test these endpoints:**

1. **Homepage** (Frontend):
   ```
   https://geotemporal-fusion.onrender.com/
   ```

2. **Health Check**:
   ```
   https://geotemporal-fusion.onrender.com/api/health
   ```

3. **API Documentation**:
   ```
   https://geotemporal-fusion.onrender.com/api/docs
   ```

4. **Model Info**:
   ```
   https://geotemporal-fusion.onrender.com/api/model/info
   ```

================================================================================
## 🔧 TESTING LOCALLY FIRST (Optional)
================================================================================

Test before deploying:

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Test in browser
# Frontend: http://localhost:8000/
# API Docs: http://localhost:8000/api/docs
# Health:   http://localhost:8000/api/health
```

================================================================================
## 📊 EXPECTED RESULTS AFTER DEPLOYMENT:
================================================================================

✅ **Frontend**: Beautiful landing page with fire emoji
✅ **API Health**: Returns JSON with "status": "healthy"
✅ **API Docs**: Interactive Swagger UI at `/api/docs`
✅ **Model Info**: Shows 12M parameters, 100% accuracy
✅ **Predictions**: Can upload images for fire risk prediction

================================================================================
## 🎯 RENDER FEATURES:
================================================================================

### Free Tier Includes:
- ✅ 750 hours/month of runtime (plenty for demos)
- ✅ Automatic HTTPS
- ✅ Custom domains
- ✅ Auto-deploy from GitHub
- ✅ Environment variables
- ✅ Live logs and metrics

### How it Works:
```
GitHub Push
     ↓
Render Auto-Detects Changes
     ↓
Rebuilds & Redeploys (2-3 min)
     ↓
Your App is Live! 🎉
```

================================================================================
## 🐛 TROUBLESHOOTING:
================================================================================

### Issue: Build Failed

**Solution**: Check logs in Render dashboard
- Look for: `pip install` errors
- Ensure requirements.txt is correct
- Python version compatibility

### Issue: Service Won't Start

**Solution**: Check Start Command
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Issue: 404 on Frontend

**Solution**: Ensure:
- index.html exists in root directory
- app/main.py has `@app.get("/")` route

### Issue: API Endpoints Not Working

**Solution**: Check routes have `/api` prefix
- Use `/api/health` not `/health`
- Update frontend fetch() calls

### Issue: Model Not Loading

**Check**:
1. `models/simple_fire_model.pth` exists
2. File size < 500MB (Render limit)
3. Path is correct in app/main.py

### Issue: Free Tier Sleeps

**Behavior**: App sleeps after 15 mins inactivity
- First request takes ~30 seconds (cold start)
- Subsequent requests are instant
- This is normal for free tier

**Solution**: Upgrade to paid tier ($7/mo) for always-on

================================================================================
## 🚀 ALTERNATIVE: DEPLOY VIA RENDER.YAML
================================================================================

Render can auto-configure from render.yaml:

1. **In Render Dashboard**:
   - Click **"New +"** → **"Blueprint"**
   - Select your GitHub repository
   - Render reads `render.yaml` automatically
   - Click **"Apply"**

2. **Done!** Service created with all settings

================================================================================
## 📝 USEFUL RENDER COMMANDS & TIPS:
================================================================================

### View Logs:
- Go to: Dashboard → Your Service → Logs
- Live tail of all output

### Manual Deploy:
- Dashboard → Your Service → Manual Deploy → Deploy Latest Commit

### Environment Variables:
- Dashboard → Your Service → Environment
- Add key-value pairs
- Changes trigger redeploy

### Custom Domain:
- Dashboard → Your Service → Settings → Custom Domains
- Add CNAME record: `your-domain.com` → `your-app.onrender.com`

### Suspend Service:
- Dashboard → Your Service → Settings → Suspend Service
- Stops billing (if paid tier)

================================================================================
## 💰 RENDER PRICING:
================================================================================

| Plan | Price | RAM | CPU | Features |
|------|-------|-----|-----|----------|
| Free | $0 | 512MB | Shared | Sleeps after 15min |
| Starter | $7/mo | 512MB | Shared | Always-on |
| Standard | $25/mo | 2GB | 1 CPU | Better performance |

**Recommendation**: Start with Free, upgrade if needed

================================================================================
## ✨ BENEFITS OF RENDER OVER VERCEL:
================================================================================

✅ **Native Python Support** - No serverless limitations
✅ **Long-Running Processes** - Can run ML models
✅ **More Memory** - 512MB vs Vercel's 1GB serverless limit
✅ **Persistent Connections** - Better for FastAPI
✅ **No File Count Limits** - Deploy all files
✅ **Full Log Access** - See everything

================================================================================
## 🎉 WHAT YOU GET:
================================================================================

After deployment, you'll have:

1. **Full-Stack App** running at your Render URL
2. **Automatic HTTPS** with valid SSL certificate
3. **Auto-deploys** from GitHub pushes
4. **Frontend** served at root `/`
5. **API** available at `/api/*`
6. **Interactive Docs** at `/api/docs`
7. **Health Monitoring** built-in
8. **Free hosting** (within limits)

================================================================================
## 📌 QUICK START SUMMARY:
================================================================================

```powershell
# 1. Commit and push
git add .
git commit -m "Deploy to Render"
git push

# 2. Go to Render.com
# 3. New Web Service → Connect GitHub repo
# 4. Configure:
#    - Build: pip install -r requirements.txt  
#    - Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
# 5. Deploy (wait 3 minutes)
# 6. Visit your URL!
```

================================================================================
## 🔗 HELPFUL LINKS:
================================================================================

- Render Dashboard: https://dashboard.render.com
- Render Docs: https://render.com/docs
- Python on Render: https://render.com/docs/deploy-fastapi
- Your GitHub Repo: https://github.com/Ramakrishna059/GEOTEMPORALFUSION-MAIN

================================================================================

That's it! Your full-stack GeoTemporalFusion app will be live on Render! 🔥

Ready to deploy? Run:
```powershell
git add . && git commit -m "Deploy to Render" && git push
```

Then create the web service on Render.com! 🚀
