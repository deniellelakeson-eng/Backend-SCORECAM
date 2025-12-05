# 🚀 Quick Render Deployment Guide

## Prerequisites Checklist

- [ ] Render account created ([signup here](https://render.com))
- [ ] GitHub repository with backend code
- [ ] Model files ready (`MobileNetV2_model.keras` in `models/` directory)

## Quick Start (5 Steps)

### 1. Push Code to GitHub

```bash
cd backend
git add .
git commit -m "Prepare for Render deployment"
git push
```

### 2. Login to Render

- Go to [https://render.com](https://render.com)
- Sign in with GitHub
- Authorize Render access

### 3. Create Web Service

- Click **"New +"** → **"Web Service"**
- Select repository: `HerbaScan-Backend-SCORECAM`
- Render will auto-detect `render.yaml` ✅
- Click **"Create Web Service"**

### 4. Wait for Build

- First build: ~10-15 minutes (TensorFlow installation)
- Monitor logs in real-time
- Build will complete automatically

### 5. Get Your URL

- After deployment: `https://herbascan-backend.onrender.com`
- Test: `curl https://herbascan-backend.onrender.com/health`

## Configuration Files

✅ **Already configured:**
- `render.yaml` - Render service configuration
- `Dockerfile` - Docker build configuration
- `requirements.txt` - Python dependencies
- `runtime.txt` - Python version (3.10)
- `Procfile` - Process command (backup)

## Model Files

**If models are >100MB:**

**Option 1: Git LFS**
```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes models/MobileNetV2_model.keras
git commit -m "Add models via Git LFS"
git push
```

**Option 2: Render Disk**
- Upload via Render dashboard → Disk
- Update `MODEL_PATH` in code

## Update Flutter App

After deployment, update your Flutter app:

```dart
// lib/core/services/online_scorecam_service.dart
static const String serverUrl = 'https://herbascan-backend.onrender.com';
```

## Free Tier Notes

⚠️ **Important:**
- Services spin down after 15 min inactivity
- First request: ~30-60 seconds (cold start)
- Subsequent requests: ~2-4 seconds
- 750 hours/month free

💡 **Upgrade to Starter ($7/month)** for always-on service.

## Troubleshooting

**Build fails?**
- Check Render logs
- Verify Dockerfile works locally: `docker build -t test .`
- Ensure all files are committed

**Model not found?**
- Use Git LFS for models <100MB
- Use Render Disk for larger models
- Check file paths in code

**Service not responding?**
- Wait 30-60 seconds (cold start on free tier)
- Check service status in Render dashboard
- Verify health endpoint: `/health`

## Support

- Render Docs: https://render.com/docs
- Render Status: https://status.render.com
- Full README: See `README.md` for detailed instructions
