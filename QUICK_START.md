# HerbaScan Backend - Quick Start Guide

## ✅ **What's Complete**

Your backend is **100% ready** to deploy! All code is written and tested.

```
✅ Python backend code complete
✅ Model files in place (mobilenetv2_rf.h5, labels.json)
✅ Docker configuration ready
✅ Railway deployment config ready
✅ API documentation complete
✅ Postman testing collection ready
```

---

## 🚀 **Deploy in 3 Steps** (15 minutes)

### **Step 1: Create GitHub Repo**

```bash
cd backend
git init
git add .
git commit -m "Initial commit: HerbaScan Grad-CAM API"

# Create repo on GitHub: https://github.com/new
# Name: herbascan-backend

git remote add origin https://github.com/YOUR_USERNAME/herbascan-backend.git
git push -u origin main
```

### **Step 2: Deploy to Railway**

1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your `herbascan-backend` repository
4. Railway will automatically detect Dockerfile and deploy
5. Wait ~5-10 minutes for first build

### **Step 3: Get Your URL**

1. In Railway dashboard, go to "Settings"
2. Under "Networking", click "Generate Domain"
3. Copy your URL: `https://YOUR-APP.up.railway.app`

---

## 🧪 **Test Your API**

### **Option A: Browser Test** (Quick)

Open in browser:
```
https://YOUR-APP.up.railway.app/health
```

Should see:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_classes": 40
}
```

### **Option B: Postman Test** (Thorough)

1. Open Postman
2. Import `HerbaScan_API.postman_collection.json`
3. Update `railway_url` variable with your Railway URL
4. Run "Health Check (Railway)" request
5. Run "Identify Plant (Railway)" request with a plant image

---

## 📱 **Use in Flutter App**

Once deployed, add this to your Flutter app:

```dart
// lib/core/services/online_gradcam_service.dart
static const String API_BASE_URL = 
  'https://YOUR-RAILWAY-URL.up.railway.app';
```

Then call:
```dart
final response = await http.post(
  Uri.parse('$API_BASE_URL/identify'),
  body: formData
);
```

---

## 🐛 **Troubleshooting**

### **Build Failed?**
- Check Railway logs in dashboard
- Verify model file is in git (if < 100MB)
- Check Dockerfile syntax

### **Model Not Loading?**
- Railway logs will show error
- Verify `mobilenetv2_rf.h5` and `labels.json` are in `models/` folder
- Check file permissions

### **API Slow?**
- First request always slower (cold start)
- Subsequent requests should be 2-4 seconds
- Consider Railway Pro for better performance

---

## 📊 **Expected Results**

| Endpoint | Response Time | Status |
|----------|--------------|--------|
| /health | < 100ms | ✅ instant |
| /test | < 100ms | ✅ instant |
| /identify | 2-5 seconds | ✅ includes ML inference |

---

## 📚 **More Help**

- **Detailed Deployment**: See `DEPLOYMENT.md`
- **Technical Docs**: See `README.md`
- **Phase Summary**: See `../PHASE1_COMPLETION_SUMMARY.md`

---

## ✨ **That's It!**

Once deployed, you have:
- ✅ Working Grad-CAM API
- ✅ 40 plant species identification
- ✅ Base64 encoded heatmap images
- ✅ Top-3 predictions with confidence scores
- ✅ Ready for Flutter integration

**Next**: Phase 2 - Offline CAM Preparation 🎯

