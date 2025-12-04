# HerbaScan Backend - Quick Start Guide

**Last Updated**: December 2025  
**Backend Version**: 0.7.6  
**Flutter App Version**: v0.7.6

## ✅ **What's Complete**

Your backend is **100% ready** to deploy! All code is written and tested.

```
✅ Python backend code complete
✅ Dual model architecture support (MobileNetV2_model.keras, herbascan_model.keras)
✅ Docker configuration ready
✅ Railway deployment config ready (Procfile added)
✅ API documentation complete
✅ Postman testing collection ready
```

---

## 📋 **Prerequisites**

Before deploying, make sure you have:

- ✅ Railway account ([signup here](https://railway.app))
- ✅ GitHub account
- ✅ Model files in `backend/models/` directory:
  - `MobileNetV2_model.keras` (MobileNetV2 architecture model - `.keras` format)
  - `herbascan_model.keras` (Custom HerbaScan architecture model - `.keras` format)
  - `labels.json` (plant class labels - optional, for backward compatibility)
  
  **Note:** At least one `.keras` model must be present. The backend will automatically use both models and select the result with highest confidence.

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

**For detailed Postman instructions:**
- See `/backend/README.md` → "🧪 Testing with Postman" section
- Includes instructions for Postman desktop app, VS Code (REST Client, Thunder Client), and other IDEs
- Includes troubleshooting guide for common Postman issues

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

## 🔄 **Updating Models**

### Quick Model Update Process

When you have a new trained model:

1. **Replace model files (`.keras` format):**
   ```bash
   # Backup old models (optional)
   cp backend/models/MobileNetV2_model.keras backend/models/MobileNetV2_model.keras.backup
   cp backend/models/herbascan_model.keras backend/models/herbascan_model.keras.backup
   
   # Copy new models
   cp /path/to/your/new_mobilenetv2_model.keras backend/models/MobileNetV2_model.keras
   cp /path/to/your/new_herbascan_model.keras backend/models/herbascan_model.keras
   ```

2. **Update labels (if classes changed):**
   ```bash
   # Backend labels (index:name format)
   # Edit backend/models/labels.json
   
   # Frontend labels (name:index format)
   # Edit assets/models/class_indices.json
   ```

3. **Test locally:**
   ```bash
   python main.py
   curl http://localhost:8000/health
   ```

4. **Regenerate Flutter assets (Phase 2):**
   ```bash
   cd backend
   
   # Option A: Update scripts to use .keras (recommended)
   # Edit extract_cam_weights.py: Change MODEL_PATH to models/MobileNetV2_model.keras
   # Edit create_multi_output_tflite.py: Change MODEL_PATH to models/MobileNetV2_model.keras
   
   # Option B: Convert .keras to .h5 temporarily
   python -c "import tensorflow as tf; model = tf.keras.models.load_model('models/MobileNetV2_model.keras'); model.save('models/mobilenetv2_rf.h5', save_format='h5')"
   
   # Run extraction scripts
   python extract_cam_weights.py
   python create_multi_output_tflite.py
   
   # Copy to Flutter assets
   cp models/cam_weights.json ../assets/models/
   cp models/mobilenetv2_multi_output.tflite ../assets/models/
   # Note: Frontend uses class_indices.json, not labels.json
   ```

5. **Update Flutter pubspec.yaml:**
   ```yaml
   flutter:
     assets:
       - assets/models/
       # This includes all files in assets/models/:
       # - MobileNetV2_model.tflite
       # - herbascan_model.tflite
       # - class_indices.json
       # - cam_weights.json
   ```

6. **Redeploy to Railway:**
   ```bash
   git add backend/models/MobileNetV2_model.keras backend/models/herbascan_model.keras
   git commit -m "Update models to v2.0"
   git push
   # Railway will automatically redeploy
   ```

**For detailed instructions, see `/backend/README.md` → "🔄 Updating Models" section.**

---

## 🐛 **Troubleshooting**

### **Build Failed?**
- Check Railway logs in dashboard
- Verify model file is in git (if < 100MB) or uploaded to Railway volumes
- Check Dockerfile syntax
- Ensure TensorFlow dependencies are correct

### **Model Not Loading?**
- Railway logs will show error
- Verify at least one `.keras` model exists: `MobileNetV2_model.keras` or `herbascan_model.keras`
- Check file permissions
- For large models (>100MB), use Railway volumes or Git LFS
- Backend supports both models - will use whichever is available

### **API Slow?**
- First request always slower (cold start: 10-30 seconds)
- Subsequent requests should be 2-4 seconds
- Consider Railway Pro for better performance
- Check Railway logs for memory issues

### **File Upload Errors?**
- Use `curl` instead of Postman for Railway testing:
  ```bash
  curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
    -F "file=@path/to/your/image.jpg"
  ```
- Postman works fine for local testing (`http://localhost:8000`)
- Flutter app works perfectly with Railway (uses `http` package)

### **TFLite Conversion Fails?**
- Check TensorFlow version: `pip show tensorflow`
- Recommended: `pip install tensorflow==2.15.0`
- See `TENSORFLOW_COMPATIBILITY_FIX.md` for details

**For more troubleshooting, see `/backend/README.md` → "🐛 Troubleshooting" section.**

---

## 📊 **Expected Results**

| Endpoint | Response Time | Status |
|----------|--------------|--------|
| /health | < 100ms | ✅ instant |
| /test | < 100ms | ✅ instant |
| /identify | 2-5 seconds | ✅ includes ML inference |

---

## 📚 **More Help**

### Key Sections in README.md

- **🔄 Updating Models**: How to update backend and Flutter models
- **🚀 Deployment to Railway**: Detailed deployment instructions
- **🐛 Troubleshooting**: Common issues and solutions
- **Phase 2: Model Extraction & Conversion**: Preparing models for Flutter

---

## ✨ **That's It!**

Once deployed, you have:
- ✅ Working Grad-CAM API
- ✅ 40 plant species identification
- ✅ Base64 encoded heatmap images
- ✅ Top-3 predictions with confidence scores
- ✅ Ready for Flutter integration
- ✅ Integrated with Hybrid XAI Explanation System (v0.5.8)

### Next Steps

1. **Test your API:**
   - Health check: `curl https://YOUR-RAILWAY-URL.railway.app/health`
   - Test identification with sample images

2. **Update Flutter app:**
   - Update `online_gradcam_service.dart` with your Railway URL
   - Test connection from Flutter app

3. **Phase 2 - Offline CAM Preparation:**
   ```bash
   cd backend
   
   # Option A: Update scripts to use .keras (recommended)
   # Edit extract_cam_weights.py: Change MODEL_PATH to models/MobileNetV2_model.keras
   # Edit create_multi_output_tflite.py: Change MODEL_PATH to models/MobileNetV2_model.keras
   
   # Option B: Convert .keras to .h5 temporarily
   python -c "import tensorflow as tf; model = tf.keras.models.load_model('models/MobileNetV2_model.keras'); model.save('models/mobilenetv2_rf.h5', save_format='h5')"
   
   # Extract CAM weights
   python extract_cam_weights.py
   # Create multi-output TFLite model
   python create_multi_output_tflite.py
   # Copy to Flutter assets
   cp models/cam_weights.json ../assets/models/
   cp models/mobilenetv2_multi_output.tflite ../assets/models/
   ```
   - See `/backend/README.md` → "Phase 2: Model Extraction & Conversion" for detailed steps
   - **Note:** Frontend uses `assets/models/class_indices.json` (name:index format), not `labels.json`

4. **Monitor Performance:**
   - Check Railway logs for errors
   - Monitor response times
   - Consider upgrading to Railway Pro for production

---

## 🎯 **Quick Reference**

### Local Development
```bash
# Run server
python main.py

# Test health
curl http://localhost:8000/health

# Test identification
curl -X POST http://localhost:8000/identify -F "file=@image.jpg"
```

### Railway Deployment
```bash
# Deploy (auto-deploys on git push)
git push

# Check logs
railway logs

# Get URL
# Railway dashboard → Settings → Networking → Generate Domain
```

### Model Updates
```bash
# Update scripts to use .keras (or convert .keras to .h5 temporarily)
# Edit extract_cam_weights.py: MODEL_PATH = "models/MobileNetV2_model.keras"
# Edit create_multi_output_tflite.py: MODEL_PATH = "models/MobileNetV2_model.keras"

# Extract CAM weights
python extract_cam_weights.py

# Create TFLite model
python create_multi_output_tflite.py

# Copy to Flutter assets
cp models/cam_weights.json ../assets/models/
cp models/mobilenetv2_multi_output.tflite ../assets/models/
# Note: Frontend uses class_indices.json (name:index), not labels.json (index:name)
```

---

**Ready to deploy? Follow the steps above and your API will be live in 15 minutes!** 🚀

