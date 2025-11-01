# HerbaScan Backend - Railway Deployment Guide

## 🚀 Quick Deploy to Railway (5 minutes)

### **Prerequisites**
- ✅ Railway account ([signup here](https://railway.app))
- ✅ GitHub account
- ✅ Model files in `backend/models/` directory

---

## 📝 **Step-by-Step Deployment**

### **Step 1: Create GitHub Repository** (If not already done)

You have two options:

#### **Option A: Create Separate Backend Repo (Recommended)**

```bash
# Navigate to backend folder
cd backend

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: HerbaScan Grad-CAM API"

# Create repo on GitHub (via web interface):
# 1. Go to https://github.com/new
# 2. Name: "herbascan-backend"
# 3. Description: "HerbaScan Grad-CAM API - True gradient-based plant identification"
# 4. Public or Private (your choice)
# 5. Click "Create repository"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/herbascan-backend.git
git branch -M main
git push -u origin main
```

#### **Option B: Use Existing Repo's Backend Folder**

```bash
# From project root (herbascan/)
git add backend/
git commit -m "Add backend API for Grad-CAM computation"
git push
```

---

### **Step 2: Deploy to Railway**

#### **2.1: Login to Railway**
1. Go to [https://railway.app](https://railway.app)
2. Click **"Login"** (use GitHub account)
3. Authorize Railway to access your GitHub

#### **2.2: Create New Project**
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your repository:
   - If separate repo: Select `herbascan-backend`
   - If same repo: Select `herbascan` (you'll specify folder next)

#### **2.3: Configure Build Settings**

If using same repo with backend folder:
1. Click **"Settings"**
2. Under **"Build"**, set:
   - **Root Directory**: `backend`
   - **Builder**: `Dockerfile`

If using separate backend repo:
- Railway will auto-detect the Dockerfile ✅

#### **2.4: Add Environment Variables (Optional)**

Click **"Variables"** tab and add:
```
PORT=8000
```

Railway automatically provides `$PORT`, but you can set a default.

#### **2.5: Deploy**

1. Railway will automatically start building
2. Wait for build to complete (~5-10 minutes, first time is slower due to TensorFlow)
3. Once deployed, you'll see a ✅ green checkmark

---

### **Step 3: Get Your API URL**

1. In Railway dashboard, click **"Settings"**
2. Under **"Networking"**, click **"Generate Domain"**
3. Railway will create a public URL like:
   ```
   https://YOUR-RAILWAY-URL.railway.app/
   ```
4. **Copy this URL** - you'll use it in Flutter app!

---

### **Step 4: Test Your Deployed API**

#### **Test 1: Health Check**

Open in browser or use curl:
```bash
curl https://YOUR-RAILWAY-URL.railway.app/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "labels_loaded": true,
  "num_classes": 40
}
```

#### **Test 2: Root Endpoint**

```bash
curl https://YOUR-RAILWAY-URL.railway.app/
```

#### **Test 3: Plant Identification (using Postman)**

**IMPORTANT - Follow these steps exactly:**

1. **Open Postman**
2. **Create New Request:**
   - Method: `POST`
   - URL: `https://YOUR-RAILWAY-URL.railway.app/identify`
3. **Set Body:**
   - Select the **"Body"** tab
   - Choose **"form-data"** (NOT "raw" or "binary")
   - Add a new field with:
     - **Key:** `file` (exactly this name)
     - Click the dropdown on the right, **change "Text" to "File"**
     - Click **"Select Files"** and choose an **ACTUAL image file** from your computer (.jpg or .png)
     - ⚠️ **DO NOT** use the placeholder "path/to/plant/image.jpg" - select a real file!
4. **Click Send**
5. **Check Response:**
   - Should see `plant_name`, `confidence`, `gradcam_image` (base64)
   - Processing time should be under 5 seconds

**Alternative: Test with curl (more reliable)**
```bash
curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
  -F "file=@path/to/your/image.jpg"
```

---

## 🐛 **Troubleshooting**

### **Build Failed**

**Check Railway logs:**
1. Go to Railway dashboard
2. Click **"Deployments"**
3. Click the failed deployment
4. Check logs for errors

**Common issues:**
- **Model file not found**: Make sure `mobilenetv2_rf.h5` is committed to git
- **Out of memory**: Railway free tier has 512MB RAM limit. Consider upgrading or optimizing model.
- **TensorFlow installation failed**: Check Dockerfile dependencies

### **File Upload Errors**

**Error: "Image bytes are empty"**
- ❌ **Problem**: You didn't select an actual file in Postman
- ✅ **Solution**: 
  1. In Postman form-data, make sure the dropdown next to key "file" says "File" (not "Text")
  2. Click "Select Files" and choose a real .jpg or .png from your computer
  3. Don't use placeholder text like "path/to/image.jpg"

**Error: "Misordered multipart fields; files should follow 'map'"**
- ❌ **Problem**: Railway proxy has issues with Postman's multipart format
- ✅ **Solutions**:
  1. Use `curl` instead of Postman:
     ```bash
     curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
       -F "file=@path/to/your/image.jpg"
     ```
  2. ⚠️ **Note**: This Postman issue does NOT affect your Flutter app! The Flutter app uses the `http` package which works perfectly with Railway
  3. For testing: Use Postman against `http://localhost:8000` OR use curl for Railway

**Error: "cannot identify image file"**
- ❌ **Problem**: File is corrupted or not a valid image
- ✅ **Solution**: 
  1. Try a different image file
  2. Make sure it's .jpg, .png, .bmp, .webp, or .gif format
  3. Check file isn't corrupted

**Check Railway logs for detailed error:**
- Go to Railway dashboard → Deployments → Latest deployment → Logs
- Look for "DEBUG:" lines to see what was received

### **Model Not Loading**

If `/health` shows `"model_loaded": false`:

1. **Check Railway logs** for error messages
2. **Verify model file size**:
   ```bash
   # Check model file size
   ls -lh backend/models/mobilenetv2_rf.h5
   ```
   - If > 100MB, git may not have uploaded it
   - Use Git LFS or Railway volumes

3. **Add model via Railway volumes:**
   - Upload model directly to Railway
   - Update `MODEL_PATH` to point to volume

### **Slow Response Times**

- First request is always slower (cold start)
- Subsequent requests should be faster
- Consider using Railway Pro for better performance

### **CORS Issues**

If Flutter app can't connect:
1. Check Railway URL is correct
2. Verify CORS is enabled in `main.py` (it is by default)
3. Try accessing API in browser first

---

## 📊 **Expected Performance**

| Metric | Target | Typical |
|--------|--------|---------|
| Build Time | - | 5-10 min (first time) |
| Cold Start | - | 10-30 sec |
| Inference Time | <5s | 2-4s |
| Memory Usage | <512MB | 300-450MB |
| Response Size | - | 50-200KB (with base64 image) |

---

## 💰 **Railway Pricing**

- **Free Tier**: $5 credit/month, 500 hours execution
- **Good for**: Development & testing
- **Upgrade if**: High traffic or need more resources

For HerbaScan:
- Estimated cost: ~$5-10/month (hobby usage)
- Consider upgrading for thesis demos

---

## 🔄 **Continuous Deployment**

Railway auto-deploys on every git push:

```bash
# Make changes to code
# Edit backend/main.py or backend/utils/gradcam.py

# Commit and push
git add .
git commit -m "Update Grad-CAM layer configuration"
git push

# Railway automatically rebuilds and redeploys ✅
```

---

## 📝 **Save Your API URL**

Once deployed, save this for Flutter integration:

```dart
// Flutter: lib/core/services/online_gradcam_service.dart
static const String API_BASE_URL = 
  'https://YOUR-RAILWAY-URL.railway.app';
```

---

## 🎉 **Next Steps After Deployment**

Once your API is live:

1. ✅ Test `/health` endpoint
2. ✅ Test `/identify` with sample plant images
3. ✅ Verify Grad-CAM quality
4. ✅ Measure response times
5. ✅ Update Flutter app with API URL

Then proceed to **Phase 2: Offline CAM Preparation**

---

## 🆘 **Need Help?**

- Railway Docs: https://docs.railway.app/
- Railway Discord: https://discord.gg/railway
- FastAPI Docs: https://fastapi.tiangolo.com/deployment/

**Common Railway Commands:**

```bash
# Install Railway CLI (optional)
npm install -g @railway/cli

# Login
railway login

# Check logs
railway logs

# Run commands in deployment
railway run python test_model_info.py
```

---

**Ready to deploy?** Follow the steps above and let me know once your API is live! 🚀

