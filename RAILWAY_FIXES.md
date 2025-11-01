# Railway Deployment Fixes

## 🔧 **Build Error Fixes Applied**

I've fixed several issues that could cause Railway build failures:

### **Issue 1: TensorFlow Version** ✅ FIXED
**Problem**: `tensorflow==2.20.0` doesn't exist (latest stable is 2.15.0)
**Fix**: Changed to `tensorflow==2.15.0` (stable version)

### **Issue 2: NumPy Compatibility** ✅ FIXED
**Problem**: `numpy>=2.0.0` may not be compatible with TensorFlow 2.15.0
**Fix**: Changed to `numpy==1.24.3` (compatible version)

### **Issue 3: Missing OpenCV** ✅ FIXED
**Problem**: `cv2` is used in `gradcam.py` but not in requirements.txt
**Fix**: Added `opencv-python-headless==4.8.1.78` (headless version, smaller)

### **Issue 4: Missing Requests** ✅ FIXED
**Problem**: Healthcheck uses `requests` but it's not in requirements
**Fix**: Added `requests==2.31.0` and updated healthcheck to use `curl`

### **Issue 5: Missing System Libraries** ✅ FIXED
**Problem**: OpenCV needs additional system libraries
**Fix**: Added `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1` to Dockerfile

---

## 📦 **Model File Issue (If Build Still Fails)**

### **Problem: Model File Too Large for Git**

If your `mobilenetv2_rf.h5` file is **> 50MB**, Git might not upload it, causing:
- Build succeeds but model not found at runtime
- `/health` shows `"model_loaded": false`

### **Solution Options:**

#### **Option A: Use Railway Volumes** (Recommended)
1. In Railway dashboard, go to your project
2. Click "Volumes" tab
3. Create new volume: `models`
4. Upload `mobilenetv2_rf.h5` to volume
5. Update `main.py` to use volume path:
   ```python
   MODEL_PATH = Path("/models/mobilenetv2_rf.h5")
   ```

#### **Option B: Use Git LFS** (If file is < 100MB)
```bash
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"

# Add and commit
git add .gitattributes
git add models/mobilenetv2_rf.h5
git commit -m "Add model with Git LFS"
git push
```

#### **Option C: Download Model on Startup** (Advanced)
Modify `main.py` to download model from cloud storage (S3, GCS) on startup.

---

## 🔍 **Debugging Steps**

### **Step 1: Check Build Logs in Railway**

1. Go to Railway dashboard
2. Click on failed deployment
3. Click **"Build Logs"** tab
4. Look for error messages:
   - "No such file or directory" → Missing files
   - "Module not found" → Missing dependency
   - "Could not find a version" → Version doesn't exist
   - "Out of memory" → Need more resources

### **Step 2: Common Error Messages**

**Error: "tensorflow==2.20.0" not found**
- ✅ **Fixed**: Changed to 2.15.0

**Error: "cv2 module not found"**
- ✅ **Fixed**: Added opencv-python-headless

**Error: "Model file not found"**
- Model file not in git (too large)
- Solution: Use Railway volumes or Git LFS

**Error: "Out of memory during build"**
- TensorFlow installation is memory-intensive
- Solution: Railway Pro plan or optimize Dockerfile

### **Step 3: Test Locally First** (Optional but Recommended)

```bash
# Test Docker build locally
cd backend
docker build -t herbascan-backend .
docker run -p 8000:8000 herbascan-backend

# Test health endpoint
curl http://localhost:8000/health
```

If local build works, Railway should work too.

---

## 📝 **Updated Files**

I've updated:
- ✅ `backend/requirements.txt` - Fixed all dependency versions
- ✅ `backend/Dockerfile` - Added system libraries and curl

**Next Steps:**
1. **Commit the fixes:**
   ```bash
   cd backend
   git add requirements.txt Dockerfile
   git commit -m "Fix Railway build errors - TensorFlow version, OpenCV, dependencies"
   git push
   ```

2. **Redeploy on Railway:**
   - Railway will automatically detect the new commit
   - Or manually trigger redeploy

3. **Check Build Logs:**
   - Should see successful build now
   - If still failing, check logs for specific error

---

## 🚨 **If Build Still Fails**

### **Share the Build Logs**

1. In Railway, click **"Build Logs"** tab
2. Copy the error message (last 50-100 lines)
3. Share it with me so I can diagnose the exact issue

### **Common Remaining Issues:**

**Issue: Model file not found**
```
⚠️  Model file not found at: models/mobilenetv2_rf.h5
```
**Solution**: Model file not in git. Use Railway volumes (see Option A above).

**Issue: Out of memory**
```
Error: killed (signal 9)
```
**Solution**: Railway free tier has 512MB limit. TensorFlow needs more. Consider:
- Railway Pro plan
- Or optimize model size (quantization)

**Issue: Port binding**
```
Error: address already in use
```
**Solution**: Railway sets `$PORT` automatically. Update `main.py`:
```python
import os
PORT = int(os.environ.get('PORT', 8000))
```

---

## ✅ **After Successful Deployment**

Once build succeeds:

1. **Test `/health` endpoint:**
   ```
   https://YOUR-APP.up.railway.app/health
   ```
   Should return: `"model_loaded": true`

2. **If model_loaded is false:**
   - Model file not in Railway container
   - Use Railway volumes to upload model

3. **Test `/identify` endpoint:**
   - Upload plant image via Postman
   - Should return predictions + Grad-CAM image

---

## 📞 **Quick Checklist**

Before redeploying:
- [ ] Commit the updated `requirements.txt`
- [ ] Commit the updated `Dockerfile`
- [ ] Push to GitHub
- [ ] Verify model file is in git OR prepare Railway volume
- [ ] Redeploy on Railway
- [ ] Check build logs for errors
- [ ] Test `/health` endpoint

---

**Try redeploying now with these fixes!** 🚀

