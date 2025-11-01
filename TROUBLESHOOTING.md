# Railway Deployment Troubleshooting Guide

## 🚨 **Build Error: "Failed to build an image"**

If you see this error in Railway, here are the fixes I've applied:

---

## ✅ **Fixes Applied**

### **1. TensorFlow Version** ✅ FIXED
- **Was**: `tensorflow==2.20.0` (doesn't exist!)
- **Now**: `tensorflow==2.15.0` (stable version)

### **2. NumPy Compatibility** ✅ FIXED
- **Was**: `numpy>=2.0.0` (incompatible with TF 2.15.0)
- **Now**: `numpy==1.24.3` (compatible version)

### **3. Missing OpenCV** ✅ FIXED
- **Was**: `cv2` used but not in requirements
- **Now**: Added `opencv-python-headless==4.8.1.78`

### **4. Missing System Libraries** ✅ FIXED
- **Added**: `libsm6`, `libxext6`, `libxrender-dev`, `libgomp1`, `curl`

### **5. PORT Environment Variable** ✅ FIXED
- **Updated**: Dockerfile and main.py to use `$PORT` from Railway

---

## 🔄 **Next Steps to Redeploy**

### **Step 1: Commit the Fixes**

```bash
cd backend
git add requirements.txt Dockerfile main.py
git commit -m "Fix Railway build errors - TensorFlow version, OpenCV, dependencies"
git push origin main
```

### **Step 2: Railway Will Auto-Redeploy**

Railway automatically detects new commits and redeploys. Wait ~5-10 minutes.

### **Step 3: Check Build Logs**

1. Go to Railway dashboard
2. Click on your project
3. Click **"Build Logs"** tab
4. Look for:
   - ✅ "Successfully built" → Build succeeded!
   - ❌ Error messages → See troubleshooting below

---

## 🐛 **If Build Still Fails**

### **Check Build Logs for Specific Error**

Common errors and solutions:

#### **Error 1: "tensorflow==2.20.0 not found"**
**Status**: ✅ Already fixed
**Action**: Should work now after commit

#### **Error 2: "cv2 module not found"**
**Status**: ✅ Already fixed
**Action**: Should work now after commit

#### **Error 3: "Out of memory" or "Killed (signal 9)"**
**Problem**: TensorFlow installation needs >512MB RAM
**Solutions**:
- **Option A**: Upgrade to Railway Pro ($20/month)
- **Option B**: Optimize build (see below)

#### **Error 4: "Model file not found"**
**Problem**: Model file (`mobilenetv2_rf.h5`) not in git (too large)
**Solutions**:
- **Option A**: Use Railway Volumes (see below)
- **Option B**: Use Git LFS (see below)
- **Option C**: Download on startup from cloud storage

---

## 📦 **Handling Large Model Files**

### **Problem**: Model File Too Large for Git (>50MB)

If `mobilenetv2_rf.h5` is >50MB, Git might not upload it.

#### **Solution A: Railway Volumes** (Easiest)

1. **In Railway Dashboard:**
   - Go to your project
   - Click **"Volumes"** tab
   - Click **"New Volume"**
   - Name: `models`
   - Mount path: `/models`

2. **Upload Model File:**
   - Click on volume
   - Upload `mobilenetv2_rf.h5` via Railway UI
   - Or use Railway CLI:
     ```bash
     railway volumes upload models mobilenetv2_rf.h5
     ```

3. **Update Code:**
   ```python
   # In backend/main.py
   MODEL_PATH = Path("/models/mobilenetv2_rf.h5")
   ```

4. **Redeploy**

#### **Solution B: Git LFS** (If file < 100MB)

```bash
# Install Git LFS
git lfs install

# Track .h5 files
git lfs track "*.h5"
git lfs track "*.h5"

# Add files
git add .gitattributes
git add models/mobilenetv2_rf.h5
git commit -m "Add model with Git LFS"
git push

# Verify LFS
git lfs ls-files
```

---

## 🔍 **Debugging: Check Build Logs**

### **How to Access Build Logs**

1. **In Railway Dashboard:**
   - Click on failed deployment (red badge)
   - Click **"Build Logs"** tab
   - Scroll to bottom for error messages

2. **Common Log Patterns:**

**Success:**
```
Successfully built abc123def456
```

**Failure - Dependency:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.20.0
```

**Failure - Module:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Failure - File:**
```
FileNotFoundError: models/mobilenetv2_rf.h5
```

**Failure - Memory:**
```
Killed (signal 9)
```

---

## 🚀 **Optimization Tips**

### **If Memory Issues Persist**

#### **Option 1: Multi-stage Docker Build**
Create optimized Dockerfile (I can create this if needed)

#### **Option 2: Reduce Model Size**
- Use model quantization
- Or download model from cloud on startup

#### **Option 3: Railway Pro**
- $20/month for more resources
- Better for production deployments

---

## 📝 **After Successful Build**

Once build succeeds:

1. **Test Health Endpoint:**
   ```
   https://YOUR-APP.up.railway.app/health
   ```
   Should return: `"model_loaded": true`

2. **If `model_loaded` is `false`:**
   - Model file not in container
   - Use Railway Volumes (Solution A above)

3. **Test Identify Endpoint:**
   - Use Postman collection
   - Upload plant image
   - Should get predictions + Grad-CAM

---

## ✅ **Checklist Before Redeploying**

- [ ] Committed updated `requirements.txt`
- [ ] Committed updated `Dockerfile`
- [ ] Committed updated `main.py`
- [ ] Pushed to GitHub
- [ ] Model file in git OR prepared Railway volume
- [ ] Railway project connected to GitHub repo
- [ ] Ready to check build logs

---

## 🆘 **Still Having Issues?**

### **Share Build Logs**

1. In Railway, click **"Build Logs"** tab
2. Copy the **last 50-100 lines** of the log
3. Share with me so I can diagnose

### **Or Try Local Build First**

```bash
cd backend
docker build -t herbascan-backend .
docker run -p 8000:8000 herbascan-backend
```

If local build works, Railway should work too.

---

**Try redeploying now!** The fixes should resolve most common issues. 🚀

