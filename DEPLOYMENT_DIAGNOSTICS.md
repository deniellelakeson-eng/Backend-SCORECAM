# Railway Deployment Diagnostics

## 🔍 **Healthcheck Failed - Troubleshooting Guide**

Your build succeeded ✅, but the healthcheck is failing ❌. This means the container is running, but the application isn't starting properly.

---

## **Step 1: Check Application Logs** 🔍

The logs you shared were **build logs** and **healthcheck logs**. We need **application/deployment logs** to see what's happening when the app starts.

### **How to Access Application Logs in Railway:**

1. Go to Railway dashboard
2. Click on your project
3. Click **"Deployments"** tab
4. Click on the **latest deployment** (should show "Failed" or yellow status)
5. Click **"View Logs"** or **"Deployment Logs"** tab
6. Look for:
   - `🌿 HerbaScan API Starting Up...`
   - `🔄 Loading Keras model...`
   - Any Python errors or tracebacks
   - `❌ Error loading model: ...`

### **What to Look For:**

**Good Signs:**
```
🌿 HerbaScan API Starting Up...
📂 Working directory: /app
✅ Labels loaded: 40 classes
✅ Model loaded successfully
✅ Server ready to accept requests!
```

**Bad Signs:**
```
❌ Error loading model: FileNotFoundError
⚠️  Model file not found
❌ ModuleNotFoundError: No module named '...'
❌ ImportError: ...
```

---

## **Step 2: Common Issues & Solutions** 🐛

### **Issue 1: Model File Not Found** ❌

**Symptoms:**
```
⚠️  Model file not found at: models/mobilenetv2_rf.h5
📝 Server will start, but /identify endpoint will not work
```

**Cause:** Model file (`mobilenetv2_rf.h5`) is too large for Git (>50MB) or not committed.

**Solutions:**

#### **Solution A: Use Railway Volumes** (Recommended)

1. **In Railway Dashboard:**
   - Go to your project
   - Click **"Volumes"** tab
   - Click **"New Volume"**
   - Name: `models`
   - Mount path: `/models`

2. **Upload Model File:**
   - Option 1: Via Railway CLI
     ```bash
     railway volumes upload models mobilenetv2_rf.h5
     ```
   - Option 2: Via Railway Web UI
     - Click on the volume
     - Use "Upload" button

3. **Update Code:**
   Update `backend/main.py`:
   ```python
   MODEL_PATH = Path("/models/mobilenetv2_rf.h5")
   ```

4. **Redeploy**

#### **Solution B: Download from Cloud Storage** (Alternative)

1. Upload model to Google Drive / Dropbox / S3
2. Download on startup:

   ```python
   # In backend/main.py startup function
   import requests
   
   MODEL_URL = "https://your-cloud-storage.com/mobilenetv2_rf.h5"
   
   if not MODEL_PATH.exists():
       print("📥 Downloading model from cloud storage...")
       response = requests.get(MODEL_URL)
       MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
       with open(MODEL_PATH, 'wb') as f:
           f.write(response.content)
       print("✅ Model downloaded")
   ```

---

### **Issue 2: Application Crashes on Startup** 💥

**Symptoms:** No application logs, healthcheck fails immediately.

**Check Application Logs for:**
- `ModuleNotFoundError`
- `ImportError`
- `AttributeError`
- Any Python traceback

**Common Causes:**

1. **Missing Dependencies:**
   - Check if all packages in `requirements.txt` installed correctly
   - Look for `WARNING` or `ERROR` in build logs

2. **Import Errors:**
   - Check if `utils/gradcam.py` and `utils/preprocessing.py` exist
   - Verify all imports are correct

3. **Port Binding Issue:**
   - Railway sets `PORT` environment variable
   - Our code handles this, but check logs for port conflicts

**Solution:** Share the full application/deployment logs (not build logs).

---

### **Issue 3: Model Takes Too Long to Load** ⏳

**Symptoms:**
- Healthcheck times out (5 minutes)
- Logs show `🔄 Loading Keras model...` but never finish

**Cause:** TensorFlow model loading can take 1-3 minutes on Railway free tier.

**Solution:**
- Wait longer (healthcheck window is 5 minutes)
- Model should load within 2-3 minutes
- If still failing, check for out-of-memory errors

---

### **Issue 4: Out of Memory (OOM)** 💾

**Symptoms:**
```
Killed (signal 9)
Out of memory
```

**Cause:** TensorFlow + model is too large for Railway free tier (512MB RAM).

**Solutions:**
1. **Upgrade to Railway Pro** ($20/month) - 2GB RAM
2. **Optimize model:**
   - Use model quantization
   - Use smaller model variant
   - Load model lazily (only when needed)

---

## **Step 3: Verify Model File Size** 📊

### **Check if Model is in Git:**

```bash
cd backend
git ls-files models/
```

**If `mobilenetv2_rf.h5` is NOT listed:**
- Model is too large for Git (likely >50MB)
- Use Railway Volumes (Solution A above)

**If model IS listed but small (<1MB):**
- Might be placeholder or wrong file
- Check actual file size:
  ```bash
  ls -lh models/mobilenetv2_rf.h5
  ```

---

## **Step 4: Test Without Model** ✅

The app should start even without the model. Test the `/health` endpoint:

```bash
curl https://YOUR-APP.up.railway.app/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": false,
  "labels_loaded": true,
  "num_classes": 40
}
```

**If you get this response:** ✅ App is running! Just need to add the model.

**If you get connection error:** ❌ App isn't starting - check application logs.

---

## **Step 5: Quick Debugging Commands** 🔧

### **Check Railway Environment:**

1. **In Railway Dashboard:**
   - Go to **"Variables"** tab
   - Check if `PORT` is set (Railway sets this automatically)

2. **View Real-time Logs:**
   - Railway dashboard → **"Logs"** tab
   - Should show startup messages

3. **Test Health Endpoint:**
   ```bash
   curl -v https://YOUR-APP.up.railway.app/health
   ```

---

## **Step 6: Share Diagnostics Information** 📤

If still having issues, share:

1. **Application/Deployment Logs** (not build logs):
   - Look for lines starting with: `🌿`, `🔄`, `✅`, `❌`, `⚠️`
   - Full Python traceback if any

2. **Model File Info:**
   ```bash
   ls -lh backend/models/
   ```

3. **Git Status:**
   ```bash
   git ls-files backend/models/
   ```

4. **Health Endpoint Response:**
   ```bash
   curl https://YOUR-APP.up.railway.app/health
   ```

---

## **Quick Fix Checklist** ✅

- [ ] Checked application/deployment logs (not just build logs)
- [ ] Verified model file exists in git OR prepared Railway volume
- [ ] Tested `/health` endpoint (should work even without model)
- [ ] Checked for Python errors in application logs
- [ ] Verified all dependencies installed correctly
- [ ] Confirmed PORT environment variable is set

---

## **Next Steps After App Starts** 🚀

Once `/health` returns `"status": "healthy"`:

1. **If `model_loaded: false`:**
   - Upload model using Railway Volumes (see Issue 1)

2. **If `model_loaded: true`:**
   - Test `/identify` endpoint with Postman
   - Upload plant image
   - Should get predictions + Grad-CAM

---

**The most important next step is to check the APPLICATION/DEPLOYMENT logs (not build logs) to see what's happening during startup!** 🔍

