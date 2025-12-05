# HerbaScan Backend API

**Last Updated**: December 2025  
**Backend Version**: 0.8.5  
**Flutter App Version**: v0.8.5

**Model Standardization**: MobileNetV2 Only (Phase 34) - HerbaScan custom model deprecated  
**AI Explanation Standardization**: Phase 35 Complete - Structured format with 42 plants

FastAPI server for true Grad-CAM (Gradient-weighted Class Activation Mapping) computation using TensorFlow.

This backend provides online GradCAM computation for the HerbaScan mobile app's Hybrid XAI Explanation System. The Flutter app uses this backend for online heatmap generation when internet connectivity is available, falling back to offline CAM when offline.

## 📋 Setup Instructions

> **Note**: This backend is part of the HerbaScan Hybrid XAI Explanation System. The Flutter app (v0.8.5) uses this backend for online GradCAM computation, while offline explanations use pre-written structured JSON data (42 plants with taxonomy, ecology, medicinal_preparation, and safety_consideration) and offline CAM heatmaps.

### 1. Place Model Files

Place the following files in the `models/` directory:

```
backend/models/
├── MobileNetV2_model.keras    ← MobileNetV2 architecture model (.keras format) - REQUIRED
└── labels.json                ← Plant class labels (optional, for backward compatibility)
```

**Where to get these files:**
- `MobileNetV2_model.keras`: MobileNetV2 architecture model (`.keras` format) - **REQUIRED**
- `labels.json`: Class labels in JSON format (optional, backend can work without it)

**Note:** As of Phase 34 (Model Standardization), the backend uses **only MobileNetV2 model** for prediction consistency between offline CAM and online GradCAM. The HerbaScan custom model (`herbascan_model.keras`) is deprecated.

**labels.json format example (backend format - index:name):**
```json
{
  "0": "Adelfa",
  "1": "Akapulko",
  "2": "Alagaw",
  "3": "AloeVera",
  ...
  "41": "YerbaBuena"
}
```

**Frontend labels format (assets/models/class_indices.json - name:index):**
```json
{
  "Adelfa": 0,
  "Akapulko": 1,
  "Alagaw": 2,
  ...
  "YerbaBuena": 41
}
```

**Important:** The frontend uses `assets/models/class_indices.json` with format `name:index`, while the backend uses `backend/models/labels.json` with format `index:name`. Both formats are supported.
*(Your dataset has 42 Philippine medicinal plant species - indices 0-41)*

---

## 🔄 Updating Models

### Adding/Updating Backend Models

When you have a new trained model or updated labels:

#### Step 1: Update Model Files

1. **Replace the model file (`.keras` format):**
   ```bash
   # Backup old model (optional)
   cp backend/models/MobileNetV2_model.keras backend/models/MobileNetV2_model.keras.backup
   
   # Copy new model (MobileNetV2 only - HerbaScan deprecated)
   cp /path/to/your/new_mobilenetv2_model.keras backend/models/MobileNetV2_model.keras
   ```

   **Note:** Only MobileNetV2 model is required. HerbaScan custom model is deprecated as of Phase 34 (Model Standardization).

2. **Update labels.json (if classes changed):**
   ```bash
   # Edit backend/models/labels.json to match your new model's classes
   # Format: {"0": "PlantName1", "1": "PlantName2", ...}
   # Make sure class indices match the model's output
   ```

3. **Update frontend labels (if classes changed):**
   ```bash
   # Edit assets/models/class_indices.json
   # Format: {"PlantName1": 0, "PlantName2": 1, ...}
   # This is the reverse format (name:index instead of index:name)
   ```

4. **Verify model compatibility:**
   - Input shape must be: `(224, 224, 3)`
   - Output shape must be: `(num_classes,)`
   - Model must have at least one convolutional layer for Grad-CAM
   - Format: `.keras` (TensorFlow Keras SavedModel format)

#### Step 2: Test Locally

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Test model loading (MobileNetV2 only)
python -c "import tensorflow as tf; from pathlib import Path; \
m1 = Path('models/MobileNetV2_model.keras'); \
if m1.exists(): model = tf.keras.models.load_model(str(m1)); print('MobileNetV2 loaded!'); print(f'Input: {model.input_shape}'); print(f'Output: {model.output_shape}'); \
else: print('ERROR: MobileNetV2_model.keras not found!')"

# Run server and test
python main.py
# In another terminal:
curl http://localhost:8000/health
```

#### Step 3: Regenerate Offline CAM Files (Phase 2)

If you updated the models, you need to regenerate the offline CAM files for the Flutter app:

**Important:** The extraction scripts (`extract_cam_weights.py` and `create_multi_output_tflite.py`) currently support `.h5` format. To use with `.keras` models:

**Option A: Update Scripts to Support .keras**
1. Edit `extract_cam_weights.py` and `create_multi_output_tflite.py`
2. Change `MODEL_PATH` from `models/mobilenetv2_rf.h5` to `models/MobileNetV2_model.keras` (or `models/herbascan_model.keras`)
3. The scripts use `tf.keras.models.load_model()` which supports both `.h5` and `.keras` formats

**Option B: Convert .keras to .h5 Temporarily**
```bash
# Convert .keras to .h5 for extraction scripts
python -c "import tensorflow as tf; \
model = tf.keras.models.load_model('models/MobileNetV2_model.keras'); \
model.save('models/mobilenetv2_rf.h5', save_format='h5')"
```

Then run extraction scripts:
```bash
cd backend

# Step 1: Extract CAM weights
python extract_cam_weights.py
# Output: backend/models/mobilenetv2_cam_weights.json

# Step 2: Create multi-output TFLite model
python create_multi_output_tflite.py
# Output: backend/models/mobilenetv2_multi_output.tflite
```

**Note:** Only MobileNetV2 model extraction is required. HerbaScan model is deprecated.

#### Step 4: Update Flutter Assets & Redeploy Backend

**Copy generated files to Flutter assets:**
```bash
cp backend/models/mobilenetv2_cam_weights.json assets/models/
cp backend/models/mobilenetv2_multi_output.tflite assets/models/
# Note: Frontend uses class_indices.json (name:index format), not labels.json
# If you need to update frontend labels, edit assets/models/class_indices.json directly
# Note: HerbaScan model files are deprecated - only MobileNetV2 is required
```

**Update `pubspec.yaml` to include:**
```yaml
flutter:
  assets:
    - assets/models/
    # This includes all files in assets/models/:
    # - MobileNetV2_model.tflite
    # - herbascan_model.tflite
    # - class_indices.json
    # - mobilenetv2_cam_weights.json
    # - mobilenetv2_multi_output.tflite (if using multi-output model)
```

**Important:** The frontend uses `assets/models/class_indices.json` (format: `{"PlantName": index}`), not `backend/models/labels.json` (format: `{"index": "PlantName"}`). Update `class_indices.json` if class mappings change.

**Rebuild Flutter app:**
```bash
flutter clean && flutter pub get && flutter run
```

**Redeploy Backend (Railway):**

If deployed to Railway, update the deployment:

**Option A: Using Git (if models are committed)**
```bash
git add backend/models/MobileNetV2_model.keras backend/models/labels.json
git commit -m "Update MobileNetV2 model to v2.0"
git push
# Railway will automatically redeploy
# Note: HerbaScan model is deprecated - only MobileNetV2 is required
```

**Option B: Using Railway Volumes (for large models)**
1. Go to Railway dashboard
2. Navigate to your project → Volumes
3. Upload MobileNetV2 model file via Railway CLI or dashboard
4. Restart the service
5. **Note:** Only MobileNetV2 model is required (HerbaScan deprecated)

**Option C: Manual Upload**
1. SSH into Railway service (if available)
2. Upload MobileNetV2 model file directly
3. Restart service
4. **Note:** Only MobileNetV2 model is required (HerbaScan deprecated)

---

---

### Phase 2: Model Extraction & Conversion

#### Overview

Phase 2 prepares your trained Keras model for offline use in the Flutter app by:
1. Extracting CAM weights from the classification layer (`extract_cam_weights.py`)
2. Creating a multi-output TFLite model with feature maps and predictions (`create_multi_output_tflite.py`)

**Why these scripts are needed:**
- The Flutter app needs CAM weights for offline Class Activation Map computation
- The Flutter app needs a TFLite model with both feature maps and predictions as outputs
- The original Keras model only has predictions, so we need to extract intermediate layers

**Important Notes for .keras Models:**
- The extraction scripts (`extract_cam_weights.py` and `create_multi_output_tflite.py`) currently reference `.h5` format
- To use with `.keras` models, either:
  1. Update `MODEL_PATH` in both scripts to point to your `.keras` file (recommended)
  2. Convert `.keras` to `.h5` temporarily: `model.save('models/mobilenetv2_rf.h5', save_format='h5')`
- `tf.keras.models.load_model()` supports both `.keras` and `.h5` formats natively

---

#### Prerequisites

Before running the scripts, ensure you have:

1. **Model file in place:**
   ```bash
   # Verify MobileNetV2 model exists (required)
   ls -lh backend/models/MobileNetV2_model.keras
   # Or if using .h5 format for extraction scripts:
   ls -lh backend/models/mobilenetv2_rf.h5
   # Note: HerbaScan model is deprecated - only MobileNetV2 is required
   ```

2. **Python environment activated:**
   ```bash
   cd backend
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Mac/Linux
   ```

3. **TensorFlow installed:**
   ```bash
   pip show tensorflow
   # Recommended: tensorflow==2.15.0 or tensorflow==2.13.0
   ```

4. **Required dependencies:**
   ```bash
   pip install tensorflow numpy
   ```

---

#### Step 1: Extract CAM Weights

**Script:** `extract_cam_weights.py`

**Purpose:** Extracts the weights from the final dense/classification layer of your Keras model. These weights are used by the Flutter app to compute Class Activation Maps (CAM) offline.

**Command:**
```bash
cd backend
python extract_cam_weights.py
```

**What the script does:**

1. **Loads the Keras model:**
   - Reads `models/MobileNetV2_model.keras` (or `models/mobilenetv2_rf.h5` if using legacy format)
   - Verifies the model can be loaded
   - **Note:** Script processes MobileNetV2 model only (HerbaScan model is deprecated)
   - `tf.keras.models.load_model()` supports both `.keras` and `.h5` formats

2. **Finds the classification layer:**
   - Searches for layers with names like: `predictions`, `dense`, `dense_1`, `fc`, `classifier`, `output`, `softmax`, `dense_final`
   - If not found by name, finds the last layer with weights
   - Prints the layer name and type

3. **Extracts weights:**
   - Gets the weight matrix (shape: `[input_features, num_classes]`)
   - Gets the bias vector (if available)
   - Converts to NumPy arrays

4. **Saves to JSON:**
   - Creates `models/mobilenetv2_cam_weights.json`
   - Includes: weights, bias, shape, layer name, layer type, metadata

**Expected output:**
```
============================================================
Phase 2.1: Extracting CAM Weights
Extracting from MobileNetV2 model (HerbaScan model deprecated)
============================================================

[1/4] Loading MobileNetV2 model from: models/MobileNetV2_model.keras
      Model loaded successfully!

[2/4] Finding classification layer...
      Found layer: 'dense_1'
      Layer type: Dense

[3/4] Extracting weights...
      Weight shape: (1280, 42)
      Weight dtype: float32
      Bias shape: (42,)

[4/4] Saving to JSON...
      Saved to: models/mobilenetv2_cam_weights.json
      File size: 65.23 KB

============================================================
SUCCESS: MobileNetV2 CAM weights extracted!
============================================================
  Layer: dense_1
  Shape: 1280 features x 42 classes
  Output: models/mobilenetv2_cam_weights.json

Next step: Run create_multi_output_tflite.py
```

**Output file structure:**
```json
{
  "weights": [[...], [...], ...],  // 2D array: [1280, 42]
  "shape": [1280, 42],
  "layer_name": "dense_1",
  "layer_type": "Dense",
  "bias": [...],  // 1D array: [42]
  "num_classes": 42,
  "feature_dim": 1280,
  "model_name": "MobileNetV2"
}
```

**Verification:**
```bash
# Check file exists and has reasonable size
ls -lh backend/models/mobilenetv2_cam_weights.json
# Should be ~65 KB

# Verify JSON is valid
python -c "import json; json.load(open('backend/models/mobilenetv2_cam_weights.json'))"
```

---

#### Step 2: Create Multi-Output TFLite Model

**Script:** `create_multi_output_tflite.py`

**Purpose:** Creates a TFLite model with 2 outputs:
1. Feature maps from the last convolutional layer (for CAM computation)
2. Final predictions (for classification)

**Command:**
```bash
python create_multi_output_tflite.py
```

**What the script does:**

1. **Loads the Keras model:**
   - Reads `models/MobileNetV2_model.keras` (or `models/mobilenetv2_rf.h5` if using legacy format)
   - Verifies the model can be loaded
   - **Note:** Script processes MobileNetV2 model only (HerbaScan model is deprecated)
   - `tf.keras.models.load_model()` supports both `.keras` and `.h5` formats

2. **Finds the last convolutional layer:**
   - Scans all layers to find convolutional layers (Conv2D, DepthwiseConv2D, SeparableConv2D)
   - Stops at the first Dense or GlobalAveragePooling2D layer (after last conv)
   - Selects the last convolutional layer before pooling/dense layers
   - Prints all conv layers found and the selected one

3. **Creates multi-output model:**
   - Creates a new Keras model with 2 outputs:
     - Output 0: Feature maps from last conv layer (shape: `[1, 7, 7, 1280]`)
     - Output 1: Predictions from original model (shape: `[1, 42]`)
   - Tests the model with dummy input to ensure both outputs work

4. **Converts to TFLite:**
   - Tries multiple conversion methods (5 fallback methods):
     - Method 1: Direct Keras model conversion
     - Method 2: Concrete function approach (better for newer TF)
     - Method 3: H5 save/reload approach
     - Method 4: SavedModel with explicit signature
     - Method 5: Minimal conversion (last resort)
   - Uses no optimization (to preserve all outputs)
   - Uses TFLITE_BUILTINS only (for maximum compatibility)

5. **Verifies the TFLite model:**
   - Loads the TFLite model
   - Checks it has exactly 2 outputs
   - Verifies output shapes:
     - One 4D output (feature maps: `[1, 7, 7, 1280]`)
     - One 2D output (predictions: `[1, 42]`)
   - Handles output order swapping (TFLite may swap outputs)

**Expected output:**
```
============================================================
Phase 2.2: Creating Multi-Output TFLite Model
Creating MobileNetV2 multi-output model (HerbaScan model deprecated)
============================================================

[1/5] Loading MobileNetV2 model from: models/MobileNetV2_model.keras
      Model loaded successfully!
      Input shape: (None, 224, 224, 3)
      Output shape: (None, 42)

[2/5] Finding last convolutional layer...
      Scanning model layers...
      Found conv layer [15]: Conv_1 -> (None, 7, 7, 1280)
      Stopping at layer [16]: global_average_pooling2d (GlobalAveragePooling2D)
      Selected last conv layer: Conv_1 -> (None, 7, 7, 1280)
      Found layer: 'Conv_1'
      Output shape: (None, 7, 7, 1280)

[3/5] Creating multi-output model...
      Building model for TFLite conversion...
      Model Output 0 (conv): (1, 7, 7, 1280)
      Model Output 1 (predictions): (1, 42)
      ✅ Multi-output model created!
      Output 0 (features) shape: (1, 7, 7, 1280)
      Output 1 (predictions) shape: (1, 42)
      Testing multi-output model with dummy input...
      ✅ Model test successful - both outputs accessible

[4/5] Converting to TFLite...
      Converting to TFLite with compatibility flags...
      Attempting conversion with from_keras_model...
      ✅ Saved to: models/mobilenetv2_multi_output.tflite
      File size: 12.45 MB

[5/5] Verifying TFLite model...
      Verifying TFLite model...
      ✅ Model loaded successfully!
      Input shape: [1 224 224 3]
      Number of outputs: 2
      Output 0 shape: [1 7 7 1280] (dimensions: 4)
      Output 1 shape: [1 42] (dimensions: 2)
      ✅ Output 0 is feature maps: [1 7 7 1280]
      ✅ Output 1 is predictions: [1 42]

============================================================
SUCCESS: Multi-output TFLite model created!
============================================================
  Output: models/mobilenetv2_multi_output.tflite
  Feature maps shape: (None, 7, 7, 1280)
  Predictions shape: (None, 42)

Next step: Copy files to Flutter assets/ folder
  1. Copy cam_weights.json to assets/models/
  2. Copy mobilenetv2_multi_output.tflite to assets/models/
  3. Update pubspec.yaml
```

**Output file:**
- File: `backend/models/mobilenetv2_multi_output.tflite` (~10-15 MB)
- Format: TensorFlow Lite (`.tflite`)
- Outputs: 2 outputs (feature maps + predictions)

**Verification:**
```bash
# Check file exists and has reasonable size
ls -lh backend/models/mobilenetv2_multi_output.tflite
# Should be ~10-15 MB

# Verify TFLite model (optional)
python -c "
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='backend/models/mobilenetv2_multi_output.tflite')
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
print(f'Number of outputs: {len(output_details)}')
for i, detail in enumerate(output_details):
    print(f'Output {i}: {detail[\"shape\"]}')
"
```

---

#### Step 3: Copy to Flutter Assets & Update pubspec.yaml

**Copy generated files:**
```bash
# From project root (MobileNetV2 only)
cp backend/models/mobilenetv2_cam_weights.json assets/models/
cp backend/models/mobilenetv2_multi_output.tflite assets/models/
cp backend/models/labels.json assets/models/  # Optional
# Note: HerbaScan model files are deprecated - only MobileNetV2 is required
```

**Update `pubspec.yaml`:**
```yaml
flutter:
  assets:
    - assets/models/mobilenetv2_cam_weights.json
    - assets/models/mobilenetv2_multi_output.tflite
    - assets/models/labels.json  # Optional
# Note: HerbaScan model files are deprecated - only MobileNetV2 is required
```

**Rebuild Flutter app:**
```bash
flutter clean && flutter pub get && flutter run
```

---

#### Complete Workflow Summary

```bash
# 1. Navigate to backend directory
cd backend

# 2. Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Extract CAM weights
python extract_cam_weights.py

# 4. Create multi-output TFLite model
python create_multi_output_tflite.py

# 5. Copy to Flutter assets and update pubspec.yaml (see Step 3 above)
# 6. Rebuild Flutter app: flutter clean && flutter pub get && flutter run
```

---

### Troubleshooting Model Updates

**Common Issues & Quick Fixes:**

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Model Not Loading** | `"model_loaded": false` | Check file path, verify model format, check TensorFlow version (2.15.0 recommended) |
| **TFLite Conversion Fails** | AttributeError or conversion errors | Try TensorFlow 2.15.0, check model structure, review script output |
| **CAM Weights Extraction Fails** | "Could not find classification layer" | Check model architecture, update layer name in script, verify model has dense layer |
| **Multi-Output TFLite Fails** | Wrong number of outputs | Verify model has conv layers, try different TF version, check conversion method used |
| **Flutter App Can't Load Models** | Model file not found | Verify files in assets/, check pubspec.yaml, run `flutter clean` |

**Detailed Solutions:**

#### Issue: TFLite Conversion Fails

**Symptoms:**
- Error: `AttributeError: 'TFLiteSavedModelConverterV2' object has no attribute '_funcs'`
- Error: "Conversion failed" or "Model has wrong number of outputs"

**Solutions:**

1. **Try different TensorFlow version:**
   ```bash
   pip install tensorflow==2.15.0
   # Or
   pip install tensorflow==2.13.0
   ```

2. **Check model structure:**
   - Ensure model has convolutional layers
   - Verify model can be loaded and run predictions
   - Check that multi-output model creation succeeds before conversion

3. **Review conversion script output:**
   - The script tries 5 different conversion methods
   - Check which method succeeded (if any)
   - Look for detailed error messages

4. **Verify outputs manually:**
   ```python
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path='backend/models/mobilenetv2_multi_output.tflite')
   interpreter.allocate_tensors()
   output_details = interpreter.get_output_details()
   print(f"Number of outputs: {len(output_details)}")
   for i, detail in enumerate(output_details):
       print(f"Output {i}: {detail['shape']}")
   ```

#### Issue: CAM Weights Extraction Fails

**Quick Fixes:**
1. Verify model file exists and can be loaded
2. Check model architecture has dense/classification layer
3. Find layer manually: `for layer in reversed(model.layers): if len(layer.get_weights()) > 0: print(layer.name)`
4. Update `extract_cam_weights.py` to add your layer name to `possible_names` list
5. Use TensorFlow 2.15.0 (recommended)

#### Issue: Multi-Output TFLite Conversion Fails

**Quick Fixes:**
1. Verify model has convolutional layers
2. Check last conv layer output shape matches expected `[1, 7, 7, 1280]`
3. Try TensorFlow 2.15.0 (uninstall current, install recommended version)
4. Script tries 5 conversion methods - check output for which succeeded
5. Verify multi-output model creation before conversion
6. Check TFLite model after conversion: `interpreter.get_output_details()`

#### Issue: Flutter App Can't Load Models

**Quick Fixes:**
1. Verify files exist in `assets/models/` directory
2. Check `pubspec.yaml` includes all model files
3. Run `flutter clean && flutter pub get && flutter run`
4. Verify TFLite model has exactly 2 outputs

---

### Model Compatibility Notes

#### TensorFlow Versions

- **Recommended:** TensorFlow 2.15.0 or 2.13.0
- **Avoid:** TensorFlow 3.x (not yet fully compatible)
- **Current:** TensorFlow 2.20.0 (may have compatibility issues with TFLite conversion)

#### Model Requirements

- **Input shape:** Must be `(224, 224, 3)`
- **Output shape:** Must be `(num_classes,)` where `num_classes` matches your dataset
- **Architecture:** Must have at least one convolutional layer
- **Format:** Keras SavedModel format (`.keras`) - **NEW** - Supports both `.keras` and `.h5` formats

#### TFLite Conversion Notes

- **Optimization:** Disabled to preserve all outputs (feature maps + predictions)
- **Ops:** Uses TFLITE_BUILTINS only for maximum compatibility
- **Output order:** May be swapped (Flutter code handles this automatically)
- **File size:** ~10-15 MB (without optimization)

---

## 🚀 Quick Setup & Local Development

### Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Run the server
python main.py
# Or: uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Test endpoints:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/test
curl -X POST http://localhost:8000/identify -F "file=@path/to/image.jpg"
```

## 🚀 Deployment to Railway

### Prerequisites

- ✅ Railway account ([signup here](https://railway.app))
- ✅ GitHub account
- ✅ Model files in `backend/models/` directory

### Step-by-Step Deployment

#### Step 1: Create GitHub Repository

**Option A: Separate Backend Repo (Recommended)**

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

**Option B: Use Existing Repo's Backend Folder**

```bash
# From project root (herbascan/)
git add backend/
git commit -m "Add backend API for Grad-CAM computation"
git push
```

#### Step 2: Deploy to Railway

1. **Login to Railway:**
   - Go to [https://railway.app](https://railway.app)
   - Click **"Login"** (use GitHub account)
   - Authorize Railway to access your GitHub

2. **Create New Project:**
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Choose your repository:
     - If separate repo: Select `herbascan-backend`
     - If same repo: Select `herbascan` (you'll specify folder next)

3. **Configure Build Settings:**

   **If using same repo with backend folder:**
   - Click **"Settings"**
   - Under **"Build"**, set:
     - **Root Directory**: `backend`
     - **Builder**: `Dockerfile`

   **If using separate backend repo:**
   - Railway will auto-detect the Dockerfile ✅

4. **Add Environment Variables (Optional):**

   Click **"Variables"** tab and add:
   ```
   PORT=8000
   ```

   Railway automatically provides `$PORT`, but you can set a default.

5. **Add Model Files:**

   **Important:** Model files are typically too large for GitHub (>100MB). Use one of these methods:

   **Option A: Git LFS (for models < 100MB)**
   ```bash
   # Install Git LFS
   git lfs install
   git lfs track "*.keras"
   git lfs track "*.h5"  # If still using .h5 format
   git add .gitattributes
   git add models/MobileNetV2_model.keras
   git add models/herbascan_model.keras
   git commit -m "Add model files via Git LFS"
   git push
   ```

   **Option B: Railway Volumes (Recommended for large models)**
   - Go to Railway dashboard → Your project → Volumes
   - Create a new volume
   - Upload model files via Railway CLI:
     ```bash
     railway volumes create
     railway volumes upload models/MobileNetV2_model.keras
     railway volumes upload models/herbascan_model.keras
     railway volumes upload models/labels.json
     ```
   - Update `MODEL_PATH` and `LABELS_PATH` in code to point to volume

   **Option C: Cloud Storage (S3, GCS)**
   - Upload models to cloud storage
   - Download on server startup (modify `main.py` to download on startup)

   **Option D: Include in Docker Image**
   - If model is < 100MB, you can include it in the Docker image
   - Ensure `models/` directory is copied in Dockerfile

6. **Deploy:**
   - Railway will automatically start building
   - Wait for build to complete (~5-10 minutes, first time is slower due to TensorFlow)
   - Once deployed, you'll see a ✅ green checkmark

#### Step 3: Get Your API URL

1. In Railway dashboard, click **"Settings"**
2. Under **"Networking"**, click **"Generate Domain"**
3. Railway will create a public URL like:
   ```
   https://YOUR-RAILWAY-URL.railway.app/
   ```
4. **Copy this URL** - you'll use it in Flutter app!

#### Step 4: Test Your Deployed API

**Test 1: Health Check**

```bash
curl https://YOUR-RAILWAY-URL.railway.app/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "labels_loaded": true,
  "num_classes": 42
}
```

**Test 2: Root Endpoint**

```bash
curl https://YOUR-RAILWAY-URL.railway.app/
```

**Test 3: Plant Identification**

```bash
curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
  -F "file=@path/to/your/image.jpg"
```

Or use Postman (see Testing section below).

#### Step 5: Update Flutter App

Once deployed, update your Flutter app with the Railway URL:

> **Note**: The Flutter app (v0.8.5) includes a Hybrid XAI Explanation System that uses this backend for online GradCAM heatmaps. The app automatically falls back to offline CAM and offline structured JSON explanations (42 plants with standardized format: taxonomy, ecology, medicinal_preparation, safety_consideration) when internet is unavailable. See the main `README.md` for details on the XAI system.

```dart
// lib/core/services/online_gradcam_service.dart
static const String serverUrl = 'https://YOUR-RAILWAY-URL.railway.app';
```

---

### Continuous Deployment

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

### Railway CLI Deployment (Alternative)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up

# Check logs
railway logs

# Run commands in deployment
railway run python test_model_info.py
```

## 📡 API Endpoints

### `GET /` - Root
Returns API information and status.

### `GET /health` - Health Check
Returns server health status and model load status.

### `GET /test` - Test Endpoint
For debugging and testing server connectivity.

### `POST /identify` - Plant Identification
Main endpoint for plant identification with Grad-CAM.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "plant_name": "4Vitex negundo(VN)",
  "scientific_name": "4Vitex negundo(VN)",
  "confidence": 0.942,
  "all_predictions": [
    {"class": "4Vitex negundo(VN)", "class_index": 34, "confidence": 0.942},
    {"class": "6Blumea balsamifera(BB)", "class_index": 36, "confidence": 0.123},
    {"class": "5Moringa oleifera(MO)", "class_index": 35, "confidence": 0.089}
  ],
  "gradcam_image": "iVBORw0KGgoAAAANSUhEUg...",
  "method": "grad-cam",
  "processing_time_ms": 3456.78
}
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (optional):

```env
MOBILENETV2_MODEL_PATH=models/MobileNetV2_model.keras
LABELS_PATH=models/labels.json
PORT=8000
HOST=0.0.0.0
```

**Note:** The backend uses only MobileNetV2 model (HerbaScan deprecated). Environment variables are optional - defaults are set in `main.py`.

## 🧪 Testing with Postman

**Collection File:** `backend/HerbaScan_API.postman_collection.json`

**Included Endpoints:** Health Check (Local/Railway), Root, Test, Identify Plant (Local/Railway)

**Variables:** `base_url` (localhost:8000), `railway_url` (your Railway URL)

### Method 1: Postman Desktop App (Recommended)

1. **Install & Import:** Download Postman, import `HerbaScan_API.postman_collection.json`
2. **Update Variables:** Set `railway_url` in collection variables tab
3. **Test Local:** Start server (`python main.py`), test Health Check and Identify Plant endpoints
4. **Test Railway:** Use Railway endpoints (if multipart errors occur, use `curl` instead)

### Method 2: VS Code Extensions

**REST Client:** Install extension, create `api-test.http` file with requests  
**Thunder Client:** Install extension, import Postman collection, update variables

**Note:** REST Client has limitations with file uploads - use Postman or curl for file uploads.

### Method 3: Other IDEs

**IntelliJ/PyCharm:** Install HTTP Client plugin, create `.http` file  
**JetBrains:** Built-in HTTP Client, create `.http` file with requests

### Method 4: curl (Command Line)

   ```bash
# Local
curl http://localhost:8000/health
curl -X POST http://localhost:8000/identify -F "file=@image.jpg"

# Railway
curl https://YOUR-RAILWAY-URL.railway.app/health
curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify -F "file=@image.jpg"
```

## 🐛 **Troubleshooting Postman Issues**

| Issue | Solution |
|-------|----------|
| **Image bytes empty** | Use form-data, set file field to "File" (not "Text"), select actual image file |
| **Multipart errors (Railway)** | Use `curl` for Railway testing, Postman works fine for local |
| **Variables not working** | Set "Current Value" in collection variables tab, save collection |
| **Can't import collection** | Check JSON format, use File → Import, update Postman version |
| **Response empty/error** | Verify server running, check URL, review server logs |
| **File upload in REST Client** | Use Thunder Client or Postman/curl for file uploads |

## 🐛 Troubleshooting

### Deployment Issues

**Common Issues:**
- **Build Failed:** Check Railway logs, verify model file committed/uploaded, check Dockerfile
- **Model file not found:** Use Git LFS (<100MB) or Railway volumes (>100MB)
- **Out of memory:** Railway free tier 512MB limit - upgrade or optimize model
- **File upload errors:** Use curl for Railway, Postman works for local; verify file format (.jpg, .png, etc.)

#### Model Not Loading in Railway

**Quick Fixes:**
1. Check Railway logs for errors
2. Verify model file size (if >100MB, use Git LFS or Railway volumes)
3. Upload model via Railway volumes if needed
4. Check file permissions and Dockerfile

#### Slow Response Times / CORS Issues

- **Cold start:** First request 10-30s, subsequent 2-4s
- **CORS:** Enabled by default in `main.py`, verify Railway URL is correct

#### Grad-CAM Layer Not Found / Out of Memory / Port Issues

**Quick Fixes:**
- **Layer not found:** Find conv layer name, update `gradcam.py`, verify model architecture
- **Out of memory:** Reduce image size, use quantization, upgrade Railway plan
- **Port in use:** Kill process (`lsof -ti:8000 | xargs kill -9`) or use different port
- **Dependencies:** Update pip, use virtual environment, check Python version (3.8-3.11), install TensorFlow 2.15.0

## 📝 Notes

### Model File Management

- **Model files (`.keras` or `.h5`) are typically NOT committed to git** (too large, >100MB)
- **Options for deployment:**
  - Use Git LFS for models < 100MB
  - Use Railway volumes for large models
  - Upload to cloud storage (S3, GCS) and download on startup
  - Include in Docker image if < 100MB
- **Model Standardization (Phase 34):** Backend uses only `MobileNetV2_model.keras`. HerbaScan custom model is deprecated for prediction consistency between offline CAM and online GradCAM.
- **AI Explanation Standardization (Phase 35):** Flutter app (v0.8.5) now uses standardized structured format for all 42 plants with identical data depth (taxonomy, ecology, medicinal_preparation, safety_consideration) for both online (Gemini API) and offline (JSON) explanations.

### TensorFlow Compatibility

- **Recommended:** TensorFlow 2.15.0 or 2.13.0
- **Current:** TensorFlow 2.20.0 (may have compatibility issues with TFLite conversion)
- **Avoid:** TensorFlow 3.x (not yet fully compatible)

### Performance Expectations

| Metric | Target | Typical |
|--------|--------|---------|
| Build Time | - | 5-10 min (first time) |
| Cold Start | - | 10-30 sec |
| Inference Time | <5s | 2-4s |
| Memory Usage | <512MB | 300-450MB |
| Response Size | - | 50-200KB (with base64 image) |

### Railway Pricing

- **Free Tier**: $5 credit/month, 500 hours execution
- **Good for**: Development & testing
- **Upgrade if**: High traffic or need more resources
- **For HerbaScan**: Estimated cost ~$5-10/month (hobby usage)

---

