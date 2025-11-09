# HerbaScan Backend API

FastAPI server for true Grad-CAM (Gradient-weighted Class Activation Mapping) computation using TensorFlow.

## 📋 Setup Instructions

### 1. Place Model Files

Place the following files in the `models/` directory:

```
backend/models/
├── mobilenetv2_rf.h5    ← Your trained Keras model (.h5 file)
└── labels.json          ← Plant class labels
```

**Where to get these files:**
- `mobilenetv2_rf.h5`: Your trained model from your AI training workflow
- `labels.json`: Your class labels in JSON format

**labels.json format example:**
```json
{
  "0": "10Coleus scutellarioides(CS)",
  "1": "11Phyllanthus niruri(PN)",
  "2": "12Corchorus olitorius(CO)",
  "3": "13Momordica charantia (MC)",
  ...
  "39": "9Centella asiatica(CA)"
}
```
*(Your dataset has 40 Philippine medicinal plant species)*

---

## 🔄 Updating Models

### Adding/Updating Backend Models

When you have a new trained model or updated labels:

#### Step 1: Update Model Files

1. **Replace the model file:**
   ```bash
   # Backup old model (optional)
   cp backend/models/mobilenetv2_rf.h5 backend/models/mobilenetv2_rf.h5.backup
   
   # Copy new model
   cp /path/to/your/new_model.h5 backend/models/mobilenetv2_rf.h5
   ```

2. **Update labels.json (if classes changed):**
   ```bash
   # Edit labels.json to match your new model's classes
   # Make sure class indices match the model's output
   ```

3. **Verify model compatibility:**
   - Input shape must be: `(224, 224, 3)`
   - Output shape must be: `(num_classes,)`
   - Model must have at least one convolutional layer for Grad-CAM

#### Step 2: Test Locally

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Test model loading
python -c "import tensorflow as tf; model = tf.keras.models.load_model('models/mobilenetv2_rf.h5'); print('Model loaded!'); print(f'Input: {model.input_shape}'); print(f'Output: {model.output_shape}')"

# Run server and test
python main.py
# In another terminal:
curl http://localhost:8000/health
```

#### Step 3: Regenerate Offline CAM Files (Phase 2)

If you updated the model, you need to regenerate the offline CAM files for the Flutter app:

```bash
cd backend

# Step 1: Extract CAM weights
python extract_cam_weights.py
# Output: backend/models/cam_weights.json

# Step 2: Create multi-output TFLite model
python create_multi_output_tflite.py
# Output: backend/models/mobilenetv2_multi_output.tflite
```

#### Step 4: Update Flutter Assets

After generating the new files, copy them to Flutter assets:

```bash
# Copy CAM weights
cp backend/models/cam_weights.json assets/models/cam_weights.json

# Copy multi-output TFLite model
cp backend/models/mobilenetv2_multi_output.tflite assets/models/mobilenetv2_multi_output.tflite

# Copy labels (if updated)
cp backend/models/labels.json assets/models/labels.json
```

#### Step 5: Update pubspec.yaml

Ensure `pubspec.yaml` includes the new model files:

```yaml
flutter:
  assets:
    - assets/models/cam_weights.json
    - assets/models/mobilenetv2_multi_output.tflite
    - assets/models/labels.json
```

#### Step 6: Redeploy Backend (Railway)

If deployed to Railway, update the deployment:

**Option A: Using Git (if model is committed)**
```bash
git add backend/models/mobilenetv2_rf.h5 backend/models/labels.json
git commit -m "Update model to v2.0"
git push
# Railway will automatically redeploy
```

**Option B: Using Railway Volumes (for large models)**
1. Go to Railway dashboard
2. Navigate to your project → Volumes
3. Upload new model files via Railway CLI or dashboard
4. Restart the service

**Option C: Manual Upload**
1. SSH into Railway service (if available)
2. Upload model files directly
3. Restart service

---

### Adding/Updating Flutter Assets Models

The Flutter app uses offline models for CAM computation when internet is unavailable.

#### Required Files in `assets/models/`:

```
assets/models/
├── cam_weights.json                    ← CAM classification weights
├── mobilenetv2_multi_output.tflite    ← Multi-output TFLite model
├── labels.json                         ← Plant class labels
└── labels.txt                          ← Optional: Text labels
```

#### Updating Process:

1. **Generate files from backend model:**
   - Run `extract_cam_weights.py` → creates `cam_weights.json`
   - Run `create_multi_output_tflite.py` → creates `mobilenetv2_multi_output.tflite`

2. **Copy to assets:**
   ```bash
   # From project root
   cp backend/models/cam_weights.json assets/models/
   cp backend/models/mobilenetv2_multi_output.tflite assets/models/
   cp backend/models/labels.json assets/models/
   ```

3. **Update pubspec.yaml:**
   ```yaml
   flutter:
     assets:
       - assets/models/cam_weights.json
       - assets/models/mobilenetv2_multi_output.tflite
       - assets/models/labels.json
   ```

4. **Rebuild Flutter app:**
   ```bash
   flutter clean
   flutter pub get
   flutter run
   ```

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

---

#### Prerequisites

Before running the scripts, ensure you have:

1. **Model file in place:**
   ```bash
   # Verify model exists
   ls -lh backend/models/mobilenetv2_rf.h5
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
   - Reads `models/mobilenetv2_rf.h5`
   - Verifies the model can be loaded

2. **Finds the classification layer:**
   - Searches for layers with names like: `predictions`, `dense`, `dense_1`, `fc`, `classifier`, `output`, `softmax`, `dense_final`
   - If not found by name, finds the last layer with weights
   - Prints the layer name and type

3. **Extracts weights:**
   - Gets the weight matrix (shape: `[input_features, num_classes]`)
   - Gets the bias vector (if available)
   - Converts to NumPy arrays

4. **Saves to JSON:**
   - Creates `models/cam_weights.json`
   - Includes: weights, bias, shape, layer name, layer type, metadata

**Expected output:**
```
============================================================
Phase 2.1: Extracting CAM Weights
============================================================

[1/4] Loading model from: models/mobilenetv2_rf.h5
      Model loaded successfully!

[2/4] Finding classification layer...
      Found layer: 'dense_1'
      Layer type: Dense

[3/4] Extracting weights...
      Weight shape: (1280, 40)
      Weight dtype: float32
      Bias shape: (40,)

[4/4] Saving to JSON...
      Saved to: models/cam_weights.json
      File size: 65.23 KB

============================================================
SUCCESS: CAM weights extracted!
============================================================
  Layer: dense_1
  Shape: 1280 features x 40 classes
  Output: models/cam_weights.json

Next step: Run create_multi_output_tflite.py
```

**Output file structure:**
```json
{
  "weights": [[...], [...], ...],  // 2D array: [1280, 40]
  "shape": [1280, 40],
  "layer_name": "dense_1",
  "layer_type": "Dense",
  "bias": [...],  // 1D array: [40]
  "num_classes": 40,
  "feature_dim": 1280
}
```

**Verification:**
```bash
# Check file exists and has reasonable size
ls -lh backend/models/cam_weights.json
# Should be ~65 KB

# Verify JSON is valid
python -c "import json; json.load(open('backend/models/cam_weights.json'))"
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
   - Reads `models/mobilenetv2_rf.h5`
   - Verifies the model can be loaded

2. **Finds the last convolutional layer:**
   - Scans all layers to find convolutional layers (Conv2D, DepthwiseConv2D, SeparableConv2D)
   - Stops at the first Dense or GlobalAveragePooling2D layer (after last conv)
   - Selects the last convolutional layer before pooling/dense layers
   - Prints all conv layers found and the selected one

3. **Creates multi-output model:**
   - Creates a new Keras model with 2 outputs:
     - Output 0: Feature maps from last conv layer (shape: `[1, 7, 7, 1280]`)
     - Output 1: Predictions from original model (shape: `[1, 40]`)
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
     - One 2D output (predictions: `[1, 40]`)
   - Handles output order swapping (TFLite may swap outputs)

**Expected output:**
```
============================================================
Phase 2.2: Creating Multi-Output TFLite Model
============================================================

[1/5] Loading model from: models/mobilenetv2_rf.h5
      Model loaded successfully!
      Input shape: (None, 224, 224, 3)
      Output shape: (None, 40)

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
      Model Output 1 (predictions): (1, 40)
      ✅ Multi-output model created!
      Output 0 (features) shape: (1, 7, 7, 1280)
      Output 1 (predictions) shape: (1, 40)
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
      Output 1 shape: [1 40] (dimensions: 2)
      ✅ Output 0 is feature maps: [1 7 7 1280]
      ✅ Output 1 is predictions: [1 40]

============================================================
SUCCESS: Multi-output TFLite model created!
============================================================
  Output: models/mobilenetv2_multi_output.tflite
  Feature maps shape: (None, 7, 7, 1280)
  Predictions shape: (None, 40)

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

#### Step 3: Copy to Flutter Assets

After both scripts succeed, copy the generated files to the Flutter app:

```bash
# From project root (herbascan/)
cp backend/models/cam_weights.json assets/models/
cp backend/models/mobilenetv2_multi_output.tflite assets/models/

# Also copy labels if updated
cp backend/models/labels.json assets/models/
```

**Verify files are copied:**
```bash
ls -lh assets/models/
# Should show:
# - cam_weights.json (~65 KB)
# - mobilenetv2_multi_output.tflite (~10-15 MB)
# - labels.json
```

---

#### Step 4: Update Flutter pubspec.yaml

Ensure `pubspec.yaml` includes the model files:

```yaml
flutter:
  assets:
    - assets/models/cam_weights.json
    - assets/models/mobilenetv2_multi_output.tflite
    - assets/models/labels.json
```

**Then rebuild Flutter app:**
```bash
flutter clean
flutter pub get
flutter run
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

# 5. Copy to Flutter assets (from project root)
cd ..
cp backend/models/cam_weights.json assets/models/
cp backend/models/mobilenetv2_multi_output.tflite assets/models/
cp backend/models/labels.json assets/models/

# 6. Update pubspec.yaml (add assets if not already there)

# 7. Rebuild Flutter app
flutter clean
flutter pub get
flutter run
```

---

### Troubleshooting Model Updates

#### Issue: Model Not Loading

**Symptoms:**
- Server shows `"model_loaded": false` in `/health` endpoint
- Error: "Model file not found" or "Invalid model"

**Solutions:**
1. **Check file path:**
   ```bash
   ls -lh backend/models/mobilenetv2_rf.h5
   # Should show file exists and has reasonable size
   ```

2. **Verify model format:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   print(model.summary())
   ```

3. **Check TensorFlow version:**
   ```bash
   pip show tensorflow
   # Recommended: tensorflow==2.15.0 or tensorflow==2.13.0
   ```

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

#### Issue: CAM Weights Extraction Fails (`extract_cam_weights.py`)

**Symptoms:**
- Error: "Could not find classification layer"
- Error: "Layer has no weights"
- Error: "Model file not found"

**Solutions:**

1. **Check model file exists:**
   ```bash
   ls -lh backend/models/mobilenetv2_rf.h5
   # Verify file exists and has reasonable size
   ```

2. **Check model can be loaded:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   print("Model loaded successfully!")
   print(f"Input shape: {model.input_shape}")
   print(f"Output shape: {model.output_shape}")
   ```

3. **Check model architecture:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   print("\nAll layers:")
   for i, layer in enumerate(model.layers):
       print(f"[{i}] {layer.name}: {type(layer).__name__}")
       if len(layer.get_weights()) > 0:
           print(f"     Has weights: {[w.shape for w in layer.get_weights()]}")
   ```

4. **Find classification layer manually:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   # Find last layer with weights
   for layer in reversed(model.layers):
       if len(layer.get_weights()) > 0:
           print(f"Last layer with weights: {layer.name}")
           print(f"  Type: {type(layer).__name__}")
           print(f"  Weights: {[w.shape for w in layer.get_weights()]}")
           break
   ```

5. **Update layer finding logic:**
   - Edit `backend/extract_cam_weights.py`
   - Find the `possible_names` list (around line 22)
   - Add your layer name to the list:
     ```python
     possible_names = [
         'predictions',
         'dense',
         'dense_1',
         'your_layer_name',  # Add your layer name here
         'fc',
         'classifier',
         'output',
         'softmax',
         'dense_final'
     ]
     ```

6. **Check TensorFlow version:**
   ```bash
   pip show tensorflow
   # Recommended: tensorflow==2.15.0 or tensorflow==2.13.0
   ```

#### Issue: Multi-Output TFLite Conversion Fails (`create_multi_output_tflite.py`)

**Symptoms:**
- Error: "Could not find convolutional layer"
- Error: "AttributeError: 'TFLiteSavedModelConverterV2' object has no attribute '_funcs'"
- Error: "Model has wrong number of outputs"
- Error: "Conversion failed"

**Solutions:**

1. **Check model has convolutional layers:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   conv_layers = []
   for i, layer in enumerate(model.layers):
       layer_type = type(layer).__name__
       if any(x in layer_type for x in ['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D']):
           conv_layers.append((i, layer.name, layer_type))
           print(f"Conv layer [{i}]: {layer.name} ({layer_type})")
   if not conv_layers:
       print("ERROR: No convolutional layers found in model!")
   ```

2. **Check last conv layer output shape:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   # Find last conv layer
   last_conv = None
   for layer in model.layers:
       if 'Conv' in type(layer).__name__:
           last_conv = layer
   if last_conv:
       print(f"Last conv layer: {last_conv.name}")
       # Create a test model to get output shape
       test_model = tf.keras.Model(inputs=model.input, outputs=last_conv.output)
       dummy_input = tf.zeros((1, 224, 224, 3))
       output = test_model(dummy_input)
       print(f"Output shape: {output.shape}")
   ```

3. **Try different TensorFlow version:**
   ```bash
   # Uninstall current version
   pip uninstall tensorflow
   
   # Install recommended version
   pip install tensorflow==2.15.0
   
   # Or try
   pip install tensorflow==2.13.0
   ```

4. **Check which conversion method is being used:**
   - The script tries 5 different methods
   - Check the output to see which method succeeded
   - If all methods fail, check the error message
   - Method 2 (concrete function) usually works best with newer TensorFlow

5. **Verify multi-output model creation:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   # Manually create multi-output model
   last_conv_layer = None
   for layer in model.layers:
       if 'Conv' in type(layer).__name__:
           last_conv_layer = layer
   if last_conv_layer:
       multi_output = tf.keras.Model(
           inputs=model.input,
           outputs=[last_conv_layer.output, model.output]
       )
       print(f"Multi-output model created!")
       print(f"Output 0 shape: {multi_output.output[0].shape}")
       print(f"Output 1 shape: {multi_output.output[1].shape}")
       # Test with dummy input
       dummy = tf.zeros((1, 224, 224, 3))
       outputs = multi_output(dummy)
       print(f"Test outputs: {len(outputs)} outputs")
   ```

6. **Check TFLite model after conversion:**
   ```python
   import tensorflow as tf
   interpreter = tf.lite.Interpreter(model_path='backend/models/mobilenetv2_multi_output.tflite')
   interpreter.allocate_tensors()
   output_details = interpreter.get_output_details()
   print(f"Number of outputs: {len(output_details)}")
   for i, detail in enumerate(output_details):
       print(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")
   ```

7. **Review script output:**
   - The script provides detailed error messages
   - Check which conversion method failed
   - Look for specific error types (AttributeError, ValueError, etc.)
   - Check TensorFlow version compatibility warnings

#### Issue: Flutter App Can't Load Models

**Symptoms:**
- Error: "Model file not found in assets"
- Error: "TFLite model has wrong number of outputs"

**Solutions:**
1. **Verify files in assets:**
   ```bash
   ls -lh assets/models/
   # Should show: cam_weights.json, mobilenetv2_multi_output.tflite, labels.json
   ```

2. **Check pubspec.yaml:**
   ```yaml
   flutter:
     assets:
       - assets/models/cam_weights.json
       - assets/models/mobilenetv2_multi_output.tflite
       - assets/models/labels.json
   ```

3. **Clean and rebuild:**
   ```bash
   flutter clean
   flutter pub get
   flutter run
   ```

4. **Verify TFLite model:**
   - Use a TFLite viewer or test the model in Python first
   - Ensure model has exactly 2 outputs

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
- **Format:** Keras H5 format (`.h5`)

#### TFLite Conversion Notes

- **Optimization:** Disabled to preserve all outputs (feature maps + predictions)
- **Ops:** Uses TFLITE_BUILTINS only for maximum compatibility
- **Output order:** May be swapped (Flutter code handles this automatically)
- **File size:** ~10-15 MB (without optimization)

---

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Locally

```bash
# Run the server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

### 4. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Test Endpoint:**
```bash
curl http://localhost:8000/test
```

**Identify Plant (using Postman or curl):**
```bash
curl -X POST \
  http://localhost:8000/identify \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/plant/image.jpg"
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
   git lfs track "*.h5"
   git add .gitattributes
   git add models/mobilenetv2_rf.h5
   git commit -m "Add model file via Git LFS"
   git push
   ```

   **Option B: Railway Volumes (Recommended for large models)**
   - Go to Railway dashboard → Your project → Volumes
   - Create a new volume
   - Upload model files via Railway CLI:
     ```bash
     railway volumes create
     railway volumes upload models/mobilenetv2_rf.h5
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
  "num_classes": 40
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
MODEL_PATH=models/mobilenetv2_rf.h5
LABELS_PATH=models/labels.json
PORT=8000
HOST=0.0.0.0
```

## 🧪 Testing with Postman

The HerbaScan API comes with a ready-to-use Postman collection (`HerbaScan_API.postman_collection.json`) that includes all endpoints for both local and Railway deployment.

### 📦 **Postman Collection File**

**Location:** `backend/HerbaScan_API.postman_collection.json`

**Included Endpoints:**
1. Health Check (Local) - `GET http://localhost:8000/health`
2. Health Check (Railway) - `GET {{railway_url}}/health`
3. Root Endpoint - `GET http://localhost:8000/`
4. Test Endpoint - `GET http://localhost:8000/test`
5. Identify Plant (Local) - `POST http://localhost:8000/identify`
6. Identify Plant (Railway) - `POST {{railway_url}}/identify`

**Variables:**
- `base_url`: `http://localhost:8000` (for local testing)
- `railway_url`: `https://YOUR-RAILWAY-URL.railway.app` (update with your Railway URL)

---

## 🖥️ **Method 1: Using Postman Desktop App** (Recommended)

### Step 1: Install Postman

1. **Download Postman:**
   - Go to [https://www.postman.com/downloads/](https://www.postman.com/downloads/)
   - Download for your operating system (Windows, Mac, Linux)
   - Install and open Postman

2. **Create a Postman account** (optional but recommended):
   - Sign up for free at [https://www.postman.com/signup/](https://www.postman.com/signup/)
   - Or use without account (limited features)

### Step 2: Import the Collection

1. **Open Postman**

2. **Import the collection:**
   - Click **"Import"** button (top left)
   - Or go to **File → Import**
   - Click **"Upload Files"** or **"Choose Files"**
   - Navigate to `backend/HerbaScan_API.postman_collection.json`
   - Select the file and click **"Import"**

3. **Verify collection imported:**
   - You should see **"HerbaScan Grad-CAM API"** in the left sidebar
   - Expand it to see all 6 requests

### Step 3: Update Railway URL Variable

1. **Open collection variables:**
   - Click on **"HerbaScan Grad-CAM API"** collection (in left sidebar)
   - Click on **"Variables"** tab (at the top)

2. **Update Railway URL:**
   - Find the `railway_url` variable
   - In the **"Current Value"** column, replace `https://YOUR-RAILWAY-URL.railway.app` with your actual Railway URL
   - Example: `https://herbascan-backend-production.up.railway.app`
   - The variable is automatically saved

3. **Verify variables:**
   - `base_url` should be: `http://localhost:8000`
   - `railway_url` should be: `https://your-actual-railway-url.railway.app`

### Step 4: Test Local Endpoints

1. **Start your local server:**
   ```bash
   cd backend
   python main.py
   ```
   Server should be running on `http://localhost:8000`

2. **Test Health Check (Local):**
   - Click on **"1. Health Check (Local)"** request
   - Click **"Send"** button
   - Expected response:
     ```json
     {
       "status": "healthy",
       "model_loaded": true,
       "labels_loaded": true,
       "num_classes": 40
     }
     ```

3. **Test Root Endpoint:**
   - Click on **"3. Root Endpoint"** request
   - Click **"Send"**
   - Should return API information

4. **Test Identify Plant (Local):**
   - Click on **"5. Identify Plant (Local)"** request
   - Go to **"Body"** tab
   - Under **"form-data"**, find the `file` field
   - **Important**: Click the dropdown next to `file` and change from **"Text"** to **"File"**
   - Click **"Select Files"** and choose a plant image from your computer
   - Click **"Send"**
   - Expected response:
     ```json
     {
       "plant_name": "4Vitex negundo(VN)",
       "scientific_name": "4Vitex negundo(VN)",
       "confidence": 0.942,
       "all_predictions": [...],
       "gradcam_image": "iVBORw0KGgoAAAANSUhEUg...",
       "method": "grad-cam",
       "processing_time_ms": 3456.78
     }
     ```

### Step 5: Test Railway Endpoints

1. **Test Health Check (Railway):**
   - Click on **"2. Health Check (Railway)"** request
   - Click **"Send"**
   - Should return the same response as local (if Railway is deployed)

2. **Test Identify Plant (Railway):**
   - Click on **"6. Identify Plant (Railway)"** request
   - Go to **"Body"** tab
   - Set up file upload (same as local)
   - Click **"Send"**
   - ⚠️ **Note**: If you get a multipart error, use `curl` instead (see troubleshooting below)

---

## 💻 **Method 2: Using VS Code** (REST Client Extension)

### Option A: REST Client Extension

1. **Install REST Client extension:**
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X / Cmd+Shift+X)
   - Search for **"REST Client"** by Huachao Mao
   - Click **"Install"**

2. **Create a REST Client file:**
   - Create a new file: `backend/api-test.http` (or `.rest`)
   - Copy the following content:

   ```http
   ### Variables
   @base_url = http://localhost:8000
   @railway_url = https://YOUR-RAILWAY-URL.railway.app

   ### 1. Health Check (Local)
   GET {{base_url}}/health

   ### 2. Health Check (Railway)
   GET {{railway_url}}/health

   ### 3. Root Endpoint
   GET {{base_url}}/

   ### 4. Test Endpoint
   GET {{base_url}}/test

   ### 5. Identify Plant (Local)
   POST {{base_url}}/identify
   Content-Type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW

   ------WebKitFormBoundary7MA4YWxkTrZu0gW
   Content-Disposition: form-data; name="file"; filename="plant.jpg"
   Content-Type: image/jpeg

   < ./path/to/your/plant/image.jpg
   ------WebKitFormBoundary7MA4YWxkTrZu0gW--

   ### 6. Identify Plant (Railway) - Use curl instead
   # REST Client doesn't handle file uploads well for Railway
   # Use: curl -X POST {{railway_url}}/identify -F "file=@path/to/image.jpg"
   ```

3. **Update variables:**
   - Replace `@railway_url` with your actual Railway URL
   - Update the file path in request 5

4. **Test endpoints:**
   - Click **"Send Request"** above each request
   - Or use Ctrl+Alt+R (Cmd+Alt+R on Mac)
   - View response in a new tab

**Note:** REST Client has limitations with file uploads. For file uploads, use Postman or curl.

### Option B: Thunder Client Extension

1. **Install Thunder Client:**
   - Open VS Code
   - Go to Extensions
   - Search for **"Thunder Client"** by Ranga Vadhineni
   - Click **"Install"**

2. **Import Postman collection:**
   - Click Thunder Client icon in VS Code sidebar
   - Click **"Collections"** tab
   - Click **"New"** → **"Import"**
   - Select **"From File"**
   - Choose `backend/HerbaScan_API.postman_collection.json`
   - Click **"Import"**

3. **Update variables:**
   - Click on collection name
   - Go to **"Variables"** tab
   - Update `railway_url` with your Railway URL

4. **Test endpoints:**
   - Click on any request
   - For file uploads, go to **"Body"** → **"Form"** tab
   - Add field: `file` (type: File)
   - Select your image file
   - Click **"Send"**

---

## 🔧 **Method 3: Using Other IDEs**

### IntelliJ IDEA / PyCharm

1. **Install HTTP Client plugin:**
   - Go to **File → Settings → Plugins**
   - Search for **"HTTP Client"**
   - Install and restart IDE

2. **Create HTTP request file:**
   - Create `backend/api-test.http`
   - Use same format as REST Client (see Method 2, Option A)

3. **Run requests:**
   - Click green play button next to each request
   - Or use Ctrl+Enter

### JetBrains HTTP Client

1. **Create `.http` file:**
   - Create `backend/api-test.http`
   - Use REST Client format (see Method 2, Option A)

2. **Run requests:**
   - Click run button or use keyboard shortcut

---

## 🧪 **Method 4: Using curl (Command Line)**

For quick testing or when Postman has issues:

### Local Testing

```bash
# Health Check
curl http://localhost:8000/health

# Root Endpoint
curl http://localhost:8000/

# Test Endpoint
curl http://localhost:8000/test

# Identify Plant
curl -X POST http://localhost:8000/identify \
  -F "file=@path/to/your/plant/image.jpg"
```

### Railway Testing

```bash
# Health Check
curl https://YOUR-RAILWAY-URL.railway.app/health

# Identify Plant
curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
  -F "file=@path/to/your/plant/image.jpg"
```

**Save response to file:**
```bash
curl -X POST http://localhost:8000/identify \
  -F "file=@image.jpg" \
  -o response.json
```

---

## 📝 **Testing Checklist**

Use this checklist to verify all endpoints work:

### Local Endpoints

- [ ] **Health Check (Local)** - Returns `{"status": "healthy", "model_loaded": true}`
- [ ] **Root Endpoint** - Returns API information
- [ ] **Test Endpoint** - Returns test message
- [ ] **Identify Plant (Local)** - Returns plant identification with Grad-CAM image

### Railway Endpoints

- [ ] **Health Check (Railway)** - Returns healthy status
- [ ] **Identify Plant (Railway)** - Returns plant identification (or use curl if Postman fails)

### Response Verification

For `/identify` endpoint, verify response includes:
- [ ] `plant_name` - Name of identified plant
- [ ] `scientific_name` - Scientific name
- [ ] `confidence` - Confidence score (0-1)
- [ ] `all_predictions` - Array of top predictions
- [ ] `gradcam_image` - Base64 encoded Grad-CAM heatmap
- [ ] `method` - Should be `"grad-cam"`
- [ ] `processing_time_ms` - Processing time in milliseconds

---

## 🐛 **Troubleshooting Postman Issues**

### Issue: "Image bytes are empty"

**Problem:** File not selected or wrong format in Postman

**Solution:**
1. Go to **Body** tab
2. Select **form-data** (not raw or binary)
3. For the `file` field, click dropdown and change from **"Text"** to **"File"**
4. Click **"Select Files"** and choose an actual image file
5. Don't use placeholder text like `path/to/image.jpg`

### Issue: "Misordered multipart fields" (Railway only)

**Problem:** Railway proxy has issues with Postman's multipart format

**Solutions:**
1. **Use curl instead** (recommended for Railway):
   ```bash
   curl -X POST https://YOUR-RAILWAY-URL.railway.app/identify \
     -F "file=@path/to/your/image.jpg"
   ```

2. **Use Postman for local testing only:**
   - Test locally with Postman: `http://localhost:8000/identify`
   - Test Railway with curl

3. **Note:** This doesn't affect Flutter app - Flutter uses `http` package which works fine with Railway

### Issue: Variables not working

**Problem:** Variables like `{{railway_url}}` not replaced

**Solution:**
1. Make sure you're in the collection view
2. Click on collection name → **Variables** tab
3. Set **Current Value** (not just Initial Value)
4. Save the collection
5. Make sure variable name matches exactly (case-sensitive)

### Issue: Can't import collection

**Problem:** JSON file not importing

**Solutions:**
1. **Check file format:**
   - Make sure file is valid JSON
   - File should be `HerbaScan_API.postman_collection.json`

2. **Try different import method:**
   - Use **File → Import** instead of drag-and-drop
   - Or use **Import → Upload Files**

3. **Check Postman version:**
   - Update to latest Postman version
   - Old versions may not support latest collection format

### Issue: Response is empty or error

**Problem:** Server not running or wrong URL

**Solutions:**
1. **Check server is running:**
   ```bash
   # Local server should be running
   python main.py
   # Should see: "Uvicorn running on http://0.0.0.0:8000"
   ```

2. **Check URL:**
   - Local: `http://localhost:8000`
   - Railway: `https://your-url.railway.app`
   - Make sure no trailing slash (except for root endpoint)

3. **Check server logs:**
   - Look at terminal where server is running
   - Check for error messages

### Issue: File upload not working in VS Code REST Client

**Problem:** REST Client has limitations with file uploads

**Solutions:**
1. **Use Thunder Client** instead (better file upload support)
2. **Use Postman desktop app** for file uploads
3. **Use curl** for command-line testing:
   ```bash
   curl -X POST http://localhost:8000/identify \
     -F "file=@path/to/image.jpg"
   ```

---

## 📚 **Additional Resources**

### Postman Documentation

- **Postman Learning Center**: [https://learning.postman.com/](https://learning.postman.com/)
- **Import Collections**: [https://learning.postman.com/docs/getting-started/importing-and-exporting-data/](https://learning.postman.com/docs/getting-started/importing-and-exporting-data/)
- **Using Variables**: [https://learning.postman.com/docs/sending-requests/variables/](https://learning.postman.com/docs/sending-requests/variables/)

### VS Code Extensions

- **REST Client**: [https://marketplace.visualstudio.com/items?itemName=humao.rest-client](https://marketplace.visualstudio.com/items?itemName=humao.rest-client)
- **Thunder Client**: [https://marketplace.visualstudio.com/items?itemName=rangav.vscode-thunder-client](https://marketplace.visualstudio.com/items?itemName=rangav.vscode-thunder-client)

### curl Documentation

- **curl Manual**: [https://curl.se/docs/manual.html](https://curl.se/docs/manual.html)
- **curl File Upload**: [https://curl.se/docs/manual.html#-F](https://curl.se/docs/manual.html#-F)

## 📊 Model Requirements

Your Keras model must have:
- Input shape: `(None, 224, 224, 3)`
- Output shape: `(None, num_classes)`
- At least one convolutional layer for Grad-CAM

**Recommended architecture:**
- MobileNetV2 base + Dense layers
- Last conv layer: `Conv_1` (or specify in code)

## 🐛 Troubleshooting

### Deployment Issues

#### Build Failed

**Check Railway logs:**
1. Go to Railway dashboard
2. Click **"Deployments"**
3. Click the failed deployment
4. Check logs for errors

**Common issues:**
- **Model file not found**: Make sure `mobilenetv2_rf.h5` is committed to git or uploaded to Railway volumes
- **Out of memory**: Railway free tier has 512MB RAM limit. Consider upgrading or optimizing model.
- **TensorFlow installation failed**: Check Dockerfile dependencies
- **Docker build timeout**: Increase build timeout in Railway settings

#### File Upload Errors

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

#### Model Not Loading in Railway

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

4. **Check file permissions:**
   - Ensure model files have read permissions
   - Check Dockerfile copies models correctly

#### Slow Response Times

- First request is always slower (cold start: 10-30 seconds)
- Subsequent requests should be faster (2-4 seconds)
- Consider using Railway Pro for better performance
- Check Railway logs for memory issues

#### CORS Issues

If Flutter app can't connect:
1. Check Railway URL is correct
2. Verify CORS is enabled in `main.py` (it is by default)
3. Try accessing API in browser first
4. Check Flutter app logs for specific error messages

---

### Local Development Issues

#### Model Not Loading

**Symptoms:**
- Server shows `"model_loaded": false` in `/health` endpoint
- Error: "Model file not found" or "Invalid model"

**Solutions:**
1. **Check file path:**
   ```bash
   ls -lh backend/models/mobilenetv2_rf.h5
   # Should show file exists and has reasonable size
   ```

2. **Verify model file is not corrupted:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   print(model.summary())
   ```

3. **Check file permissions:**
   ```bash
   chmod 644 backend/models/mobilenetv2_rf.h5
   ```

4. **Check TensorFlow version:**
   ```bash
   pip show tensorflow
   # Recommended: tensorflow==2.15.0 or tensorflow==2.13.0
   ```

#### Grad-CAM Layer Not Found

**Symptoms:**
- Error: "Layer not found" or "Grad-CAM computation failed"

**Solutions:**
1. **Find your model's last conv layer name:**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('backend/models/mobilenetv2_rf.h5')
   for layer in model.layers:
       if 'conv' in layer.name.lower() or 'Conv2D' in str(type(layer)):
           print(f"{layer.name}: {type(layer).__name__}")
   ```

2. **Update layer name in `main.py`:**
   - Open `backend/utils/gradcam.py`
   - Find the layer name variable
   - Update to match your model's layer name

3. **Check model architecture:**
   - Ensure model has convolutional layers
   - Verify model structure matches expected format

#### Out of Memory

**Symptoms:**
- Error: "Out of memory" or "OOM"
- Server crashes when processing images

**Solutions:**
1. **Reduce image size:**
   - Edit `backend/utils/preprocessing.py`
   - Reduce input image size (e.g., from 224x224 to 192x192)

2. **Use model quantization:**
   - Convert model to quantized TFLite
   - Use smaller model architecture

3. **Increase server memory:**
   - For local: Close other applications
   - For Railway: Upgrade to Pro plan

4. **Optimize Grad-CAM computation:**
   - Reduce batch size
   - Process images one at a time

#### Port Already in Use

**Symptoms:**
- Error: "Address already in use" or "Port 8000 is already in use"

**Solutions:**
1. **Find and kill process using port 8000:**
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # Mac/Linux
   lsof -ti:8000 | xargs kill -9
   ```

2. **Use a different port:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8001
   ```

#### Dependencies Installation Issues

**Symptoms:**
- Error: "Package not found" or "Installation failed"

**Solutions:**
1. **Update pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Use virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   pip install -r requirements.txt
   ```

3. **Check Python version:**
   ```bash
   python --version
   # Recommended: Python 3.8-3.11
   ```

4. **Install specific TensorFlow version:**
   ```bash
   pip install tensorflow==2.15.0
   ```

## 📝 Notes

### Model File Management

- **Model files (`.h5`) are typically NOT committed to git** (too large, >100MB)
- **Options for deployment:**
  - Use Git LFS for models < 100MB
  - Use Railway volumes for large models
  - Upload to cloud storage (S3, GCS) and download on startup
  - Include in Docker image if < 100MB

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

