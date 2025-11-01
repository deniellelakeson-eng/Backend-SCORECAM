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

### Option A: Deploy via GitHub

1. **Create a new GitHub repository** for this backend
2. **Push code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: HerbaScan Backend API"
   git remote add origin https://github.com/yourusername/herbascan-backend.git
   git push -u origin main
   ```

3. **Connect to Railway:**
   - Go to [Railway.app](https://railway.app/)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Select your backend repository
   - Railway will automatically detect the Dockerfile

4. **Add model files to Railway:**
   - Since model files are too large for GitHub, you'll need to:
     - Option A: Use Railway volumes
     - Option B: Upload to cloud storage (S3, GCS) and download on startup
     - Option C: Use Railway's file storage

5. **Get your deployment URL:**
   - Railway will provide a URL like: `https://herbascan-api.railway.app`
   - Update Flutter app to use this URL

### Option B: Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
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

### **Option 1: Import Postman Collection** (Easiest)

1. **Import the collection:**
   - Open Postman
   - Click "Import"
   - Select `HerbaScan_API.postman_collection.json`
   - Collection includes all endpoints (local + Railway)

2. **Update Railway URL:**
   - Click on "HerbaScan Grad-CAM API" collection
   - Go to "Variables" tab
   - Update `railway_url` with your actual Railway URL

3. **Test endpoints:**
   - Start with "Health Check"
   - Then try "Identify Plant"
   - Check response for `confidence`, `gradcam_image`, `processing_time_ms`

### **Option 2: Manual Setup**

1. **Create New Request:**
   - Method: `POST`
   - URL: `http://localhost:8000/identify`
2. **Set Body:**
   - Type: `form-data`
   - Key: `file` (change type to "File")
   - Value: Select a plant image
3. **Send and verify response**

## 📊 Model Requirements

Your Keras model must have:
- Input shape: `(None, 224, 224, 3)`
- Output shape: `(None, num_classes)`
- At least one convolutional layer for Grad-CAM

**Recommended architecture:**
- MobileNetV2 base + Dense layers
- Last conv layer: `Conv_1` (or specify in code)

## 🐛 Troubleshooting

### Model not loading
- Check if `mobilenetv2_rf.h5` is in `models/` directory
- Verify model file is not corrupted
- Check file permissions

### Grad-CAM layer not found
- Update `layer_name` in `main.py` (line 137)
- Find your model's last conv layer name:
  ```python
  model.summary()
  ```

### Out of memory
- Reduce image size in `preprocess_image()`
- Use model quantization
- Increase server memory allocation

## 📝 Notes

- Model file (`.h5`) is NOT committed to git (too large)
- Add model files manually to server after deployment
- For production, consider using cloud storage for models
- TensorFlow 2.15.0 is recommended for compatibility

## 🔗 Related Documentation

- FastAPI: https://fastapi.tiangolo.com/
- TensorFlow: https://www.tensorflow.org/
- Railway: https://railway.app/
- Grad-CAM Paper: https://arxiv.org/abs/1610.02391

## 📞 Support

For issues or questions, refer to:
- Main project: `../README.md`
- Grad-CAM Architecture: `../GRADCAM_ARCHITECTURE.md`
- Implementation Plan: `../GRADCAM_HYBRID_IMPLEMENTATION_PLAN.md`

