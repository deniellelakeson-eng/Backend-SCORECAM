"""
HerbaScan Backend API
FastAPI server for true Grad-CAM computation using TensorFlow
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
import json
import base64
import io
import time
from PIL import Image
from pathlib import Path

from utils.gradcam import generate_gradcam_for_image
from utils.preprocessing import preprocess_image, array_to_pil_image

# Initialize FastAPI app
app = FastAPI(
    title="HerbaScan Grad-CAM API",
    description="True gradient-based plant identification with explainable AI",
    version="1.0.0"
)

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model (loaded once at startup)
model = None
labels = None

# Model paths: Check Railway volume first, then local models/ directory
# Railway volume is mounted at /models if you create a volume named "models"
RAILWAY_VOLUME_PATH = Path("/models/mobilenetv2_rf.h5")
LOCAL_MODEL_PATH = Path("models/mobilenetv2_rf.h5")
MODEL_PATH = None  # Will be determined at startup
LABELS_PATH = Path("models/labels.json")

# Initialize labels as empty dict to prevent None errors
labels = {}


@app.on_event("startup")
async def load_model():
    """Load Keras model and labels on server startup."""
    global model, labels, MODEL_PATH
    
    try:
        # Determine model path: Check Railway volume first, then local
        if RAILWAY_VOLUME_PATH.exists():
            MODEL_PATH = RAILWAY_VOLUME_PATH
            print("📍 Using model from Railway Volume")
        else:
            MODEL_PATH = LOCAL_MODEL_PATH
            print("📍 Using model from local models/ directory")
        
        print("=" * 50)
        print("🌿 HerbaScan API Starting Up...")
        print("=" * 50)
        print(f"📂 Working directory: {Path.cwd()}")
        print(f"📂 Model path: {MODEL_PATH.absolute()}")
        print(f"📂 Labels path: {LABELS_PATH.absolute()}")
        print(f"📂 Model exists: {MODEL_PATH.exists()}")
        print(f"📂 Labels exists: {LABELS_PATH.exists()}")
        print("=" * 50)
        
        # Initialize labels first (even if model fails)
        if LABELS_PATH.exists():
            with open(LABELS_PATH, 'r') as f:
                labels = json.load(f)
            print(f"✅ Labels loaded: {len(labels)} classes")
        else:
            print(f"⚠️  Labels file not found at: {LABELS_PATH}")
            print("📝 Creating dummy labels (40 classes for HerbaScan)")
            labels = {str(i): f"Plant_{i}" for i in range(40)}
        
        # Check if model file exists
        if not MODEL_PATH.exists():
            print(f"⚠️  Model file not found at: {MODEL_PATH}")
            print("📝 Server will start, but /identify endpoint will not work")
            print("💡 To fix: Upload model file using Railway Volumes or Git LFS")
            return
        
        print("🔄 Loading Keras model (this may take 30-60 seconds)...")
        
        # Load model
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        print("=" * 50)
        print("✅ Server ready to accept requests!")
        print("=" * 50)
        
    except Exception as e:
        import traceback
        print(f"❌ Error loading model: {str(e)}")
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        print("📝 Server will start, but /identify endpoint will not work")
        # Ensure labels are set even on error
        if labels is None:
            labels = {str(i): f"Plant_{i}" for i in range(40)}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HerbaScan Grad-CAM API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "identify": "/identify (POST)",
            "test": "/test"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_loaded": labels is not None,
        "num_classes": len(labels) if labels else 0
    }


@app.get("/test")
async def test_endpoint():
    """Test endpoint for debugging."""
    return {
        "message": "HerbaScan API is working!",
        "model_status": "loaded" if model is not None else "not loaded",
        "model_path": str(MODEL_PATH) if MODEL_PATH else "not determined yet",
        "model_exists": MODEL_PATH.exists() if MODEL_PATH else False,
        "railway_volume_exists": RAILWAY_VOLUME_PATH.exists(),
        "local_model_exists": LOCAL_MODEL_PATH.exists(),
        "labels_count": len(labels) if labels else 0
    }


@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    """
    Main endpoint: Identify plant and generate Grad-CAM visualization.
    
    Args:
        file: Uploaded image file (multipart/form-data)
    
    Returns:
        JSON with:
            - plant_name: Identified plant name
            - scientific_name: Scientific name (if available)
            - confidence: Confidence score (0-1)
            - all_predictions: Top 3 predictions with confidence
            - gradcam_image: Base64-encoded Grad-CAM overlay image
            - method: "grad-cam"
            - processing_time_ms: Processing time in milliseconds
    """
    start_time = time.time()
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Read uploaded image
        image_bytes = await file.read()
        
        # Preprocess image for model
        img_array = preprocess_image(image_bytes, target_size=(224, 224))
        
        # Load original image for overlay
        original_image = Image.open(io.BytesIO(image_bytes))
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Run inference
        predictions = model.predict(img_array, verbose=0)
        predictions = predictions[0]  # Remove batch dimension
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        
        # Get predicted class (highest confidence)
        predicted_class_idx = top_3_indices[0]
        predicted_class_name = labels.get(str(predicted_class_idx), f"Plant_{predicted_class_idx}")
        confidence = float(predictions[predicted_class_idx])
        
        # Generate Grad-CAM
        gradcam_result = generate_gradcam_for_image(
            model=model,
            img_array=img_array,
            original_image=original_image,
            class_idx=predicted_class_idx,
            layer_name='Conv_1'  # Adjust based on your model architecture
        )
        
        # Convert overlay image to base64
        overlay_img = gradcam_result['overlay']
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        gradcam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare all predictions
        all_predictions = []
        for idx in top_3_indices:
            all_predictions.append({
                "class": labels.get(str(idx), f"Plant_{idx}"),
                "class_index": int(idx),
                "confidence": float(predictions[idx])
            })
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response
        response = {
            "plant_name": predicted_class_name,
            "scientific_name": predicted_class_name,  # Add mapping if available
            "confidence": confidence,
            "all_predictions": all_predictions,
            "gradcam_image": gradcam_base64,
            "method": "grad-cam",
            "processing_time_ms": round(processing_time, 2)
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    import os
    
    print("🌿 Starting HerbaScan Grad-CAM API...")
    print(f"📂 Railway volume path: {RAILWAY_VOLUME_PATH}")
    print(f"📂 Local model path: {LOCAL_MODEL_PATH}")
    print(f"📂 Labels path: {LABELS_PATH}")
    
    # Railway provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

