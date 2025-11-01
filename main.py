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
import os
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
MODEL_PATH = Path("models/mobilenetv2_rf.h5")
LABELS_PATH = Path("models/labels.json")


@app.on_event("startup")
async def load_model():
    """Load Keras model and labels on server startup."""
    global model, labels
    
    try:
        print("🔄 Loading Keras model...")
        
        # Check if model file exists
        if not MODEL_PATH.exists():
            print(f"⚠️  Model file not found at: {MODEL_PATH}")
            print("📝 Please place your mobilenetv2_rf.h5 file in the models/ directory")
            return
        
        # Load model
        model = tf.keras.models.load_model(str(MODEL_PATH))
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
        
        # Load labels
        if LABELS_PATH.exists():
            with open(LABELS_PATH, 'r') as f:
                labels = json.load(f)
            print(f"✅ Labels loaded: {len(labels)} classes")
        else:
            print(f"⚠️  Labels file not found at: {LABELS_PATH}")
            print("📝 Please place your labels.json file in the models/ directory")
            # Create dummy labels (40 classes for HerbaScan)
            labels = {str(i): f"Plant_{i}" for i in range(40)}
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("📝 Server will start, but /identify endpoint will not work")


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
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "labels_count": len(labels) if labels else 0
    }


@app.post("/identify")
async def identify_plant(file: UploadFile):
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
        
        # Log for debugging
        print(f"DEBUG: Received file '{file.filename}', content_type: {file.content_type}, bytes type: {type(image_bytes)}, bytes len: {len(image_bytes) if isinstance(image_bytes, bytes) else 'N/A'}")
        
        # Ensure we have bytes
        if not isinstance(image_bytes, bytes):
            raise HTTPException(
                status_code=400,
                detail=f"Expected bytes, got {type(image_bytes)}"
            )
        
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
    
    # Read PORT from environment variable (Railway provides this)
    port = int(os.environ.get("PORT", 8000))
    
    print("🌿 Starting HerbaScan Grad-CAM API...")
    print(f"📂 Model path: {MODEL_PATH}")
    print(f"📂 Labels path: {LABELS_PATH}")
    print(f"🌐 Starting on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

