"""
HerbaScan Backend API
FastAPI server for true Score-CAM computation using TensorFlow
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

from utils.scorecam import generate_scorecam_for_image
from utils.preprocessing import preprocess_image, array_to_pil_image

# Initialize FastAPI app
app = FastAPI(
    title="HerbaScan Score-CAM API",
    description="True score-weighted plant identification with explainable AI",
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

# Global variables for models (loaded once at startup)
mobilenetv2_model = None
labels = None
MOBILENETV2_MODEL_PATH = Path("models/MobileNetV2_model.keras")
LABELS_PATH = Path("models/labels.json")


@app.on_event("startup")
async def load_models():
    """Load MobileNetV2 Keras model and labels on server startup."""
    global mobilenetv2_model, labels
    
    try:
        print("🔄 Loading MobileNetV2 Keras model...")
        
        # Load MobileNetV2 model (ONLY MODEL - HerbaScan deprecated)
        if MOBILENETV2_MODEL_PATH.exists():
            try:
                mobilenetv2_model = tf.keras.models.load_model(str(MOBILENETV2_MODEL_PATH))
                print(f"✅ MobileNetV2 model loaded successfully from {MOBILENETV2_MODEL_PATH}")
            except Exception as e:
                print(f"❌ Error loading MobileNetV2 model: {str(e)}")
                print("❌ Server will start, but /identify endpoint will not work")
                return
        else:
            print(f"❌ MobileNetV2 model file not found at: {MOBILENETV2_MODEL_PATH}")
            print("❌ Please ensure MobileNetV2_model.keras exists in models/ directory")
            return
        
        # Check if model is loaded
        if mobilenetv2_model is None:
            print("❌ MobileNetV2 model failed to load! Please check the model file.")
            return
        
        # Load labels
        if LABELS_PATH.exists():
            with open(LABELS_PATH, 'r') as f:
                labels = json.load(f)
            print(f"✅ Labels loaded: {len(labels)} classes")
        else:
            print(f"⚠️  Labels file not found at: {LABELS_PATH}")
            print("📝 Please place your labels.json file in the models/ directory")
            # Create dummy labels (42 classes for HerbaScan)
            labels = {str(i): f"Plant_{i}" for i in range(42)}
        
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        print("📝 Server will start, but /identify endpoint will not work")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HerbaScan Score-CAM API",
        "version": "1.0.0",
        "status": "running",
        "mobilenetv2_loaded": mobilenetv2_model is not None,
        "models_loaded": mobilenetv2_model is not None,
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
        "mobilenetv2_loaded": mobilenetv2_model is not None,
        "models_loaded": mobilenetv2_model is not None,
        "labels_loaded": labels is not None,
        "num_classes": len(labels) if labels else 0
    }


@app.get("/test")
async def test_endpoint():
    """Test endpoint for debugging."""
    return {
        "message": "HerbaScan API is working!",
        "mobilenetv2_status": "loaded" if mobilenetv2_model is not None else "not loaded",
        "mobilenetv2_path": str(MOBILENETV2_MODEL_PATH),
        "mobilenetv2_exists": MOBILENETV2_MODEL_PATH.exists(),
        "labels_count": len(labels) if labels else 0
    }


@app.post("/identify")
async def identify_plant(file: UploadFile = File(...)):
    """
    Main endpoint: Identify plant and generate Score-CAM visualization.
    
    Args:
        file: Uploaded image file (multipart/form-data)
    
    Returns:
        JSON with:
            - plant_name: Identified plant name
            - scientific_name: Scientific name (if available)
            - confidence: Confidence score (0-1)
            - all_predictions: Top 3 predictions with confidence
            - gradcam_image: Base64-encoded Score-CAM overlay image
            - method: "score-cam"
            - processing_time_ms: Processing time in milliseconds
    """
    start_time = time.time()
    
    # Check if MobileNetV2 model is loaded
    if mobilenetv2_model is None:
        raise HTTPException(
            status_code=503,
            detail="MobileNetV2 model not loaded. Please check server logs."
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
        
        # Run inference on MobileNetV2 model (ONLY MODEL - HerbaScan deprecated)
        try:
            preds = mobilenetv2_model.predict(img_array, verbose=0)
            preds = preds[0]  # Remove batch dimension
            max_idx = np.argmax(preds)
            max_conf = float(preds[max_idx])
            
            best_predictions = preds
            best_model = mobilenetv2_model
            best_confidence = max_conf
            best_class_idx = max_idx
            model_name_used = "MobileNetV2"
            print(f"📊 MobileNetV2 prediction: class {max_idx}, confidence {max_conf:.4f}")
        except Exception as e:
            print(f"❌ Error running MobileNetV2 inference: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to run inference: {str(e)}"
            )
        
        if best_predictions is None or best_model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to run inference on MobileNetV2 model"
            )
        
        print(f"✅ Using MobileNetV2 model (confidence: {best_confidence:.4f})")
        
        # Get top 3 predictions
        top_3_indices = np.argsort(best_predictions)[-3:][::-1]
        
        # Helper function to get plant name from index
        # labels.json format: {"PlantName": index}
        def get_plant_name_from_index(idx):
            for plant_name, plant_idx in labels.items():
                if plant_idx == idx:
                    return plant_name
            return f"Plant_{idx}"
        
        # Get predicted class (highest confidence)
        predicted_class_idx = best_class_idx
        predicted_class_name = get_plant_name_from_index(predicted_class_idx)
        confidence = best_confidence
        
        # Generate Score-CAM using the best model (auto-detects last conv layer)
        scorecam_result = generate_scorecam_for_image(
            model=best_model,
            img_array=img_array,
            original_image=original_image,
            class_idx=predicted_class_idx,
            layer_name=None  # Auto-detect last conv layer
        )
        
        # Convert overlay image to base64
        overlay_img = scorecam_result['overlay']
        buffered = io.BytesIO()
        overlay_img.save(buffered, format="PNG")
        scorecam_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare all predictions (format for frontend)
        all_predictions = []
        predictions = []  # Frontend expects 'predictions' key
        for idx in top_3_indices:
            plant_name = get_plant_name_from_index(int(idx))
            pred_data = {
                "label": plant_name,
                "plantName": plant_name,  # UI expects 'plantName'
                "scientificName": plant_name,  # Add mapping if available
                "class_index": int(idx),
                "index": int(idx),
                "confidence": float(best_predictions[idx]),
                "isDOHApproved": False  # Add lookup logic later if needed
            }
            all_predictions.append({
                "class": plant_name,
                "class_index": int(idx),
                "confidence": float(best_predictions[idx])
            })
            predictions.append(pred_data)  # Frontend format
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Prepare response
        response = {
            "plant_name": predicted_class_name,
            "scientific_name": predicted_class_name,  # Add mapping if available
            "confidence": confidence,
            "predictions": predictions,  # Frontend expects this key
            "all_predictions": all_predictions,  # Keep for backward compatibility
            "gradcam_image": scorecam_base64,  # Keep key name for backward compatibility
            "method": "score-cam",
            "model_used": model_name_used,
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
    
    print("🌿 Starting HerbaScan Score-CAM API...")
    print(f"📂 MobileNetV2 model path: {MOBILENETV2_MODEL_PATH}")
    print(f"📂 Labels path: {LABELS_PATH}")
    print(f"🌐 Starting on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

