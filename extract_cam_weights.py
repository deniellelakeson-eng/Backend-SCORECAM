"""
Phase 2.1: Extract CAM Weights from MobileNetV2 Model

This script extracts the weights from the final dense/classification layer
of your trained MobileNetV2 model. These weights are needed for offline CAM computation.

Output: mobilenetv2_cam_weights.json (shape: [1280, 42] for 42 plant classes)

Note: HerbaScan custom model is deprecated - only MobileNetV2 is supported.
"""

import tensorflow as tf
import json
import numpy as np
from pathlib import Path

# Configuration
# Try .keras models first, then fallback to .h5
# Handle both running from backend/ directory and project root
_script_dir = Path(__file__).parent
_backend_models_dir = _script_dir / "models"
_root_models_dir = _script_dir.parent / "backend" / "models"

# Try backend/models first (when run from backend/), then root/backend/models (when run from root)
if _backend_models_dir.exists():
    models_dir = _backend_models_dir
else:
    models_dir = _root_models_dir

MOBILENETV2_MODEL_PATH = models_dir / "MobileNetV2_model.keras"
HERBASCAN_MODEL_PATH = models_dir / "herbascan_model.keras"
MOBILENETV2_OUTPUT_PATH = models_dir / "mobilenetv2_cam_weights.json"
HERBASCAN_OUTPUT_PATH = models_dir / "herbascan_cam_weights.json"

def find_classification_layer(model):
    """Find the final dense/classification layer."""
    # Common names for classification layers
    possible_names = [
        'predictions',
        'dense',
        'dense_1',
        'fc',
        'classifier',
        'output',
        'softmax',
        'dense_final'
    ]
    
    # Search for the layer
    for layer in reversed(model.layers):
        if any(name in layer.name.lower() for name in possible_names):
            return layer
    
    # If not found, get the last layer that has weights
    for layer in reversed(model.layers):
        if len(layer.get_weights()) > 0:
            return layer
    
    raise ValueError("Could not find classification layer!")

def extract_cam_weights_for_model(model_path, model_name, output_path):
    """Extract CAM weights from a single model."""
    try:
        # Load model
        print(f"\n[1/4] Loading {model_name} model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        print("      Model loaded successfully!")
        
        # Find classification layer
        print(f"\n[2/4] Finding classification layer...")
        classification_layer = find_classification_layer(model)
        print(f"      Found layer: '{classification_layer.name}'")
        print(f"      Layer type: {type(classification_layer).__name__}")
        
        # Get weights
        print(f"\n[3/4] Extracting weights...")
        layer_weights = classification_layer.get_weights()
        
        if len(layer_weights) == 0:
            print("      ERROR: Layer has no weights!")
            return False
        
        # Get weight matrix (first element is usually the weight matrix)
        weights = layer_weights[0]  # Shape: [input_features, num_classes]
        print(f"      Weight shape: {weights.shape}")
        print(f"      Weight dtype: {weights.dtype}")
        
        # Get bias if available
        bias = None
        if len(layer_weights) > 1:
            bias = layer_weights[1].tolist()
            print(f"      Bias shape: {layer_weights[1].shape}")
        
        # Convert to list for JSON
        print(f"\n[4/4] Saving to JSON...")
        output_data = {
            'weights': weights.tolist(),
            'shape': list(weights.shape),
            'layer_name': classification_layer.name,
            'layer_type': type(classification_layer).__name__,
            'bias': bias,
            'num_classes': weights.shape[1],
            'feature_dim': weights.shape[0],
            'model_name': model_name
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Calculate file size
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"      Saved to: {output_path}")
        print(f"      File size: {file_size:.2f} KB")
        
        print("\n" + "=" * 60)
        print(f"SUCCESS: {model_name} CAM weights extracted!")
        print("=" * 60)
        print(f"  Layer: {classification_layer.name}")
        print(f"  Shape: {weights.shape[0]} features x {weights.shape[1]} classes")
        print(f"  Output: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR processing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def extract_cam_weights():
    """Extract CAM weights from MobileNetV2 model (HerbaScan deprecated)."""
    print("=" * 60)
    print("Phase 2.1: Extracting CAM Weights")
    print("Extracting from MobileNetV2 model (HerbaScan model deprecated)")
    print("=" * 60)
    
    results = []
    
    # Process MobileNetV2 model
    if MOBILENETV2_MODEL_PATH.exists():
        print("\n" + "=" * 60)
        print("Processing MobileNetV2 Model")
        print("=" * 60)
        success = extract_cam_weights_for_model(
            MOBILENETV2_MODEL_PATH,
            "MobileNetV2",
            MOBILENETV2_OUTPUT_PATH
        )
        results.append(("MobileNetV2", success))
    else:
        print(f"\n⚠️  MobileNetV2 model not found at: {MOBILENETV2_MODEL_PATH}")
        print("   Skipping MobileNetV2 CAM weights extraction")
        results.append(("MobileNetV2", False))
    
    # Process HerbaScan model (DEPRECATED - kept for backward compatibility)
    if HERBASCAN_MODEL_PATH.exists():
        print("\n" + "=" * 60)
        print("Processing HerbaScan Model (DEPRECATED)")
        print("=" * 60)
        print("⚠️  WARNING: HerbaScan model is deprecated. Use MobileNetV2 only.")
        success = extract_cam_weights_for_model(
            HERBASCAN_MODEL_PATH,
            "HerbaScan",
            HERBASCAN_OUTPUT_PATH
        )
        results.append(("HerbaScan (Deprecated)", success))
    else:
        print(f"\n⚠️  HerbaScan model not found at: {HERBASCAN_MODEL_PATH}")
        print("   HerbaScan model is deprecated - MobileNetV2 is the only supported model")
        results.append(("HerbaScan (Deprecated)", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: CAM Weights Extraction")
    print("=" * 60)
    for model_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED/SKIPPED"
        print(f"  {model_name}: {status}")
    
    # Check if MobileNetV2 succeeded (required)
    mobilenetv2_success = results[0][1] if results and results[0][0] == "MobileNetV2" else False
    
    if mobilenetv2_success:
        print("\n✅ MobileNetV2 CAM weights extracted successfully!")
        print("Next step: Run create_multi_output_tflite.py")
        return True
    else:
        print("\n❌ ERROR: MobileNetV2 model processing failed!")
        print(f"Please ensure MobileNetV2_model.keras exists in: {models_dir}")
        print("Note: HerbaScan model is deprecated - MobileNetV2 is required.")
        return False

if __name__ == "__main__":
    success = extract_cam_weights()
    exit(0 if success else 1)

