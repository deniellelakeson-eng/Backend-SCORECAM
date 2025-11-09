"""
Phase 2.1: Extract CAM Weights from Keras Model

This script extracts the weights from the final dense/classification layer
of your trained model. These weights are needed for offline CAM computation.

Output: cam_weights.json (shape: [1280, 40] for 40 plant classes)
"""

import tensorflow as tf
import json
import numpy as np
from pathlib import Path

# Configuration
MODEL_PATH = Path("models/mobilenetv2_rf.h5")
OUTPUT_PATH = Path("models/cam_weights.json")

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

def extract_cam_weights():
    """Extract CAM weights from the model."""
    print("=" * 60)
    print("Phase 2.1: Extracting CAM Weights")
    print("=" * 60)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at: {MODEL_PATH}")
        print(f"Please place mobilenetv2_rf.h5 in the models/ directory")
        return False
    
    try:
        # Load model
        print(f"\n[1/4] Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(str(MODEL_PATH))
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
            'feature_dim': weights.shape[0]
        }
        
        # Save to JSON
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Calculate file size
        file_size = OUTPUT_PATH.stat().st_size / 1024  # KB
        print(f"      Saved to: {OUTPUT_PATH}")
        print(f"      File size: {file_size:.2f} KB")
        
        print("\n" + "=" * 60)
        print("SUCCESS: CAM weights extracted!")
        print("=" * 60)
        print(f"  Layer: {classification_layer.name}")
        print(f"  Shape: {weights.shape[0]} features x {weights.shape[1]} classes")
        print(f"  Output: {OUTPUT_PATH}")
        print("\nNext step: Run create_multi_output_tflite.py")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = extract_cam_weights()
    exit(0 if success else 1)

