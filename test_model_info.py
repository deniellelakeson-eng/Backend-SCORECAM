"""
Quick script to check model architecture and find the last convolutional layer.
Run this to verify model configuration before deployment.
"""

import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path("models/mobilenetv2_rf.h5")

try:
    print("=" * 60)
    print("🔍 HerbaScan Model Information")
    print("=" * 60)
    
    # Load model
    print(f"\n📂 Loading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("✅ Model loaded successfully!\n")
    
    # Basic model info
    print("📊 Model Summary:")
    print(f"   • Total layers: {len(model.layers)}")
    print(f"   • Input shape: {model.input_shape}")
    print(f"   • Output shape: {model.output_shape}")
    
    # Find convolutional layers
    print("\n🔍 Convolutional Layers (for Grad-CAM):")
    conv_layers = []
    for i, layer in enumerate(model.layers):
        if 'conv' in layer.name.lower():
            conv_layers.append((i, layer.name, layer.output_shape))
            print(f"   [{i}] {layer.name:40s} → {layer.output_shape}")
    
    if conv_layers:
        last_conv = conv_layers[-1]
        print(f"\n✅ RECOMMENDED LAYER FOR GRAD-CAM:")
        print(f"   Layer Name: '{last_conv[1]}'")
        print(f"   Layer Index: {last_conv[0]}")
        print(f"   Output Shape: {last_conv[2]}")
        print(f"\n💡 Use this in main.py: layer_name='{last_conv[1]}'")
    else:
        print("\n⚠️  No convolutional layers found!")
    
    # Full model summary
    print("\n" + "=" * 60)
    print("📋 Full Model Architecture:")
    print("=" * 60)
    model.summary()
    
    print("\n" + "=" * 60)
    print("✅ Model check complete!")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\n💡 Make sure mobilenetv2_rf.h5 is in the models/ directory")


