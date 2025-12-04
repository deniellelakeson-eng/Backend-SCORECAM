"""
Quick script to check model architecture and find the last convolutional layer.
Run this to verify model configuration before deployment.
"""

import tensorflow as tf
from pathlib import Path

# Try .keras models first, fallback to .h5
MOBILENETV2_PATH = Path("models/MobileNetV2_model.keras")
HERBASCAN_PATH = Path("models/herbascan_model.keras")
LEGACY_MODEL_PATH = Path("models/mobilenetv2_rf.h5")

try:
    print("=" * 60)
    print("🔍 HerbaScan Model Information")
    print("=" * 60)
    
    # Try to load .keras models first
    model = None
    model_path = None
    
    if MOBILENETV2_PATH.exists():
        print(f"\n📂 Loading MobileNetV2 model from: {MOBILENETV2_PATH}")
        model = tf.keras.models.load_model(str(MOBILENETV2_PATH))
        model_path = MOBILENETV2_PATH
        print("✅ MobileNetV2 model loaded successfully!\n")
    elif HERBASCAN_PATH.exists():
        print(f"\n📂 Loading HerbaScan model from: {HERBASCAN_PATH}")
        model = tf.keras.models.load_model(str(HERBASCAN_PATH))
        model_path = HERBASCAN_PATH
        print("✅ HerbaScan model loaded successfully!\n")
    elif LEGACY_MODEL_PATH.exists():
        print(f"\n📂 Loading legacy model from: {LEGACY_MODEL_PATH}")
        model = tf.keras.models.load_model(str(LEGACY_MODEL_PATH))
        model_path = LEGACY_MODEL_PATH
        print("✅ Legacy model loaded successfully!\n")
    else:
        raise FileNotFoundError("No model files found. Please ensure at least one .keras or .h5 model exists in models/")
    
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
    print("\n💡 Make sure at least one model file exists:")
    print("   - models/MobileNetV2_model.keras")
    print("   - models/herbascan_model.keras")
    print("   - models/mobilenetv2_rf.h5 (legacy)")


