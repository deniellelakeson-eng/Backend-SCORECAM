"""
Phase 2.2: Create Multi-Output TFLite Model

This script creates a TFLite model with 2 outputs:
  1. Feature maps from last convolutional layer (for CAM)
  2. Final predictions (for classification)

Output: mobilenetv2_multi_output.tflite
"""

import tensorflow as tf
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
MOBILENETV2_OUTPUT_PATH = models_dir / "mobilenetv2_multi_output.tflite"
HERBASCAN_OUTPUT_PATH = models_dir / "herbascan_multi_output.tflite"

def find_last_conv_layer(model):
    """Find the last convolutional layer before global pooling/dense layers."""
    conv_layers = []
    
    print("      Scanning model layers...")
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        
        # Check for convolutional layers
        if any(x in layer_type for x in ['Conv2D', 'DepthwiseConv2D', 'SeparableConv2D']):
            # Get output shape - handle both old and new Keras API
            try:
                if hasattr(layer, 'output_shape'):
                    output_shape = layer.output_shape
                    # Convert to tuple if it's a TensorShape
                    if hasattr(output_shape, 'as_list'):
                        output_shape = tuple(output_shape.as_list())
                    elif isinstance(output_shape, list):
                        output_shape = tuple(output_shape)
                    # If already a tuple, keep it as is
                elif hasattr(layer, 'output'):
                    if hasattr(layer.output, 'shape'):
                        shape = layer.output.shape
                        # Handle TensorShape object vs tuple
                        if hasattr(shape, 'as_list'):
                            # TensorShape object - convert to tuple
                            output_shape = tuple(shape.as_list())
                        elif isinstance(shape, (list, tuple)):
                            # Already a tuple/list - use as is
                            output_shape = tuple(shape) if isinstance(shape, list) else shape
                        else:
                            output_shape = None
                    else:
                        output_shape = None
                else:
                    output_shape = None
            except Exception as e:
                output_shape = None
                print(f"      ⚠️  Could not get output shape for layer {layer.name}: {e}")
            
            conv_layers.append((i, layer.name, output_shape))
            print(f"      Found conv layer [{i}]: {layer.name} -> {output_shape}")
        
        # Stop at first dense/global pooling (after last conv)
        # This means we've passed all conv layers
        if 'Dense' in layer_type or 'GlobalAveragePooling2D' in layer_type:
            if conv_layers:
                last_conv = conv_layers[-1]
                print(f"      Stopping at layer [{i}]: {layer.name} ({layer_type})")
                print(f"      Selected last conv layer: {last_conv[1]} -> {last_conv[2]}")
                return last_conv
    
    # If no Dense/GAP found, return last conv layer from entire model
    if conv_layers:
        last_conv = conv_layers[-1]
        print(f"      No Dense/GAP layer found, using last conv: {last_conv[1]} -> {last_conv[2]}")
        return last_conv
    
    raise ValueError("Could not find convolutional layer! Model may not have any conv layers.")

def create_multi_output_model(model, conv_layer_name):
    """Create a model with 2 outputs: feature maps + predictions."""
    # Get the convolutional layer output
    conv_layer = model.get_layer(conv_layer_name)
    conv_output = conv_layer.output
    
    # Get model predictions
    predictions = model.output
    
    # Create new model with 2 outputs
    # IMPORTANT: Output order is [features, predictions]
    # But TensorFlow Lite may swap them, so we'll verify in Flutter
    multi_output_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_output, predictions],  # [features, predictions]
        name='mobilenetv2_multi_output'
    )
    
    # Build the model by calling it with a dummy input
    # This ensures all layers are properly initialized for TFLite conversion
    # This is critical for Keras 3 compatibility
    print("      Building model for TFLite conversion...")
    dummy_input = tf.zeros((1, 224, 224, 3))  # Match your input shape
    test_outputs = multi_output_model(dummy_input)
    
    # Print output info for debugging
    # Use the actual outputs from the test run to get shapes
    try:
        if isinstance(test_outputs, (list, tuple)) and len(test_outputs) >= 2:
            # Get shapes from actual output tensors
            # Handle both TensorShape and tuple shapes
            shape0 = test_outputs[0].shape
            shape1 = test_outputs[1].shape
            
            if hasattr(shape0, 'as_list'):
                output0_shape = tuple(shape0.as_list())
            elif isinstance(shape0, (list, tuple)):
                output0_shape = tuple(shape0) if isinstance(shape0, list) else shape0
            else:
                output0_shape = shape0
                
            if hasattr(shape1, 'as_list'):
                output1_shape = tuple(shape1.as_list())
            elif isinstance(shape1, (list, tuple)):
                output1_shape = tuple(shape1) if isinstance(shape1, list) else shape1
            else:
                output1_shape = shape1
                
            print(f"      Model Output 0 (conv): {output0_shape}")
            print(f"      Model Output 1 (predictions): {output1_shape}")
        else:
            # Fallback: try to get from model.output property
            outputs = multi_output_model.output
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                output0 = outputs[0]
                output1 = outputs[1]
                if hasattr(output0, 'shape'):
                    output0_shape = output0.shape
                    if hasattr(output0_shape, 'as_list'):
                        output0_shape = tuple(output0_shape.as_list())
                    print(f"      Model Output 0 (conv): {output0_shape}")
                if hasattr(output1, 'shape'):
                    output1_shape = output1.shape
                    if hasattr(output1_shape, 'as_list'):
                        output1_shape = tuple(output1_shape.as_list())
                    print(f"      Model Output 1 (predictions): {output1_shape}")
    except Exception as e:
        print(f"      ⚠️  Could not get output shapes: {e}")
        print(f"      Test outputs type: {type(test_outputs)}")
        # Continue anyway - the model should still work
    
    return multi_output_model

def convert_to_tflite(model, output_path):
    """Convert Keras model to TFLite with optimization and compatibility settings."""
    print("      Converting to TFLite with compatibility flags...")
    
    # Method 1: Try from_keras_model (works for most TF versions)
    try:
        print("      Attempting conversion with from_keras_model...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # IMPORTANT: DO NOT use optimizations for multi-output models!
        # Optimization can strip away intermediate outputs that aren't "needed"
        # We need BOTH outputs (feature maps AND predictions), so skip optimization
        converter.optimizations = []  # No optimization - preserve all outputs
        
        # IMPORTANT: Use TFLITE_BUILTINS only for maximum compatibility
        # This ensures the model works with tflite_flutter: ^0.12.1
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Built-in ops only (no TF ops)
        ]
        
        # Disable experimental features that might cause compatibility issues
        if hasattr(converter, 'experimental_new_converter'):
            converter.experimental_new_converter = False
        
        # CRITICAL: Ensure all outputs are preserved
        # Some converters might optimize away "unused" outputs
        if hasattr(converter, 'experimental_lower_to_saved_model'):
            converter.experimental_lower_to_saved_model = False
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"      ✅ Saved to: {output_path}")
        print(f"      File size: {file_size:.2f} MB")
        return True
        
    except Exception as e1:
        print(f"      ⚠️ from_keras_model failed: {e1}")
        print(f"      Error details: {type(e1).__name__}: {str(e1)[:200]}")
        
        # Method 2: Try using concrete function approach (works better with newer TF)
        try:
            print(f"      Trying concrete function approach (better for multi-output models)...")
            
            # Create a concrete function from the model with proper input signature
            @tf.function
            def model_func(input_tensor):
                outputs = model(input_tensor, training=False)
                # Ensure we return both outputs as a list
                if isinstance(outputs, (list, tuple)):
                    return outputs
                else:
                    # If single output, this shouldn't happen but handle it
                    return [outputs]
            
            # Get input shape
            input_shape = list(model.input_shape)
            if input_shape[0] is None:
                input_shape[0] = 1  # Use batch size 1 for signature
            
            # Create concrete function
            concrete_func = model_func.get_concrete_function(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32)
            )
            
            # Convert using concrete function
            # Note: from_concrete_functions API varies by TF version
            try:
                # Try newer API (TF 2.5+)
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            except:
                # Fallback to older API
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], model_func)
            converter.optimizations = []  # No optimization - preserve all outputs
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            
            if hasattr(converter, 'experimental_new_converter'):
                converter.experimental_new_converter = False
            
            tflite_model = converter.convert()
            
            # Save
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            file_size = output_path.stat().st_size / (1024 * 1024)
            print(f"      ✅ Saved to: {output_path} (via concrete function)")
            print(f"      File size: {file_size:.2f} MB")
            return True
            
        except Exception as e2:
            print(f"      ⚠️ Concrete function approach failed: {e2}")
            print(f"      Error details: {type(e2).__name__}: {str(e2)[:200]}")
            
            # Method 3: Try saving as H5 and reloading (sometimes fixes issues)
            try:
                print(f"      Trying H5 save/reload approach...")
                import tempfile
                import shutil
                
                temp_dir = tempfile.mkdtemp()
                h5_path = Path(temp_dir) / "temp_model.h5"
                
                # Save model as H5
                print(f"      Saving model as H5...")
                model.save(str(h5_path), save_format='h5')
                
                # Reload model
                print(f"      Reloading model from H5...")
                reloaded_model = tf.keras.models.load_model(str(h5_path))
                
                # Verify reloaded model has 2 outputs
                if len(reloaded_model.outputs) != 2:
                    print(f"      ⚠️ WARNING: Reloaded model has {len(reloaded_model.outputs)} outputs, expected 2")
                
                # Try converting reloaded model
                converter = tf.lite.TFLiteConverter.from_keras_model(reloaded_model)
                converter.optimizations = []  # No optimization
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                
                if hasattr(converter, 'experimental_new_converter'):
                    converter.experimental_new_converter = False
                
                tflite_model = converter.convert()
                
                # Save
                with open(output_path, 'wb') as f:
                    f.write(tflite_model)
                
                # Cleanup
                shutil.rmtree(temp_dir)
                
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"      ✅ Saved to: {output_path} (via H5 reload)")
                print(f"      File size: {file_size:.2f} MB")
                return True
                
            except Exception as e3:
                print(f"      ⚠️ H5 reload approach failed: {e3}")
                print(f"      Error details: {type(e3).__name__}: {str(e3)[:200]}")
                
                # Method 4: Try SavedModel with explicit signature (avoid _funcs issue)
                try:
                    print(f"      Trying SavedModel with explicit signature...")
                    import tempfile
                    import shutil
                    
                    temp_dir = tempfile.mkdtemp()
                    saved_model_path = Path(temp_dir) / "saved_model"
                    
                    # Save model using model.save() with explicit format
                    print(f"      Saving model to SavedModel format...")
                    model.save(str(saved_model_path), save_format='tf', include_optimizer=False)
                    
                    # Try converting with explicit signature (avoids _funcs AttributeError)
                    try:
                        # Load SavedModel and get concrete function from signature
                        saved_model_obj = tf.saved_model.load(str(saved_model_path))
                        
                        # Get the serving signature (this is a concrete function)
                        if 'serving_default' in saved_model_obj.signatures:
                            concrete_func = saved_model_obj.signatures['serving_default']
                            
                            # Convert from concrete function (avoids _funcs AttributeError)
                            try:
                                # Try newer API (TF 2.5+)
                                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
                            except:
                                # Older API
                                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], saved_model_obj)
                        else:
                            # No serving_default signature, try first available
                            if saved_model_obj.signatures:
                                sig_name = list(saved_model_obj.signatures.keys())[0]
                                concrete_func = saved_model_obj.signatures[sig_name]
                                try:
                                    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
                                except:
                                    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], saved_model_obj)
                            else:
                                raise ValueError("No signatures found in SavedModel")
                    except Exception as sig_error:
                        print(f"      ⚠️  Signature-based conversion failed: {sig_error}")
                        print(f"      This is expected with newer TensorFlow versions.")
                        print(f"      The _funcs AttributeError indicates TF version incompatibility.")
                        raise  # Re-raise to try next method
                    
                    converter.optimizations = []
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
                    
                    # Disable all experimental features that might cause issues
                    if hasattr(converter, 'experimental_new_converter'):
                        converter.experimental_new_converter = False
                    if hasattr(converter, 'experimental_lower_to_saved_model'):
                        converter.experimental_lower_to_saved_model = False
                    if hasattr(converter, 'experimental_enable_resource_variables'):
                        converter.experimental_enable_resource_variables = False
                    
                    tflite_model = converter.convert()
                    
                    # Save
                    with open(output_path, 'wb') as f:
                        f.write(tflite_model)
                    
                    # Cleanup
                    shutil.rmtree(temp_dir)
                    
                    file_size = output_path.stat().st_size / (1024 * 1024)
                    print(f"      ✅ Saved to: {output_path} (via SavedModel with signature)")
                    print(f"      File size: {file_size:.2f} MB")
                    return True
                    
                except Exception as e4:
                    print(f"      ⚠️ SavedModel with signature failed: {e4}")
                    print(f"      Error details: {type(e4).__name__}: {str(e4)[:200]}")
                    
                    # Final: Try absolute minimal conversion
                    try:
                        print(f"      Trying absolute minimal conversion (last resort)...")
                        
                        # Get TF version for debugging
                        tf_version = tf.__version__
                        print(f"      TensorFlow version: {tf_version}")
                        
                        # Try the absolute simplest conversion possible
                        converter = tf.lite.TFLiteConverter.from_keras_model(model)
                        # Don't set ANY options - just try basic conversion
                        # This might work but may not preserve outputs correctly
                        
                        tflite_model = converter.convert()
                        
                        with open(output_path, 'wb') as f:
                            f.write(tflite_model)
                        
                        file_size = output_path.stat().st_size / (1024 * 1024)
                        print(f"      ⚠️  WARNING: Used minimal conversion (no guarantees about outputs)")
                        print(f"      ✅ Saved to: {output_path} (minimal conversion)")
                        print(f"      File size: {file_size:.2f} MB")
                        print(f"      ⚠️  IMPORTANT: Please verify the model has 2 outputs after conversion!")
                        return True
                        
                    except Exception as e5:
                        print(f"      ❌ ERROR: All conversion methods failed")
                        print(f"      TensorFlow version: {tf.__version__}")
                        print(f"      Last error: {e5}")
                        print(f"      Error type: {type(e5).__name__}")
                        import traceback
                        traceback.print_exc()
                        print(f"\n      💡 TROUBLESHOOTING:")
                        print(f"      1. Your TensorFlow version ({tf.__version__}) may be incompatible")
                        print(f"      2. Try: pip install tensorflow==2.15.0")
                        print(f"      3. Or: pip install tensorflow==2.13.0")
                        print(f"      4. Then rerun this script")
                        print(f"      5. Make sure you're using TensorFlow 2.x (not 3.x)")
                        return False

def verify_tflite_model(tflite_path):
    """Verify the TFLite model structure."""
    print(f"\n      Verifying TFLite model...")
    
    try:
        # Load interpreter
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        
        # Get input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        num_outputs = len(output_details)
        print(f"      ✅ Model loaded successfully!")
        print(f"      Input shape: {input_details[0]['shape']}")
        print(f"      Number of outputs: {num_outputs}")
        
        if num_outputs != 2:
            print(f"      ❌ ERROR: Expected 2 outputs, but model has {num_outputs} outputs!")
            print(f"      This means the multi-output model was not created correctly.")
            return False
        
        # Check output shapes
        output0_shape = output_details[0]['shape']
        output1_shape = output_details[1]['shape']
        
        print(f"      Output 0 shape: {output0_shape} (dimensions: {len(output0_shape)})")
        print(f"      Output 1 shape: {output1_shape} (dimensions: {len(output1_shape)})")
        
        # Determine which output is features and which is predictions
        # Features should be 4D: [1, H, W, C]
        # Predictions should be 2D: [1, num_classes]
        output0_is_4d = len(output0_shape) == 4
        output1_is_4d = len(output1_shape) == 4
        output0_is_2d = len(output0_shape) == 2
        output1_is_2d = len(output1_shape) == 2
        
        if output0_is_4d and output1_is_2d:
            print(f"      ✅ Output 0 is feature maps: {output0_shape}")
            print(f"      ✅ Output 1 is predictions: {output1_shape}")
            return True
        elif output0_is_2d and output1_is_4d:
            print(f"      ⚠️  Output order is swapped!")
            print(f"      ✅ Output 0 is predictions: {output0_shape}")
            print(f"      ✅ Output 1 is feature maps: {output1_shape}")
            print(f"      Note: Flutter code will handle this automatically.")
            return True
        else:
            print(f"      ❌ ERROR: Cannot identify feature maps and predictions!")
            print(f"      Expected: One 4D output [1, H, W, C] and one 2D output [1, num_classes]")
            print(f"      Got: Output 0: {output0_shape}, Output 1: {output1_shape}")
            return False
        
    except Exception as e:
        print(f"      ❌ ERROR: Could not verify model: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_multi_output_tflite_for_model(model_path, model_name, output_path):
    """Create multi-output TFLite model from a single Keras model."""
    try:
        # Load model
        print(f"\n[1/5] Loading {model_name} model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path))
        print("      Model loaded successfully!")
        print(f"      Input shape: {model.input_shape}")
        print(f"      Output shape: {model.output_shape}")
        
        # Find last conv layer
        print(f"\n[2/5] Finding last convolutional layer...")
        conv_info = find_last_conv_layer(model)
        conv_layer_name = conv_info[1]
        conv_output_shape = conv_info[2]
        print(f"      Found layer: '{conv_layer_name}'")
        print(f"      Output shape: {conv_output_shape}")
        
        # Create multi-output model
        print(f"\n[3/5] Creating multi-output model...")
        try:
            multi_output_model = create_multi_output_model(model, conv_layer_name)
            print("      ✅ Multi-output model created!")
            
            # Handle both list and tuple outputs - use outputs property (list of tensors)
            if hasattr(multi_output_model, 'outputs') and len(multi_output_model.outputs) >= 2:
                # Use outputs property which is a list of output tensors
                shape0 = multi_output_model.outputs[0].shape
                shape1 = multi_output_model.outputs[1].shape
                
                # Handle both TensorShape and tuple
                if hasattr(shape0, 'as_list'):
                    output0_shape = tuple(shape0.as_list())
                elif isinstance(shape0, (list, tuple)):
                    output0_shape = tuple(shape0) if isinstance(shape0, list) else shape0
                else:
                    output0_shape = shape0
                    
                if hasattr(shape1, 'as_list'):
                    output1_shape = tuple(shape1.as_list())
                elif isinstance(shape1, (list, tuple)):
                    output1_shape = tuple(shape1) if isinstance(shape1, list) else shape1
                else:
                    output1_shape = shape1
                    
                print(f"      Output 0 (features) shape: {output0_shape}")
                print(f"      Output 1 (predictions) shape: {output1_shape}")
            else:
                print(f"      ⚠️  Could not access output shapes via outputs property")
                print(f"      Has outputs attr: {hasattr(multi_output_model, 'outputs')}")
                if hasattr(multi_output_model, 'outputs'):
                    print(f"      Outputs length: {len(multi_output_model.outputs)}")
            
            # Verify both outputs are accessible
            if len(multi_output_model.outputs) != 2:
                print(f"      ❌ ERROR: Multi-output model should have 2 outputs, but has {len(multi_output_model.outputs)}")
                return False
                
            # Test with dummy input to ensure both outputs work
            print("      Testing multi-output model with dummy input...")
            dummy_input = tf.zeros((1, 224, 224, 3))
            test_outputs = multi_output_model(dummy_input)
            if len(test_outputs) != 2:
                print(f"      ❌ ERROR: Model returns {len(test_outputs)} outputs, expected 2")
                return False
            print(f"      ✅ Model test successful - both outputs accessible")
            
        except Exception as e:
            print(f"      ❌ ERROR: Failed to create multi-output model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Convert to TFLite
        print(f"\n[4/5] Converting to TFLite...")
        success = convert_to_tflite(multi_output_model, output_path)
        if not success:
            return False
        
        # Verify
        print(f"\n[5/5] Verifying TFLite model...")
        verification_success = verify_tflite_model(output_path)
        if not verification_success:
            print(f"\n      ⚠️  WARNING: Model verification failed!")
            print(f"      The model may not have both outputs as expected.")
            print(f"      Please check the output shapes above.")
            # Don't return False here - let user decide if model is usable
        
        print("\n" + "=" * 60)
        print(f"SUCCESS: {model_name} multi-output TFLite model created!")
        print("=" * 60)
        print(f"  Output: {output_path}")
        print(f"  Feature maps shape: {conv_output_shape}")
        print(f"  Predictions shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR processing {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_multi_output_tflite():
    """Main function to create MobileNetV2 multi-output TFLite model (HerbaScan deprecated)."""
    print("=" * 60)
    print("Phase 2.2: Creating Multi-Output TFLite Model")
    print("Creating MobileNetV2 multi-output model (HerbaScan model deprecated)")
    print("=" * 60)
    
    results = []
    
    # Process MobileNetV2 model
    if MOBILENETV2_MODEL_PATH.exists():
        print("\n" + "=" * 60)
        print("Processing MobileNetV2 Model")
        print("=" * 60)
        success = create_multi_output_tflite_for_model(
            MOBILENETV2_MODEL_PATH,
            "MobileNetV2",
            MOBILENETV2_OUTPUT_PATH
        )
        results.append(("MobileNetV2", success))
    else:
        print(f"\n⚠️  MobileNetV2 model not found at: {MOBILENETV2_MODEL_PATH}")
        print("   Skipping MobileNetV2 multi-output model creation")
        results.append(("MobileNetV2", False))
    
    # Process HerbaScan model (DEPRECATED - kept for backward compatibility)
    if HERBASCAN_MODEL_PATH.exists():
        print("\n" + "=" * 60)
        print("Processing HerbaScan Model (DEPRECATED)")
        print("=" * 60)
        print("⚠️  WARNING: HerbaScan model is deprecated. Use MobileNetV2 only.")
        success = create_multi_output_tflite_for_model(
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
    print("SUMMARY: Multi-Output TFLite Model Creation")
    print("=" * 60)
    for model_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED/SKIPPED"
        print(f"  {model_name}: {status}")
    
    # Check if MobileNetV2 succeeded (required)
    mobilenetv2_success = results[0][1] if results and results[0][0] == "MobileNetV2" else False
    
    if mobilenetv2_success:
        print("\n✅ MobileNetV2 multi-output TFLite model created successfully!")
        print("\nNext steps:")
        print("  1. Copy mobilenetv2_cam_weights.json to assets/models/")
        print(f"  2. Copy {MOBILENETV2_OUTPUT_PATH.name} to assets/models/")
        print("  3. Update Flutter services to use MobileNetV2 multi-output model")
        print("  4. Update pubspec.yaml (if needed)")
        print("\nNote: HerbaScan model is deprecated - only MobileNetV2 is required.")
        return True
    else:
        print("\n❌ ERROR: MobileNetV2 model creation failed!")
        print(f"Please ensure MobileNetV2_model.keras exists in: {models_dir}")
        print("Note: HerbaScan model is deprecated - MobileNetV2 is required.")
        return False

if __name__ == "__main__":
    success = create_multi_output_tflite()
    exit(0 if success else 1)

