import os
import numpy as np
import tensorflow as tf
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

def convert_enhanced_model_to_onnx_alternative():
    """
    Alternative ONNX conversion method with better error handling
    """
    
    MODEL_PATH = "models/enhanced_v2_model_6sec"
    
    print("🔄 Alternative ONNX Conversion for Enhanced Multi-Feature Model...")
    
    # Step 1: Load the trained model
    print("📂 Loading H5 model...")
    try:
        model_h5_path = os.path.join(MODEL_PATH, 'enhanced_model.h5')
        model = tf.keras.models.load_model(model_h5_path)
        print(f"✅ Model loaded successfully from {model_h5_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Step 2: Load metadata
    print("📋 Loading model metadata...")
    try:
        metadata_path = os.path.join(MODEL_PATH, 'enhanced_model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Metadata loaded: {len(metadata['class_names'])} classes")
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return None
    
    # Step 3: Try different conversion approaches
    mel_input_shape = tuple(metadata['mel_input_shape'])
    enhanced_feature_count = metadata['enhanced_feature_count']
    
    print(f"📏 Input shapes:")
    print(f"   - Mel spectrogram: (batch, {mel_input_shape})")
    print(f"   - Enhanced features: (batch, {enhanced_feature_count})")
    
    # Method 1: Try basic tf2onnx conversion
    print("\n🔄 Method 1: Basic tf2onnx conversion...")
    try:
        import tf2onnx
        import onnx
        
        # Create input signature
        input_signature = [
            tf.TensorSpec(shape=(None, *mel_input_shape), dtype=tf.float32, name='mel_input'),
            tf.TensorSpec(shape=(None, enhanced_feature_count), dtype=tf.float32, name='enhanced_input')
        ]
        
        # Convert with minimal parameters
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        
        # Save ONNX model
        onnx_path = os.path.join(MODEL_PATH, 'enhanced_model.onnx')
        onnx.save_model(onnx_model, onnx_path)
        print(f"✅ Method 1 successful! ONNX model saved to: {onnx_path}")
        
        # Verify the model
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("✅ ONNX model verification successful!")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Try SavedModel intermediate conversion
    print("\n🔄 Method 2: SavedModel intermediate conversion...")
    try:
        # Save as SavedModel first
        saved_model_path = os.path.join(MODEL_PATH, 'temp_saved_model')
        model.save(saved_model_path, save_format='tf')
        print(f"✅ SavedModel created at: {saved_model_path}")
        
        # Convert SavedModel to ONNX
        import tf2onnx
        import onnx
        
        onnx_model, _ = tf2onnx.convert.from_saved_model(saved_model_path, opset=13)
        
        # Save ONNX model
        onnx_path = os.path.join(MODEL_PATH, 'enhanced_model.onnx')
        onnx.save_model(onnx_model, onnx_path)
        print(f"✅ Method 2 successful! ONNX model saved to: {onnx_path}")
        
        # Clean up temporary SavedModel
        import shutil
        shutil.rmtree(saved_model_path)
        print("🧹 Temporary files cleaned up")
        
        # Verify the model
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("✅ ONNX model verification successful!")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try with concrete functions
    print("\n🔄 Method 3: Concrete function conversion...")
    try:
        # Create concrete function
        @tf.function
        def model_fn(mel_input, enhanced_input):
            return model([mel_input, enhanced_input])
        
        # Get concrete function
        concrete_fn = model_fn.get_concrete_function(
            tf.TensorSpec(shape=(None, *mel_input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(None, enhanced_feature_count), dtype=tf.float32)
        )
        
        # Convert to ONNX
        import tf2onnx
        import onnx
        
        onnx_model, _ = tf2onnx.convert.from_function(concrete_fn, opset=13)
        
        # Save ONNX model
        onnx_path = os.path.join(MODEL_PATH, 'enhanced_model.onnx')
        onnx.save_model(onnx_model, onnx_path)
        print(f"✅ Method 3 successful! ONNX model saved to: {onnx_path}")
        
        # Verify the model
        onnx_model_check = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model_check)
        print("✅ ONNX model verification successful!")
        
        return onnx_path
        
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    # Method 4: Manual ONNX creation (fallback)
    print("\n🔄 Method 4: Creating TensorFlow Lite model as fallback...")
    try:
        # Convert to TensorFlow Lite instead
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(MODEL_PATH, 'enhanced_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved to: {tflite_path}")
        print("💡 Note: Created TFLite model instead of ONNX for better compatibility")
        
        # Create info file about the conversion
        conversion_info = {
            "conversion_type": "tensorflow_lite",
            "model_file": "enhanced_model.tflite",
            "input_shapes": {
                "mel_input": list(mel_input_shape),
                "enhanced_input": [enhanced_feature_count]
            },
            "note": "ONNX conversion failed, using TensorFlow Lite instead"
        }
        
        with open(os.path.join(MODEL_PATH, 'conversion_info.json'), 'w') as f:
            json.dump(conversion_info, f, indent=2)
        
        return tflite_path
        
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")
    
    # If all methods fail
    print("\n❌ All conversion methods failed!")
    print("\n🔧 Troubleshooting suggestions:")
    print("   1. Check your tf2onnx version: pip install tf2onnx --upgrade")
    print("   2. Check your TensorFlow version compatibility")
    print("   3. Try simplifying your model architecture")
    print("   4. Consider using TensorFlow Lite instead of ONNX")
    
    return None

def test_converted_model(model_path, original_model):
    """Test the converted model"""
    print("\n🧪 Testing converted model...")
    
    try:
        # Load metadata for input shapes
        metadata_path = os.path.join(os.path.dirname(model_path), 'enhanced_model_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        mel_input_shape = tuple(metadata['mel_input_shape'])
        enhanced_feature_count = metadata['enhanced_feature_count']
        
        # Create dummy inputs
        dummy_mel = np.random.randn(1, *mel_input_shape).astype(np.float32)
        dummy_enhanced = np.random.randn(1, enhanced_feature_count).astype(np.float32)
        
        # Test original model
        tf_prediction = original_model.predict([dummy_mel, dummy_enhanced], verbose=0)
        print(f"✅ Original TensorFlow model output shape: {tf_prediction.shape}")
        
        # Test converted model based on file extension
        if model_path.endswith('.onnx'):
            # Test ONNX model
            import onnxruntime as ort
            ort_session = ort.InferenceSession(model_path)
            
            onnx_inputs = {
                'mel_input': dummy_mel,
                'enhanced_input': dummy_enhanced
            }
            onnx_prediction = ort_session.run(None, onnx_inputs)[0]
            
            # Compare predictions
            diff = np.abs(tf_prediction - onnx_prediction).max()
            print(f"✅ ONNX model output shape: {onnx_prediction.shape}")
            print(f"📊 Max difference: {diff:.8f}")
            
            if diff < 1e-5:
                print("🎯 Predictions match very closely!")
            elif diff < 1e-3:
                print("✅ Predictions match well!")
            else:
                print("⚠️ Significant difference in predictions")
        
        elif model_path.endswith('.tflite'):
            # Test TFLite model
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Set inputs
            interpreter.set_tensor(input_details[0]['index'], dummy_mel)
            interpreter.set_tensor(input_details[1]['index'], dummy_enhanced)
            
            # Run inference
            interpreter.invoke()
            tflite_prediction = interpreter.get_tensor(output_details[0]['index'])
            
            # Compare predictions
            diff = np.abs(tf_prediction - tflite_prediction).max()
            print(f"✅ TFLite model output shape: {tflite_prediction.shape}")
            print(f"📊 Max difference: {diff:.8f}")
            
            if diff < 1e-5:
                print("🎯 Predictions match very closely!")
            elif diff < 1e-3:
                print("✅ Predictions match well!")
            else:
                print("⚠️ Significant difference in predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing converted model: {e}")
        return False

def main():
    """Main conversion function"""
    print("🚀 Starting Alternative ONNX Conversion...")
    print("=" * 70)
    
    # Load original model for testing
    MODEL_PATH = "models/enhanced_v2_model_6sec"
    try:
        original_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, 'enhanced_model.h5'))
    except Exception as e:
        print(f"❌ Could not load original model for testing: {e}")
        original_model = None
    
    # Run conversion
    result_path = convert_enhanced_model_to_onnx_alternative()
    
    if result_path:
        print(f"\n✅ Conversion successful!")
        print(f"📁 Model saved to: {result_path}")
        
        # Test the converted model
        if original_model:
            test_converted_model(result_path, original_model)
        
        print("\n🎉 Conversion complete!")
        print("🚀 You can now use the converted model in your Streamlit app!")
        
    else:
        print("\n❌ All conversion attempts failed.")
        print("\n💡 Alternative options:")
        print("   1. Use the original H5 model directly in TensorFlow")
        print("   2. Simplify your model architecture")
        print("   3. Use TensorFlow Serving instead of ONNX")
        print("   4. Convert to TensorFlow Lite for mobile deployment")

if __name__ == "__main__":
    main()