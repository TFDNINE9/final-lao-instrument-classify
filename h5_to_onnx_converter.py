import os
import numpy as np
import tensorflow as tf
import tf2onnx
import onnx
import json
import joblib

def convert_pin_focused_model_to_onnx():
    """Convert the pin-focused H5 model to ONNX format"""
    
    # Paths
    model_dir = "models/enhanced_v2_model_6sec"
    h5_model_path = os.path.join(model_dir, "enhanced_model.h5")
    onnx_model_path = os.path.join(model_dir, "enhanced_v2_model_6sec.onnx")
    metadata_path = os.path.join(model_dir, "enhanced_model_metadata.json")
    
    print("🔄 Converting Pin-Focused H5 model to ONNX...")
    
    # Check if files exist
    if not os.path.exists(h5_model_path):
        print(f"❌ H5 model not found: {h5_model_path}")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"❌ Metadata not found: {metadata_path}")
        return False
    
    try:
        # Load the H5 model
        print("📥 Loading H5 model...")
        model = tf.keras.models.load_model(h5_model_path)
        print("✅ H5 model loaded successfully")
        
        # Load metadata to get input shapes
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        mel_input_shape = metadata['mel_input_shape']
        enhanced_feature_size = metadata['enhanced_features']
        
        print(f"📊 Model info:")
        print(f"   • Mel input shape: {mel_input_shape}")
        print(f"   • Enhanced features: {enhanced_feature_size}")
        print(f"   • Total parameters: {model.count_params():,}")
        
        # Define input signatures for both inputs
        input_signatures = [
            tf.TensorSpec(shape=[None] + mel_input_shape, dtype=tf.float32, name='mel_input'),
            tf.TensorSpec(shape=[None, enhanced_feature_size], dtype=tf.float32, name='enhanced_input')
        ]
        
        print("🔄 Converting to ONNX format...")
        
        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(
            model, 
            input_signature=input_signatures, 
            opset=13,
            output_path=onnx_model_path
        )
        
        print(f"✅ ONNX model saved: {onnx_model_path}")
        
        # Verify the ONNX model
        print("🔍 Verifying ONNX model...")
        onnx_model_check = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model_check)
        print("✅ ONNX model verification successful")
        
        # Test the ONNX model with sample data
        print("🧪 Testing ONNX model with sample data...")
        test_conversion(onnx_model_path, mel_input_shape, enhanced_feature_size)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_conversion(onnx_model_path, mel_input_shape, enhanced_feature_size):
    """Test the converted ONNX model"""
    try:
        import onnxruntime as ort
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_model_path)
        
        # Create sample input data
        batch_size = 1
        sample_mel = np.random.randn(batch_size, *mel_input_shape).astype(np.float32)
        sample_enhanced = np.random.randn(batch_size, enhanced_feature_size).astype(np.float32)
        
        # Get input names
        input_names = [input.name for input in session.get_inputs()]
        print(f"📝 ONNX input names: {input_names}")
        
        # Run inference
        outputs = session.run(None, {
            input_names[0]: sample_mel,
            input_names[1]: sample_enhanced
        })
        
        print(f"🎯 ONNX inference successful!")
        print(f"   • Output shape: {outputs[0].shape}")
        print(f"   • Output probabilities sum: {np.sum(outputs[0]):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = convert_pin_focused_model_to_onnx()
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("📁 Files created:")
        print("   • pin_focused_model.onnx")
        print("✅ Ready for Streamlit deployment!")
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")