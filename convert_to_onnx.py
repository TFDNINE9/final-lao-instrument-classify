import os
import sys
import json
import tensorflow as tf
import tf2onnx
import onnx
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Convert TensorFlow model to ONNX format')
parser.add_argument('--model_path', type=str, required=True, help='Path to the TensorFlow model (.h5)')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the ONNX model')
parser.add_argument('--label_mapping_path', type=str, required=True, help='Path to the label mapping JSON file')

args = parser.parse_args()

print(f"TensorFlow version: {tf.__version__}")
print(f"tf2onnx version: {tf2onnx.__version__}")

def convert_to_onnx(model_path, output_path, label_mapping_path):
    """Convert a TensorFlow model to ONNX format"""
    
    print(f"Loading TensorFlow model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying to load model with additional options...")
        try:
            # For older TensorFlow versions with custom losses/metrics
            model = tf.keras.models.load_model(
                model_path, 
                compile=False  # Skip loading optimizer/loss
            )
            print("Model loaded successfully with compile=False")
        except Exception as e2:
            print(f"Error loading model (second attempt): {e2}")
            sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get model input shape
    input_shape = model.input_shape
    print(f"Model input shape: {input_shape}")
    
    # Load label mapping
    try:
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
            print(f"Loaded label mapping with {len(label_mapping)} classes")
            
        # Save label mapping alongside the ONNX model
        label_mapping_out_path = os.path.join(os.path.dirname(output_path), 'label_mapping.json')
        with open(label_mapping_out_path, 'w') as f:
            json.dump(label_mapping, f, indent=4)
        print(f"Saved label mapping to: {label_mapping_out_path}")
    except Exception as e:
        print(f"Warning: Could not load or save label mapping: {e}")
    
    # Convert to ONNX
    print("Converting to ONNX format...")
    try:
        # Create input signature
        spec = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
        
        # Convert model
        output_path_temp = output_path + ".temp"
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path_temp)
        
        # Save model (sometimes the above function doesn't actually save)
        onnx.save_model(model_proto, output_path)
        
        # Clean up temp file if it exists
        if os.path.exists(output_path_temp):
            os.remove(output_path_temp)
            
        print(f"ONNX model saved to: {output_path}")
        
        # Verify the model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verified successfully")
        except Exception as e:
            print(f"Warning: ONNX model verification failed: {e}")
        
        return True
    except Exception as e:
        print(f"Error converting to ONNX: {e}")
        return False

def main():
    """Main function"""
    success = convert_to_onnx(
        args.model_path,
        args.output_path,
        args.label_mapping_path
    )
    
    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()