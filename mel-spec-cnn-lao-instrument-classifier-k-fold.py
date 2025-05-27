import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import tf2onnx
import onnx

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"GPU configuration error: {e}")

# Configuration - UPDATED FOR 6-SECOND SEGMENTS AND IMPROVED GENERALIZATION
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Data augmentation parameters - BALANCED TO AVOID OVERFITTING
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.85, 1.15)  # Slightly wider range
    PITCH_SHIFT_RANGE = (-2, 2)
    NOISE_FACTOR = 0.01
    
    # Training parameters - ADJUSTED FOR BETTER GENERALIZATION
    BATCH_SIZE = 16
    EPOCHS = 100  # Reduced from 150 to avoid overfitting
    LEARNING_RATE = 0.0005
    EARLY_STOPPING_PATIENCE = 15
    
    # Regularization - INCREASED TO REDUCE OVERCONFIDENCE
    DROPOUT_RATE = 0.6  # Increased from 0.5
    L2_REGULARIZATION = 0.003  # Increased from 0.002
    
    # Label smoothing to reduce overconfidence
    LABEL_SMOOTHING = 0.1
    
    # K-fold cross validation
    USE_KFOLD = True
    K_FOLDS = 5
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/mel_cnn_model_6sec"
    
    # Temperature scaling for prediction
    SOFTMAX_TEMPERATURE = 1.5  # Values > 1 make predictions less confident
    
    # Instrument mapping (transliteration standardization)
    INSTRUMENT_MAPPING = {
        'khean': ['khean', 'khaen', 'แคน', 'ແຄນ'],
        'khong_vong': ['khong', 'kong', 'ຄ້ອງວົງ', 'khong_vong'],
        'pin': ['pin', 'ພິນ'],
        'ranad': ['ranad', 'nad', 'ລະນາດ'],
        'saw': ['saw', 'so', 'ຊໍ', 'ຊໍອູ້'],
        'sing': ['sing', 'ຊິ່ງ']
    }

# Create model directory
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)

def map_instrument_folder(folder_name, class_names):
    """Map a folder name to the corresponding instrument class name"""
    folder_lower = folder_name.lower()
    
    for standard_name, variants in Config.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    # Skip unknown/noise/background folders by returning None
    if 'unknown' in folder_lower or 'noise' in folder_lower or 'background' in folder_lower:
        return None
    
    # Try to match by name
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower  # Return as is if no match

def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram features with intelligent segment selection"""
    # Process the best segment from the audio
    best_segment = process_audio_with_best_segment(audio, sr, segment_duration=Config.SEGMENT_DURATION)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=best_segment,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmax=Config.FMAX
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return mel_spec_normalized, mel_spec_db

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio based on energy and spectral content"""
    # Calculate segment length in samples
    segment_len = int(segment_duration * sr)
    
    # If audio is shorter than segment duration, just pad
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Create segments with 50% overlap
    hop_len = int(segment_len / 2)
    n_hops = max(1, int((len(audio) - segment_len) / hop_len) + 1)
    segments = []
    segment_starts = []
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start < segment_len * 0.8:  # Skip too short segments
            continue
        segments.append(audio[start:end])
        segment_starts.append(start)
    
    if not segments:  # Just in case no valid segments found
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Calculate metrics for each segment
    metrics = []
    for segment in segments:
        # Energy (RMS)
        rms = np.sqrt(np.mean(segment**2))
        
        # Spectral contrast (harmonic-to-noise ratio)
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Spectral flux (tonal variation)
        stft = np.abs(librosa.stft(segment))
        if stft.shape[1] > 1:  # Make sure we have at least 2 frames
            flux = np.mean(np.diff(stft, axis=1)**2)
        else:
            flux = 0
        
        # Combined score (weighted sum of metrics)
        score = rms + 0.3 * contrast + 0.2 * flux
        metrics.append(score)
    
    # Find the best segment
    best_idx = np.argmax(metrics)
    return segments[best_idx]

def augment_audio(audio, sr):
    """Apply data augmentation techniques"""
    augmented_samples = []
    
    # Original audio
    augmented_samples.append(audio)
    
    if Config.USE_AUGMENTATION:
        # Time stretching
        stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
        stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
        augmented_samples.append(stretched)
        
        # Pitch shifting
        pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        augmented_samples.append(shifted)
        
        # Add noise
        noise = np.random.normal(0, Config.NOISE_FACTOR, len(audio))
        noisy = audio + noise
        augmented_samples.append(noisy)
        
        # Combined augmentation: pitch shift + noise
        shifted_noisy = shifted + np.random.normal(0, Config.NOISE_FACTOR/2, len(shifted))
        augmented_samples.append(shifted_noisy)
    
    return augmented_samples

def process_dataset():
    """Process the dataset with mel spectrograms"""
    print("Processing dataset with 6-second mel spectrograms...")
    
    features_list = []
    labels = []
    
    # Get all instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    # First pass to collect class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument is not None:  # Skip unknown/noise folders
            class_names.add(instrument)
    
    class_names = list(class_names)
    print(f"Detected instrument classes: {class_names}")
    
    # Process each folder
    for folder in tqdm(instrument_folders, desc="Processing folders"):
        instrument = map_instrument_folder(folder, class_names)
        
        # Skip folders that don't map to our known instruments
        if instrument is None:
            print(f"Skipping folder: {folder} (not a target instrument)")
            continue
            
        folder_path = os.path.join(Config.DATA_PATH, folder)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {instrument}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:
                    continue
                
                # Apply augmentation
                augmented_samples = augment_audio(audio, sr)
                
                for aug_audio in augmented_samples:
                    # Extract mel spectrogram with intelligent segment selection
                    mel_spec, _ = extract_mel_spectrogram(aug_audio, sr)
                    
                    # Add channel dimension for CNN
                    mel_spec = np.expand_dims(mel_spec, axis=-1)
                    
                    features_list.append(mel_spec)
                    labels.append(instrument)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, class_names

def build_enhanced_cnn_model(input_shape, num_classes):
    """Build an enhanced CNN model for longer mel spectrograms with improved regularization"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Global pooling and dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_with_kfold(X, y, class_names):
    """Train the model with k-fold cross validation"""
    print("\nTraining with K-fold cross validation...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Initialize k-fold
    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_accuracies = []
    fold_histories = []
    fold_models = []
    fold_cms = []
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Training Fold {fold+1}/{Config.K_FOLDS} ---")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Compute class weights for this fold
        class_weights = None
        if Config.USE_CLASS_WEIGHTS:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(weights))
            print("\nClass weights for this fold:", class_weights)
        
        # Build model
        input_shape = X_train.shape[1:]
        model = build_enhanced_cnn_model(input_shape, len(class_names))
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(X_val, y_val)
        print(f"\nFold {fold+1} validation accuracy: {test_acc:.4f}")
        
        # Generate confusion matrix
        y_pred = np.argmax(model.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        
        # Store results for this fold
        fold_accuracies.append(test_acc)
        fold_histories.append(history.history)
        fold_models.append(model)
        fold_cms.append(cm)
        
        # Print classification report for this fold
        print("\nClassification Report for Fold {}:".format(fold+1))
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Plot confusion matrix for this fold
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, f'confusion_matrix_fold_{fold+1}.png'))
        plt.close()
    
    # Print summary of k-fold results
    print("\nK-fold Cross Validation Results:")
    print(f"Accuracy for each fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard deviation: {np.std(fold_accuracies):.4f}")
    
    # Find best model
    best_model_idx = np.argmax(fold_accuracies)
    best_model = fold_models[best_model_idx]
    best_history = fold_histories[best_model_idx]
    
    print(f"\nBest model from fold {best_model_idx+1} with accuracy: {fold_accuracies[best_model_idx]:.4f}")
    
    # Plot training history for best model
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(best_history['accuracy'], label='Train')
    plt.plot(best_history['val_accuracy'], label='Validation')
    plt.title('Accuracy - Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(best_history['loss'], label='Train')
    plt.plot(best_history['val_loss'], label='Validation')
    plt.title('Loss - Best Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'best_model_history.png'))
    plt.close()
    
    return best_model, class_names, fold_accuracies, fold_cms

def train_single_split(X, y, class_names):
    """Train the model with a single train-test split"""
    print("\nTraining with single train-test split...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Compute class weights
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
        class_weights = dict(enumerate(weights))
        print("\nClass weights:", class_weights)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_enhanced_cnn_model(input_shape, len(class_names))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=Config.LABEL_SMOOTHING),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print(f"\nValidation accuracy: {test_acc:.4f}")
    
    # Generate confusion matrix
    y_pred = np.argmax(model.predict(X_val), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'confusion_matrix.png'))
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'training_history.png'))
    plt.close()
    
    return model, history, cm

def convert_to_onnx(model, model_path):
    """Convert Keras model to ONNX format"""
    # Get model input shape
    input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save ONNX model
    onnx_path = os.path.join(model_path, 'mel_cnn_model_6sec.onnx')
    onnx.save_model(onnx_model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")
    
    return onnx_path

def main():
    """Main function"""
    print("Starting Mel-Spectrogram CNN Lao Instrument Classifier training with 6-second segments...")
    print(f"Using {'K-fold cross validation' if Config.USE_KFOLD else 'single train-test split'}")
    
    # Process dataset
    X, y, class_names = process_dataset()
    
    # Train model - either with k-fold or single split
    if Config.USE_KFOLD:
        model, class_names, fold_accuracies, fold_cms = train_with_kfold(X, y, class_names)
        
        # Additional metadata for k-fold
        kfold_results = {
            'fold_accuracies': [float(acc) for acc in fold_accuracies],
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies))
        }
    else:
        model, history, cm = train_single_split(X, y, class_names)
        kfold_results = None
    
    # Save model metadata
    model_metadata = {
        'class_names': class_names,
        'feature_type': 'mel_spectrogram',
        'n_mels': Config.N_MELS,
        'sample_rate': Config.SAMPLE_RATE,
        'segment_duration': Config.SEGMENT_DURATION,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'fmax': Config.FMAX,
        'input_shape': [Config.N_MELS, None, 1],  # Height, width, channels
        'kfold_cross_validation': Config.USE_KFOLD,
        'kfold_results': kfold_results,
        'label_smoothing': Config.LABEL_SMOOTHING,
        'dropout_rate': Config.DROPOUT_RATE,
        'l2_regularization': Config.L2_REGULARIZATION,
        'softmax_temperature': Config.SOFTMAX_TEMPERATURE  # Store for prediction time
    }
    
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save model in h5 format
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'model.h5'))
    
    # Convert to ONNX
    convert_to_onnx(model, Config.MODEL_SAVE_PATH)
    
    print(f"\nTraining complete! 6-second CNN model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()