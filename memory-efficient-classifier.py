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
import gc  # For garbage collection
import warnings
warnings.filterwarnings('ignore')

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"GPU configuration error: {e}")

# Configuration - MEMORY OPTIMIZED AND FIXED
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # REDUCED FEATURE CHANNELS TO SAVE MEMORY
    USE_DELTA_FEATURES = True  # Keep this - it's important
    USE_MULTI_RESOLUTION = False  # Disable to save memory
    USE_HARMONIC_PERCUSSIVE = True  # Keep this - it's valuable
    
    # Data augmentation - REDUCED TO SAVE MEMORY
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.85, 1.15)
    PITCH_SHIFT_RANGE = (-2, 2)
    NOISE_FACTOR = 0.01
    AUGMENTATION_FACTOR = 3  # Reduced from ~10 to 3
    
    # Training parameters
    BATCH_SIZE = 16  # Smaller batch size to save memory
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 20
    
    # Regularization
    DROPOUT_RATE = 0.6
    L2_REGULARIZATION = 0.005
    
    # K-fold cross validation
    USE_KFOLD = True
    K_FOLDS = 5
    
    # Memory optimization
    USE_FLOAT32 = True  # Use float32 instead of float64
    USE_GENERATOR = True  # Use data generator instead of loading all data
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/mel_cnn_model_memory_efficient"
    
    # Temperature scaling
    SOFTMAX_TEMPERATURE = 1.5
    
    # Instrument mapping
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
    
    if 'unknown' in folder_lower or 'noise' in folder_lower or 'background' in folder_lower:
        return None
    
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower

def apply_label_smoothing(y_true, smoothing=0.1):
    """Apply label smoothing to one-hot encoded labels"""
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / num_classes
    return smooth_positives * y_true + smooth_negatives

def extract_memory_efficient_features(audio, sr):
    """Extract features with memory efficiency in mind"""
    # Process the best segment
    best_segment = process_audio_with_best_segment(audio, sr, segment_duration=Config.SEGMENT_DURATION)
    
    features = []
    
    # Base mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=best_segment, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS, fmax=Config.FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db)
    
    # Harmonic-percussive separation (valuable for instrument distinction)
    if Config.USE_HARMONIC_PERCUSSIVE:
        try:
            harmonic, percussive = librosa.effects.hpss(best_segment)
            
            # Harmonic component
            mel_harmonic = librosa.feature.melspectrogram(
                y=harmonic, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS, fmax=Config.FMAX
            )
            features.append(librosa.power_to_db(mel_harmonic, ref=np.max))
            
            # Percussive component
            mel_percussive = librosa.feature.melspectrogram(
                y=percussive, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS, fmax=Config.FMAX
            )
            features.append(librosa.power_to_db(mel_percussive, ref=np.max))
        except Exception as e:
            print(f"Warning: Harmonic-percussive separation failed: {e}")
    
    # Delta features (temporal dynamics)
    if Config.USE_DELTA_FEATURES:
        try:
            delta = librosa.feature.delta(mel_spec_db)
            features.append(delta)
        except Exception as e:
            print(f"Warning: Delta features failed: {e}")
    
    # Stack features
    if len(features) > 1:
        combined_features = np.stack(features, axis=-1)
    else:
        combined_features = np.expand_dims(features[0], axis=-1)
    
    # Normalize each channel
    for i in range(combined_features.shape[-1]):
        channel = combined_features[:, :, i]
        mean_val = channel.mean()
        std_val = channel.std()
        if std_val > 1e-8:  # Avoid division by zero
            combined_features[:, :, i] = (channel - mean_val) / std_val
        else:
            combined_features[:, :, i] = channel - mean_val
    
    # Convert to float32 to save memory
    if Config.USE_FLOAT32:
        combined_features = combined_features.astype(np.float32)
    
    return combined_features

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio"""
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # For training, randomly select segment 70% of the time
    if np.random.random() < 0.7:
        max_start = len(audio) - segment_len
        start = np.random.randint(0, max_start)
        return audio[start:start + segment_len]
    
    # Otherwise, find best segment based on energy
    hop_len = int(segment_len / 2)
    n_hops = max(1, int((len(audio) - segment_len) / hop_len) + 1)
    segments = []
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start >= segment_len * 0.8:
            segments.append(audio[start:end])
    
    if not segments:
        return audio[:segment_len]
    
    # Calculate energy for each segment
    energies = [np.sqrt(np.mean(seg**2)) for seg in segments]
    best_idx = np.argmax(energies)
    return segments[best_idx]

def augment_audio_memory_efficient(audio, sr, augmentation_factor=3):
    """Memory-efficient augmentation - returns fewer samples"""
    augmented_samples = []
    
    # Always include original
    augmented_samples.append(audio)
    
    if Config.USE_AUGMENTATION and augmentation_factor > 1:
        # Randomly select augmentation techniques
        augmentation_choices = []
        
        # Time stretch
        if np.random.random() < 0.6:
            try:
                stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
                stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
                # Ensure same length as original
                if len(stretched) > len(audio):
                    stretched = stretched[:len(audio)]
                else:
                    stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))), mode='constant')
                augmentation_choices.append(stretched)
            except Exception as e:
                print(f"Warning: Time stretch failed: {e}")
        
        # Pitch shift
        if np.random.random() < 0.6:
            try:
                pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
                shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
                augmentation_choices.append(shifted)
            except Exception as e:
                print(f"Warning: Pitch shift failed: {e}")
        
        # Add noise
        if np.random.random() < 0.6:
            try:
                noise = np.random.normal(0, Config.NOISE_FACTOR * np.std(audio), len(audio))
                noisy = audio + noise
                augmentation_choices.append(noisy)
            except Exception as e:
                print(f"Warning: Noise addition failed: {e}")
        
        # Add up to augmentation_factor-1 augmented samples
        n_to_add = min(len(augmentation_choices), augmentation_factor - 1)
        if n_to_add > 0:
            selected_indices = np.random.choice(len(augmentation_choices), n_to_add, replace=False)
            for idx in selected_indices:
                augmented_samples.append(augmentation_choices[idx])
    
    return augmented_samples

def create_file_list():
    """Create a list of all audio files with their labels"""
    file_list = []
    
    # Check if data path exists
    if not os.path.exists(Config.DATA_PATH):
        raise FileNotFoundError(f"Dataset path '{Config.DATA_PATH}' not found!")
    
    # Get all instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    if not instrument_folders:
        raise ValueError("No instrument folders found in dataset!")
    
    # Collect class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument is not None:
            class_names.add(instrument)
    
    class_names = sorted(list(class_names))
    
    if not class_names:
        raise ValueError("No valid instrument classes found!")
    
    # Collect all files
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, class_names)
        if instrument is None:
            print(f"Skipping folder: {folder} (not a target instrument)")
            continue
            
        folder_path = os.path.join(Config.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            file_list.append((file_path, instrument))
    
    if not file_list:
        raise ValueError("No audio files found in dataset!")
    
    return file_list, class_names

def process_dataset_memory_efficient():
    """Process dataset with memory efficiency"""
    print("Processing dataset with memory-efficient approach...")
    
    # Get file list
    file_list, class_names = create_file_list()
    print(f"Found {len(file_list)} audio files")
    print(f"Classes: {class_names}")
    
    # Process files in batches
    features_list = []
    labels = []
    processed_count = 0
    error_count = 0
    
    # Process in smaller batches
    batch_size = 50  # Reduced batch size for better memory management
    
    for i in tqdm(range(0, len(file_list), batch_size), desc="Processing batches"):
        batch_files = file_list[i:i+batch_size]
        batch_features = []
        batch_labels = []
        
        for file_path, instrument in tqdm(batch_files, desc="Processing files", leave=False):
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:
                    continue
                
                # Apply augmentation with limited factor
                augmented_samples = augment_audio_memory_efficient(
                    audio, sr, Config.AUGMENTATION_FACTOR
                )
                
                for aug_audio in augmented_samples:
                    try:
                        # Extract features
                        features = extract_memory_efficient_features(aug_audio, sr)
                        batch_features.append(features)
                        batch_labels.append(instrument)
                        processed_count += 1
                    except Exception as e:
                        print(f"Error extracting features from {file_path}: {str(e)}")
                        error_count += 1
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                error_count += 1
        
        # Add batch to main lists
        features_list.extend(batch_features)
        labels.extend(batch_labels)
        
        # Force garbage collection
        gc.collect()
        
        # Print progress
        if i % (batch_size * 10) == 0:
            print(f"Processed {processed_count} samples so far...")
    
    print(f"Processing complete: {processed_count} samples processed, {error_count} errors")
    
    if not features_list:
        raise ValueError("No features extracted! Check your audio files and processing pipeline.")
    
    # Convert to numpy arrays with float32
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels)
    
    # Clear the lists to free memory
    del features_list
    del labels
    gc.collect()
    
    # Print dataset summary
    print(f"\nDataset summary:")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    print(f"Memory usage: {X.nbytes / 1e9:.2f} GB")
    
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, class_names

def build_memory_efficient_cnn(input_shape, num_classes):
    """Build a CNN model with good regularization but reasonable size"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First conv block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Second conv block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Third conv block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Fourth conv block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
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
        
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
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
    
    # Store results
    fold_accuracies = []
    fold_histories = []
    fold_models = []
    fold_reports = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Training Fold {fold+1}/{Config.K_FOLDS} ---")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Compute class weights
        weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = dict(enumerate(weights))
        print(f"Class weights: {class_weights}")
        
        # Build model
        input_shape = X_train.shape[1:]
        model = build_memory_efficient_cnn(input_shape, len(class_names))
        
        # Print model summary for first fold
        if fold == 0:
            print("\nModel Architecture:")
            model.summary()
        
        # Compile model - FIXED: Using categorical crossentropy without label_smoothing parameter
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',  # Standard loss function
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=Config.EPOCHS,
                batch_size=Config.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"\nFold {fold+1} validation accuracy: {test_acc:.4f}")
            
            # Generate predictions and classification report
            y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
            report = classification_report(y_val, y_pred, target_names=class_names, output_dict=True)
            
            fold_accuracies.append(test_acc)
            fold_histories.append(history.history)
            fold_models.append(model)
            fold_reports.append(report)
            
            # Print classification report
            print(f"\nClassification Report for Fold {fold+1}:")
            print(classification_report(y_val, y_pred, target_names=class_names))
            
            # Plot confusion matrix for this fold
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - Fold {fold+1}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, f'confusion_matrix_fold_{fold+1}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error training fold {fold+1}: {e}")
            fold_accuracies.append(0.0)
            fold_histories.append({})
            fold_models.append(None)
            fold_reports.append({})
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    # Summary
    valid_accuracies = [acc for acc in fold_accuracies if acc > 0]
    
    if valid_accuracies:
        print("\nK-fold Cross Validation Results:")
        print(f"Accuracies: {[f'{acc:.4f}' for acc in valid_accuracies]}")
        print(f"Mean: {np.mean(valid_accuracies):.4f} ± {np.std(valid_accuracies):.4f}")
        
        # Select best model
        best_idx = np.argmax(fold_accuracies)
        best_model = fold_models[best_idx]
        
        # Plot training history for best model
        if best_model and fold_histories[best_idx]:
            plt.figure(figsize=(12, 5))
            
            history = fold_histories[best_idx]
            
            plt.subplot(1, 2, 1)
            plt.plot(history.get('accuracy', []), label='Train')
            plt.plot(history.get('val_accuracy', []), label='Validation')
            plt.title(f'Accuracy - Best Model (Fold {best_idx+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.get('loss', []), label='Train')
            plt.plot(history.get('val_loss', []), label='Validation')
            plt.title(f'Loss - Best Model (Fold {best_idx+1})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'best_model_history.png'))
            plt.close()
        
        return best_model, class_names, valid_accuracies
    else:
        raise Exception("All folds failed to train!")

def main():
    """Main function"""
    print("Starting Memory-Efficient CNN Training...")
    print("Optimizations:")
    print("- Limited augmentation factor")
    print("- Float32 data type")
    print("- Reduced feature channels")
    print("- Batch processing")
    print("- Fixed TensorFlow compatibility issues")
    
    try:
        # Process dataset
        X, y, class_names = process_dataset_memory_efficient()
        
        # Train model
        model, class_names, fold_accuracies = train_with_kfold(X, y, class_names)
        
        if model is None:
            raise Exception("No valid model was trained!")
        
        # Save results
        model_metadata = {
            'class_names': class_names,
            'feature_type': 'mel_spectrogram_hp_delta',
            'n_mels': Config.N_MELS,
            'sample_rate': Config.SAMPLE_RATE,
            'segment_duration': Config.SEGMENT_DURATION,
            'n_fft': Config.N_FFT,
            'hop_length': Config.HOP_LENGTH,
            'fmax': Config.FMAX,
            'input_shape': list(model.input_shape[1:]),
            'use_delta_features': Config.USE_DELTA_FEATURES,
            'use_harmonic_percussive': Config.USE_HARMONIC_PERCUSSIVE,
            'kfold_results': {
                'fold_accuracies': [float(acc) for acc in fold_accuracies],
                'mean_accuracy': float(np.mean(fold_accuracies)),
                'std_accuracy': float(np.std(fold_accuracies))
            },
            'augmentation_factor': Config.AUGMENTATION_FACTOR,
            'softmax_temperature': Config.SOFTMAX_TEMPERATURE
        }
        
        # Save model and metadata
        os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
        model.save(os.path.join(Config.MODEL_SAVE_PATH, 'model.h5'))
        
        with open(os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"\nTraining complete! Model saved to {Config.MODEL_SAVE_PATH}")
        print(f"Final results: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()