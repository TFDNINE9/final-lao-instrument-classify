import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Configuration
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Data augmentation parameters
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.85, 1.15)
    PITCH_SHIFT_RANGE = (-2, 2)
    NOISE_FACTOR = 0.01
    
    # Training parameters - Adjusted for larger dataset
    BATCH_SIZE = 32  # Increased from 16 to process data faster
    EPOCHS = 75  # Increased for larger dataset
    LEARNING_RATE = 0.0008  # Slightly reduced to help with larger dataset
    EARLY_STOPPING_PATIENCE = 12
    
    # Regularization - Adjusted for larger dataset
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.002  # Reduced for larger dataset
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/enhanced_model_larger_dataset"
    
    # Instrument mapping
    INSTRUMENT_MAPPING = {
        'khean': ['khean', 'khaen', 'แคน', 'ແຄນ'],
        'khong_vong': ['khong', 'kong', 'ຄ້ອງວົງ', 'khong_vong'],
        'pin': ['pin', 'ພິນ'],
        'ranad': ['ranad', 'nad', 'ລະນາດ'],
        'saw': ['saw', 'so', 'ຊໍ', 'ຊໍອູ້'],
        'sing': ['sing', 'ຊິ່ງ'],
        'unknown': ['unknown', 'other', 'misc']
    }

# Create model directory
model_path = f"{Config.MODEL_SAVE_PATH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_path, exist_ok=True)

def map_instrument_folder(folder_name, class_names):
    """Map a folder name to the corresponding instrument class name"""
    folder_lower = folder_name.lower()
    
    # Special case for unknown folders
    if folder_lower.startswith('unknown-'):
        return 'unknown'
    
    for standard_name, variants in Config.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    # Try to match by name
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower  # Return as is if no match

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
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start < segment_len * 0.8:  # Skip too short segments
            continue
        segments.append(audio[start:end])
    
    if not segments:  # Just in case no valid segments found
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Calculate metrics for each segment
    metrics = []
    for segment in segments:
        # Energy (RMS)
        rms = np.sqrt(np.mean(segment**2))
        
        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Combined score
        score = rms + 0.3 * contrast
        metrics.append(score)
    
    # Find the best segment
    best_idx = np.argmax(metrics)
    return segments[best_idx]

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
    
    return mel_spec_normalized

def augment_audio(audio, sr):
    """Apply data augmentation techniques"""
    augmented_samples = []
    
    # Original audio
    augmented_samples.append(audio)
    
    if Config.USE_AUGMENTATION:
        # Time stretching
        stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
        stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
        # Ensure same length as original
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))), mode='constant')
        augmented_samples.append(stretched)
        
        # Pitch shifting
        pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        augmented_samples.append(shifted)
        
        # Add noise
        noise = np.random.normal(0, Config.NOISE_FACTOR, len(audio))
        noisy = audio + noise
        augmented_samples.append(noisy)
    
    return augmented_samples

def process_dataset():
    """Process the dataset with proper train/test split before augmentation"""
    print("Processing dataset with 6-second mel spectrograms...")
    
    # Collect all file paths first
    all_files = []
    all_labels = []
    
    # Get all instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    # First pass to collect class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument is not None:
            class_names.add(instrument)
    
    class_names = sorted(list(class_names))
    print(f"Detected instrument classes: {class_names}")
    
    # Collect all audio files (before augmentation)
    for folder in tqdm(instrument_folders, desc="Finding audio files"):
        instrument = map_instrument_folder(folder, class_names)
        if instrument is None:
            print(f"Skipping folder: {folder} (not a target instrument)")
            continue
            
        folder_path = os.path.join(Config.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            all_files.append(file_path)
            all_labels.append(instrument)
    
    # Split files into train/test BEFORE augmentation
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_SEED, 
        stratify=all_labels
    )
    
    print(f"Split dataset: {len(train_files)} training files, {len(test_files)} testing files")
    
    # Print class distribution in train and test sets
    train_class_dist = {}
    test_class_dist = {}
    for label in train_labels:
        train_class_dist[label] = train_class_dist.get(label, 0) + 1
    for label in test_labels:
        test_class_dist[label] = test_class_dist.get(label, 0) + 1
    
    print("\nTraining set class distribution:")
    for cls, count in sorted(train_class_dist.items()):
        print(f"  {cls}: {count} files ({count/len(train_labels)*100:.1f}%)")
    
    print("\nTest set class distribution:")
    for cls, count in sorted(test_class_dist.items()):
        print(f"  {cls}: {count} files ({count/len(test_labels)*100:.1f}%)")
    
    # Process training files with augmentation - with progress tracking
    X_train = []
    y_train = []
    file_counter = 0
    
    print(f"\nProcessing training files with augmentation...")
    progress_bar = tqdm(total=len(train_files))
    
    for file_path, label in zip(train_files, train_labels):
        try:
            audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
            if len(audio) < sr * 0.5:  # Skip very short files
                continue
                
            # Apply augmentation to TRAINING data only
            augmented_samples = augment_audio(audio, sr)
            
            for aug_audio in augmented_samples:
                mel_spec = extract_mel_spectrogram(aug_audio, sr)
                mel_spec = np.expand_dims(mel_spec, axis=-1)
                X_train.append(mel_spec)
                y_train.append(label)
            
            file_counter += 1
            if file_counter % 10 == 0:  # Update progress every 10 files
                progress_bar.update(10)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Update progress bar to completion
    progress_bar.update(len(train_files) - file_counter)
    progress_bar.close()
    
    # Process test files WITHOUT augmentation
    X_test = []
    y_test = []
    
    for file_path, label in tqdm(zip(test_files, test_labels), desc="Processing testing files", total=len(test_files)):
        try:
            audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
            if len(audio) < sr * 0.5:  # Skip very short files
                continue
                
            # No augmentation for test data - just extract features
            mel_spec = extract_mel_spectrogram(audio, sr)
            mel_spec = np.expand_dims(mel_spec, axis=-1)
            X_test.append(mel_spec)
            y_test.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Print dataset summary
    print(f"\nDataset summary after processing:")
    print(f"Original files: {len(all_files)}")
    print(f"Training samples (after augmentation): {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Augmentation multiplier: ~{len(X_train)/len(train_files):.1f}x")
    print(f"Feature shape: {X_train.shape[1:]}")
    
    return X_train, X_test, y_train, y_test, class_names

def build_model(input_shape, num_classes):
    """Build CNN model with improved architecture for larger dataset"""
    
    # Try both Sequential and Functional API approaches with error handling
    try:
        # Attempt Sequential API (newer TensorFlow)
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=input_shape),
            
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.4),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(Config.DROPOUT_RATE),
            
            # Global pooling and first dense layer
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dropout(Config.DROPOUT_RATE),
            
            # Output layer
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
        
    except Exception as e:
        print(f"Error building model with Sequential API: {e}")
        print("Trying alternative approach...")
        
        # For older TensorFlow versions, use Functional API
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # First block
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Second block
        x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Third block
        x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Fourth block
        x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
        
        # Global pooling and dense layers
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(Config.DROPOUT_RATE)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

def train_model(X_train, y_train, X_test, y_test, class_names):
    """Train the model with improved approach for larger dataset"""
    print("\nTraining model...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_to_int, f, indent=4)
    
    # Set validation split
    validation_split = 0.15  # Use 15% of training data for validation
    
    # Build model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, len(class_names))
    
    # Compute class weights
    class_weights = None
    if Config.USE_CLASS_WEIGHTS:
        weights = compute_class_weight(class_weight='balanced', 
                                       classes=np.unique(y_train_encoded), 
                                       y=y_train_encoded)
        class_weights = dict(enumerate(weights))
        print("\nClass weights:")
        for i, w in enumerate(weights):
            print(f"  {class_names[i]}: {w:.4f}")
    
    # Compile model
    try:
        # For newer TensorFlow versions
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
    except:
        # For older TensorFlow versions
        optimizer = tf.keras.optimizers.Adam(lr=Config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create a summary of the model architecture
    try:
        model.summary()
        
        # Try to save model summary to file
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        with open(os.path.join(model_path, 'model_summary.txt'), 'w') as f:
            f.write('\n'.join(stringlist))
    except:
        print("Warning: Unable to generate model summary")
    
    # Prepare callbacks
    callbacks_list = []
    
    # Early stopping - handle different parameter names in different TF versions
    try:
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=Config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        callbacks_list.append(es_callback)
    except:
        try:
            # For older TensorFlow that uses 'val_acc' instead of 'val_accuracy'
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            )
            callbacks_list.append(es_callback)
        except:
            # Fallback without restore_best_weights
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE
            )
            callbacks_list.append(es_callback)
    
    # Learning rate reduction
    try:
        lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7
        )
        callbacks_list.append(lr_callback)
    except:
        print("Warning: ReduceLROnPlateau not supported in this TF version")
    
    # Model checkpoint - save best model during training
    try:
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        callbacks_list.append(checkpoint_callback)
    except:
        try:
            # Try with 'val_acc' for older TF
            checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_path, 'best_model.h5'),
                monitor='val_acc',
                save_best_only=True,
                verbose=1
            )
            callbacks_list.append(checkpoint_callback)
        except:
            print("Warning: ModelCheckpoint not fully supported")
    
    # Train the model
    print(f"\nStarting training with batch size {Config.BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train_encoded,
        validation_split=validation_split,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Try to load the best model from checkpoint
    best_model = None
    try:
        best_model = tf.keras.models.load_model(os.path.join(model_path, 'best_model.h5'))
        print("Loaded best model from checkpoint")
    except:
        print("Using final model (could not load checkpoint)")
        best_model = model
    
    # Evaluate model on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = best_model.evaluate(X_test, y_test_encoded)
    print(f"\nTest accuracy: {test_acc:.4f}")
    
    # Generate confusion matrix
    y_pred = np.argmax(best_model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Calculate precision, recall, F1 scores for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_encoded, y_pred, average=None, labels=range(len(class_names))
    )
    
    # Store results
    results = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist(),
        'class_names': class_names,
        'confusion_matrix': cm.tolist(),
        'unknown_class_metrics': {
            'precision': float(precision[label_to_int['unknown']]),
            'recall': float(recall[label_to_int['unknown']]),
            'f1': float(f1[label_to_int['unknown']])
        }
    }
    
    # Save results to file
    with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'confusion_matrix.png'))
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Handle different metric names in different TF versions
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history[acc_key], label='Train')
    plt.plot(history.history[val_acc_key], label='Validation')
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
    plt.savefig(os.path.join(model_path, 'training_history.png'))
    plt.close()
    
    # Try to save the model
    try:
        best_model.save(os.path.join(model_path, 'model.h5'))
    except Exception as e:
        print(f"Warning: Could not save model: {e}")
        try:
            best_model.save_weights(os.path.join(model_path, 'model_weights.h5'))
            print("Saved model weights instead")
        except Exception as e:
            print(f"Warning: Could not save model weights: {e}")
    
    return best_model, y_test_encoded, y_pred, test_acc

def main():
    """Main function"""
    print("Starting Lao Instrument Classifier Training for Larger Dataset...")
    
    # Process dataset with proper train/test split
    X_train, X_test, y_train, y_test, class_names = process_dataset()
    
    # Train model
    model, y_test_encoded, y_pred, test_acc = train_model(X_train, y_train, X_test, y_test, class_names)
    
    # Calculate additional metrics for 'unknown' class
    unknown_idx = list(class_names).index('unknown')
    unknown_true = (y_test_encoded == unknown_idx)
    unknown_pred = (y_pred == unknown_idx)
    
    # Calculate precision, recall, F1 for unknown class
    unknown_tp = np.sum(unknown_true & unknown_pred)
    unknown_fp = np.sum(~unknown_true & unknown_pred)
    unknown_fn = np.sum(unknown_true & ~unknown_pred)
    
    unknown_precision = unknown_tp / (unknown_tp + unknown_fp) if (unknown_tp + unknown_fp) > 0 else 0
    unknown_recall = unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) > 0 else 0
    unknown_f1 = 2 * (unknown_precision * unknown_recall) / (unknown_precision + unknown_recall) if (unknown_precision + unknown_recall) > 0 else 0
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'unknown_precision': float(unknown_precision),
        'unknown_recall': float(unknown_recall),
        'unknown_f1': float(unknown_f1),
        'class_names': class_names
    }
    
    with open(os.path.join(model_path, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTraining complete! Model and artifacts saved to {model_path}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Unknown class metrics:")
    print(f"  Precision: {unknown_precision:.4f}")
    print(f"  Recall: {unknown_recall:.4f}")
    print(f"  F1 Score: {unknown_f1:.4f}")

if __name__ == "__main__":
    main()