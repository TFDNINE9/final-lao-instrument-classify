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
from scipy.signal import butter, filtfilt

# Configure GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except Exception as e:
        print(f"GPU configuration error: {e}")

# Configuration - UPDATED FOR BETTER GENERALIZATION
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters - DIVERSIFIED
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # MULTIPLE FEATURE REPRESENTATIONS
    USE_DELTA_FEATURES = True  # Add delta and delta-delta features
    USE_MULTI_RESOLUTION = True  # Multiple time-frequency resolutions
    USE_HARMONIC_PERCUSSIVE = True  # Separate harmonic and percussive components
    
    # Data augmentation parameters - MORE AGGRESSIVE
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.7, 1.3)  # Wider range
    PITCH_SHIFT_RANGE = (-4, 4)  # Wider range
    NOISE_FACTOR = 0.02  # More noise
    MIXUP_ALPHA = 0.2  # Add mixup augmentation
    USE_SPEC_AUGMENT = True  # SpecAugment technique
    
    # Training parameters - REDUCED CAPACITY
    BATCH_SIZE = 16  # Larger batch size
    EPOCHS = 150
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 25
    
    # Regularization - MORE AGGRESSIVE
    DROPOUT_RATE = 0.7  # Higher dropout
    L2_REGULARIZATION = 0.01  # Higher L2
    USE_BATCH_NORM_NOISE = True  # Add noise to batch norm
    
    # Label smoothing to reduce overconfidence
    LABEL_SMOOTHING = 0.2  # Increased
    
    # K-fold cross validation
    USE_KFOLD = True
    K_FOLDS = 5
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/mel_cnn_model_improved"
    
    # Temperature scaling for prediction
    SOFTMAX_TEMPERATURE = 2.0  # Higher temperature for less confident predictions
    
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
    
    return folder_lower

def extract_multi_resolution_features(audio, sr):
    """Extract features at multiple time-frequency resolutions"""
    features = []
    
    # Original resolution
    mel_spec1 = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmax=Config.FMAX
    )
    features.append(librosa.power_to_db(mel_spec1, ref=np.max))
    
    if Config.USE_MULTI_RESOLUTION:
        # Higher time resolution (shorter window)
        mel_spec2 = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=64, fmax=Config.FMAX
        )
        # Resize to match dimensions
        mel_spec2_db = librosa.power_to_db(mel_spec2, ref=np.max)
        mel_spec2_resized = np.resize(mel_spec2_db, (128, mel_spec1.shape[1]))
        features.append(mel_spec2_resized)
        
        # Lower frequency resolution (longer window)
        mel_spec3 = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=4096, hop_length=1024, n_mels=128, fmax=Config.FMAX
        )
        mel_spec3_db = librosa.power_to_db(mel_spec3, ref=np.max)
        # Resize time dimension to match
        mel_spec3_resized = np.resize(mel_spec3_db, (128, mel_spec1.shape[1]))
        features.append(mel_spec3_resized)
    
    return features

def extract_enhanced_features(audio, sr):
    """Extract enhanced mel spectrogram features with multiple representations"""
    # Process the best segment from the audio
    best_segment = process_audio_with_best_segment(audio, sr, segment_duration=Config.SEGMENT_DURATION)
    
    all_features = []
    
    # Multi-resolution features
    multi_res_features = extract_multi_resolution_features(best_segment, sr)
    all_features.extend(multi_res_features)
    
    # Harmonic-percussive separation
    if Config.USE_HARMONIC_PERCUSSIVE:
        harmonic, percussive = librosa.effects.hpss(best_segment)
        
        # Harmonic mel-spectrogram
        mel_harmonic = librosa.feature.melspectrogram(
            y=harmonic, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS, fmax=Config.FMAX
        )
        all_features.append(librosa.power_to_db(mel_harmonic, ref=np.max))
        
        # Percussive mel-spectrogram
        mel_percussive = librosa.feature.melspectrogram(
            y=percussive, sr=sr, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS, fmax=Config.FMAX
        )
        all_features.append(librosa.power_to_db(mel_percussive, ref=np.max))
    
    # Delta features
    if Config.USE_DELTA_FEATURES:
        mel_spec = all_features[0]  # Use the first (original) mel-spectrogram
        delta = librosa.feature.delta(mel_spec)
        delta_delta = librosa.feature.delta(mel_spec, order=2)
        all_features.extend([delta, delta_delta])
    
    # Stack all features
    combined_features = np.stack(all_features, axis=-1)
    
    # Normalize each channel independently
    for i in range(combined_features.shape[-1]):
        channel = combined_features[:, :, i]
        combined_features[:, :, i] = (channel - channel.mean()) / (channel.std() + 1e-8)
    
    return combined_features

def spec_augment(mel_spec, num_mask=2, freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):
    """Apply SpecAugment to mel spectrogram"""
    augmented = mel_spec.copy()
    
    for _ in range(num_mask):
        # Frequency masking
        all_frames_num, all_freqs_num = augmented.shape[:2]
        freq_percentage = np.random.uniform(0.0, freq_masking_max_percentage)
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        augmented[f0:f0 + num_freqs_to_mask, :] = 0
        
        # Time masking
        time_percentage = np.random.uniform(0.0, time_masking_max_percentage)
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        augmented[:, t0:t0 + num_frames_to_mask] = 0
    
    return augmented

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
    
    # For training, randomly select a segment instead of always choosing the "best" one
    # This prevents overfitting to specific segment selection patterns
    if len(segments) > 1:
        # 70% chance to pick random segment, 30% chance to pick best
        if np.random.random() < 0.7:
            return segments[np.random.randint(0, len(segments))]
    
    # Calculate metrics for best segment selection
    metrics = []
    for segment in segments:
        # Energy (RMS)
        rms = np.sqrt(np.mean(segment**2))
        
        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Spectral flux
        stft = np.abs(librosa.stft(segment))
        if stft.shape[1] > 1:
            flux = np.mean(np.diff(stft, axis=1)**2)
        else:
            flux = 0
        
        score = rms + 0.3 * contrast + 0.2 * flux
        metrics.append(score)
    
    best_idx = np.argmax(metrics)
    return segments[best_idx]

def aggressive_augment_audio(audio, sr):
    """Apply aggressive data augmentation techniques"""
    augmented_samples = []
    
    # Original audio
    augmented_samples.append(audio)
    
    if Config.USE_AUGMENTATION:
        # Time stretching with random factor
        for _ in range(2):  # Generate 2 time-stretched versions
            stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            augmented_samples.append(stretched)
        
        # Pitch shifting with random steps
        for _ in range(2):  # Generate 2 pitch-shifted versions
            pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
            shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
            augmented_samples.append(shifted)
        
        # Add different types of noise
        # White noise
        noise = np.random.normal(0, Config.NOISE_FACTOR, len(audio))
        noisy = audio + noise
        augmented_samples.append(noisy)
        
        # Pink noise (1/f noise)
        pink_noise = np.random.randn(len(audio))
        # Simple pink noise approximation
        b, a = butter(1, 0.1)
        pink_noise = filtfilt(b, a, pink_noise) * Config.NOISE_FACTOR
        augmented_samples.append(audio + pink_noise)
        
        # Combined augmentations
        # Time stretch + pitch shift
        stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
        pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
        combined = librosa.effects.time_stretch(audio, rate=stretch_factor)
        combined = librosa.effects.pitch_shift(combined, sr=sr, n_steps=pitch_shift)
        augmented_samples.append(combined)
        
        # Random volume changes
        for _ in range(2):
            volume_factor = np.random.uniform(0.5, 1.5)
            augmented_samples.append(audio * volume_factor)
        
        # Random EQ (emphasize different frequency ranges)
        # Low-pass filter
        b, a = butter(5, 4000 / (sr / 2), 'low')
        low_passed = filtfilt(b, a, audio)
        augmented_samples.append(low_passed)
        
        # High-pass filter
        b, a = butter(5, 500 / (sr / 2), 'high')
        high_passed = filtfilt(b, a, audio)
        augmented_samples.append(high_passed)
    
    return augmented_samples

def mixup_batch(features, labels, alpha=0.2):
    """Apply mixup augmentation to a batch"""
    batch_size = features.shape[0]
    
    # Generate random lambda values
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.maximum(lam, 1 - lam)
    
    # Random permutation for mixing
    index = np.random.permutation(batch_size)
    
    # Mix features
    mixed_features = features.copy()
    for i in range(batch_size):
        if np.random.random() < 0.5:  # 50% chance to apply mixup
            mixed_features[i] = lam[i] * features[i] + (1 - lam[i]) * features[index[i]]
    
    return mixed_features, labels  # Keep original labels for simplicity

def process_dataset():
    """Process the dataset with enhanced features"""
    print("Processing dataset with enhanced multi-channel features...")
    
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
                
                # Apply aggressive augmentation
                augmented_samples = aggressive_augment_audio(audio, sr)
                
                for aug_audio in augmented_samples:
                    # Extract enhanced features
                    features = extract_enhanced_features(aug_audio, sr)
                    
                    # Apply SpecAugment during training
                    if Config.USE_SPEC_AUGMENT and np.random.random() < 0.5:
                        for i in range(features.shape[-1]):
                            features[:, :, i] = spec_augment(features[:, :, i])
                    
                    features_list.append(features)
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

def build_regularized_cnn_model(input_shape, num_classes):
    """Build a heavily regularized CNN model to prevent overfitting"""
    
    # Custom layer to add noise during training
    class GaussianNoise(tf.keras.layers.Layer):
        def __init__(self, stddev, **kwargs):
            super().__init__(**kwargs)
            self.stddev = stddev
        
        def call(self, inputs, training=None):
            if training:
                noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev)
                return inputs + noise
            return inputs
    
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # Add input noise
        GaussianNoise(0.1),
        
        # First convolutional block - reduced capacity
        tf.keras.layers.Conv2D(16, (5, 5), padding='same', 
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(32, (5, 5), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        
        # Fourth convolutional block - reduced filters
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Global pooling
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense layers with heavy regularization
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Add another dense layer with less neurons
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Custom callback to monitor overfitting
class OverfittingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, patience=10):
        super().__init__()
        self.patience = patience
        self.wait = 0
        self.best_gap = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        current_gap = logs.get('accuracy') - logs.get('val_accuracy')
        
        if current_gap < self.best_gap:
            self.best_gap = current_gap
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and current_gap > 0.1:  # 10% gap
                print(f"\nOverfitting detected! Gap: {current_gap:.4f}")
                self.model.stop_training = True

def train_with_kfold(X, y, class_names):
    """Train the model with k-fold cross validation and heavy regularization"""
    print("\nTraining with K-fold cross validation and aggressive regularization...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Initialize k-fold
    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_accuracies = []
    fold_val_accuracies = []
    fold_histories = []
    fold_models = []
    fold_cms = []
    
    # For each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n--- Training Fold {fold+1}/{Config.K_FOLDS} ---")
        
        # Split data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        # Apply mixup to training data
        if Config.MIXUP_ALPHA > 0:
            X_train, y_train = mixup_batch(X_train, y_train, Config.MIXUP_ALPHA)
        
        # Compute class weights for this fold
        class_weights = None
        if Config.USE_CLASS_WEIGHTS:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(weights))
            print("\nClass weights for this fold:", class_weights)
        
        # Build model
        input_shape = X_train.shape[1:]
        model = build_regularized_cnn_model(input_shape, len(class_names))
        
        # Compile model with label smoothing
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=Config.LABEL_SMOOTHING),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # Monitor loss instead of accuracy
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6
            ),
            OverfittingMonitor(patience=15)  # Custom overfitting monitor
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
        
        # Get training accuracy for the last epoch
        train_acc = history.history['accuracy'][-1]
        print(f"Fold {fold+1} training accuracy: {train_acc:.4f}")
        print(f"Fold {fold+1} overfitting gap: {train_acc - test_acc:.4f}")
        
        # Generate confusion matrix
        y_pred = np.argmax(model.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        
        # Store results for this fold
        fold_accuracies.append(test_acc)
        fold_val_accuracies.append(test_acc)
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
    print(f"Validation accuracy for each fold: {[f'{acc:.4f}' for acc in fold_val_accuracies]}")
    print(f"Average validation accuracy: {np.mean(fold_val_accuracies):.4f}")
    print(f"Standard deviation: {np.std(fold_val_accuracies):.4f}")
    
    # Select the model with best generalization (smallest overfitting gap)
    overfitting_gaps = []
    for i, history in enumerate(fold_histories):
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        gap = final_train_acc - final_val_acc
        overfitting_gaps.append(gap)
    
    best_model_idx = np.argmin(overfitting_gaps)  # Smallest gap
    best_model = fold_models[best_model_idx]
    best_history = fold_histories[best_model_idx]
    
    print(f"\nBest model from fold {best_model_idx+1} with smallest overfitting gap: {overfitting_gaps[best_model_idx]:.4f}")
    
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
    
    return best_model, class_names, fold_val_accuracies, fold_cms

def convert_to_onnx(model, model_path):
    """Convert Keras model to ONNX format"""
    # Get model input shape
    input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save ONNX model
    onnx_path = os.path.join(model_path, 'mel_cnn_model_improved.onnx')
    onnx.save_model(onnx_model, onnx_path)
    print(f"ONNX model saved to {onnx_path}")
    
    return onnx_path

def main():
    """Main function"""
    print("Starting Improved Mel-Spectrogram CNN Training...")
    print("Key improvements:")
    print("- Multi-resolution mel-spectrograms")
    print("- Harmonic-percussive separation")
    print("- Delta and delta-delta features")
    print("- Aggressive data augmentation")
    print("- SpecAugment")
    print("- Mixup augmentation")
    print("- Heavily regularized architecture")
    print("- Overfitting monitoring")
    
    # Process dataset
    X, y, class_names = process_dataset()
    
    # Train model with k-fold
    model, class_names, fold_accuracies, fold_cms = train_with_kfold(X, y, class_names)
    
    # Additional metadata for k-fold
    kfold_results = {
        'fold_accuracies': [float(acc) for acc in fold_accuracies],
        'mean_accuracy': float(np.mean(fold_accuracies)),
        'std_accuracy': float(np.std(fold_accuracies))
    }
    
    # Save model metadata
    model_metadata = {
        'class_names': class_names,
        'feature_type': 'enhanced_mel_spectrogram',
        'n_mels': Config.N_MELS,
        'sample_rate': Config.SAMPLE_RATE,
        'segment_duration': Config.SEGMENT_DURATION,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'fmax': Config.FMAX,
        'input_shape': list(model.input_shape[1:]),
        'use_delta_features': Config.USE_DELTA_FEATURES,
        'use_multi_resolution': Config.USE_MULTI_RESOLUTION,
        'use_harmonic_percussive': Config.USE_HARMONIC_PERCUSSIVE,
        'kfold_cross_validation': Config.USE_KFOLD,
        'kfold_results': kfold_results,
        'label_smoothing': Config.LABEL_SMOOTHING,
        'dropout_rate': Config.DROPOUT_RATE,
        'l2_regularization': Config.L2_REGULARIZATION,
        'softmax_temperature': Config.SOFTMAX_TEMPERATURE
    }
    
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save model in h5 format
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'model.h5'))
    
    # Convert to ONNX
    convert_to_onnx(model, Config.MODEL_SAVE_PATH)
    
    print(f"\nTraining complete! Improved model saved to {Config.MODEL_SAVE_PATH}")
    print(f"Expected validation accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print("\nNote: Lower accuracy but better generalization is expected!")

if __name__ == "__main__":
    main()