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

# Enhanced Configuration with discriminative features
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Enhanced feature parameters
    USE_ENHANCED_FEATURES = True  # NEW: Enable discriminative features
    FEATURE_STACK_SIZE = 139  # 128 mel + 11 additional features
    
    # Data augmentation parameters - ENHANCED
    USE_AUGMENTATION = True
    USE_INSTRUMENT_SPECIFIC_AUG = True  # NEW: Instrument-specific augmentation
    TIME_STRETCH_RANGE = (0.9, 1.1)  # Reduced to preserve character
    PITCH_SHIFT_RANGE = (-1, 1)      # Reduced to preserve character
    NOISE_FACTOR = 0.008              # Reduced
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 120  # Increased for better convergence
    LEARNING_RATE = 0.0003  # Reduced for stability
    EARLY_STOPPING_PATIENCE = 20
    
    # Regularization
    DROPOUT_RATE = 0.6
    L2_REGULARIZATION = 0.003
    LABEL_SMOOTHING = 0.1
    
    # K-fold cross validation
    USE_KFOLD = True
    K_FOLDS = 5
    
    # Class balancing with confusion penalty
    USE_CLASS_WEIGHTS = True
    USE_CONFUSION_PENALTY = True  # NEW: Penalty for saw-khaen confusion
    CONFUSION_PENALTY_WEIGHT = 0.3
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/enhanced_mel_cnn_model_6sec"
    
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

def extract_enhanced_features(audio, sr):
    """Extract enhanced features that better distinguish saw from khaen"""
    
    # 1. Standard mel spectrogram (128 features)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=Config.N_FFT, 
        hop_length=Config.HOP_LENGTH, n_mels=Config.N_MELS, 
        fmax=Config.FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if not Config.USE_ENHANCED_FEATURES:
        # Return standard mel spectrogram only
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        return mel_spec_normalized
    
    # 2. Spectral Rolloff (1 feature) - Different for bowed vs reed instruments
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=Config.HOP_LENGTH)
    
    # 3. Zero Crossing Rate (1 feature) - Different for continuous vs articulated
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=Config.HOP_LENGTH)
    
    # 4. Spectral Contrast (7 features) - Emphasizes harmonic structure
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=Config.HOP_LENGTH)
    
    # 5. Spectral Centroid (1 feature) - Brightness measure
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=Config.HOP_LENGTH)
    
    # 6. RMS Energy (1 feature) - Volume dynamics
    rms = librosa.feature.rms(y=audio, hop_length=Config.HOP_LENGTH)
    
    # Get minimum time dimension
    min_time = min(
        mel_spec_db.shape[1], rolloff.shape[1], zcr.shape[1], 
        contrast.shape[1], centroid.shape[1], rms.shape[1]
    )
    
    # Resize all features to same time dimension
    mel_spec_resized = mel_spec_db[:, :min_time]  # 128 x time
    rolloff_resized = rolloff[:, :min_time]       # 1 x time
    zcr_resized = zcr[:, :min_time]               # 1 x time
    contrast_resized = contrast[:, :min_time]     # 7 x time
    centroid_resized = centroid[:, :min_time]     # 1 x time
    rms_resized = rms[:, :min_time]               # 1 x time
    
    # Combine all features (128 + 1 + 1 + 7 + 1 + 1 = 139 features)
    combined_features = np.vstack([
        mel_spec_resized,    # 128 features
        rolloff_resized,     # 1 feature
        zcr_resized,         # 1 feature
        contrast_resized,    # 7 features
        centroid_resized,    # 1 feature
        rms_resized          # 1 feature
    ])
    
    # Normalize the combined features
    combined_normalized = (combined_features - combined_features.mean()) / (combined_features.std() + 1e-8)
    
    return combined_normalized

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio based on energy and spectral content"""
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    hop_len = int(segment_len / 2)
    n_hops = max(1, int((len(audio) - segment_len) / hop_len) + 1)
    segments = []
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start < segment_len * 0.8:
            continue
        segments.append(audio[start:end])
    
    if not segments:
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Calculate metrics for each segment
    metrics = []
    for segment in segments:
        rms = np.sqrt(np.mean(segment**2))
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        stft = np.abs(librosa.stft(segment))
        if stft.shape[1] > 1:
            flux = np.mean(np.diff(stft, axis=1)**2)
        else:
            flux = 0
        score = rms + 0.3 * contrast + 0.2 * flux
        metrics.append(score)
    
    best_idx = np.argmax(metrics)
    return segments[best_idx]

def enhanced_augment_audio(audio, sr, instrument_type):
    """Enhanced augmentation with instrument-specific effects"""
    augmented_samples = []
    augmented_names = []
    
    # Original audio
    augmented_samples.append(audio)
    augmented_names.append("Original")
    
    if not Config.USE_AUGMENTATION:
        return augmented_samples, augmented_names
    
    # Instrument-specific augmentations
    if Config.USE_INSTRUMENT_SPECIFIC_AUG:
        if instrument_type == 'saw':
            # Saw-specific augmentations (bowed string instrument)
            
            # 1. Tremolo effect (simulates bow vibration)
            tremolo_freq = np.random.uniform(4, 8)  # Hz
            tremolo_depth = np.random.uniform(0.05, 0.12)
            t = np.linspace(0, len(audio)/sr, len(audio))
            tremolo = 1 + tremolo_depth * np.sin(2 * np.pi * tremolo_freq * t)
            tremolo_audio = audio * tremolo
            augmented_samples.append(tremolo_audio)
            augmented_names.append("Saw_Tremolo")
            
            # 2. Bow attack enhancement
            attack_enhanced = np.copy(audio)
            try:
                onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples')
                for onset in onset_frames[:3]:  # Limit to first 3 onsets
                    end_sample = min(onset + int(0.1 * sr), len(audio))
                    if end_sample > onset:
                        attack_enhanced[onset:end_sample] *= np.random.uniform(1.1, 1.25)
            except:
                pass  # Skip if onset detection fails
            augmented_samples.append(attack_enhanced)
            augmented_names.append("Saw_Attack")
            
        elif instrument_type == 'khean':
            # Khaen-specific augmentations (reed instrument)
            
            # 1. Breathing effect (volume fluctuation)
            breathing_freq = np.random.uniform(0.5, 2)
            breathing_depth = np.random.uniform(0.05, 0.1)
            t = np.linspace(0, len(audio)/sr, len(audio))
            breathing = 1 + breathing_depth * np.sin(2 * np.pi * breathing_freq * t)
            breathing_audio = audio * breathing
            augmented_samples.append(breathing_audio)
            augmented_names.append("Khean_Breathing")
            
            # 2. Reed flutter (higher frequency modulation)
            flutter_freq = np.random.uniform(8, 15)
            flutter_depth = np.random.uniform(0.02, 0.06)
            t = np.linspace(0, len(audio)/sr, len(audio))
            flutter = 1 + flutter_depth * np.sin(2 * np.pi * flutter_freq * t)
            flutter_audio = audio * flutter
            augmented_samples.append(flutter_audio)
            augmented_names.append("Khean_Flutter")
    
    # Standard augmentations for all instruments
    
    # Time stretching (reduced range to preserve character)
    stretch_factor = np.random.uniform(*Config.TIME_STRETCH_RANGE)
    stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
    if len(stretched) > len(audio):
        stretched = stretched[:len(audio)]
    else:
        stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))), mode='constant')
    augmented_samples.append(stretched)
    augmented_names.append(f"Stretch_{stretch_factor:.2f}")
    
    # Pitch shifting (reduced range)
    pitch_shift = np.random.uniform(*Config.PITCH_SHIFT_RANGE)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
    augmented_samples.append(shifted)
    augmented_names.append(f"Pitch_{pitch_shift:.1f}")
    
    # Add subtle noise
    noise = np.random.normal(0, Config.NOISE_FACTOR, len(audio))
    noisy = audio + noise
    augmented_samples.append(noisy)
    augmented_names.append("Noise")
    
    return augmented_samples, augmented_names

def create_confusion_penalty_loss(class_names, penalty_weight=0.3):
    """Create loss function that penalizes saw-khaen confusion - SHAPE FIXED VERSION"""
    
    # Find indices of saw and khaen
    try:
        saw_idx = class_names.index('saw')
        khaen_idx = class_names.index('khean')
    except ValueError:
        # If classes not found, return standard loss
        return tf.keras.losses.SparseCategoricalCrossentropy()
    
    num_classes = len(class_names)
    
    def confusion_penalty_loss(y_true, y_pred):
        # Manual label smoothing implementation
        if Config.LABEL_SMOOTHING > 0:
            # Ensure y_true is properly shaped - squeeze if needed
            y_true_squeezed = tf.squeeze(y_true)
            
            # Convert sparse labels to one-hot
            y_true_one_hot = tf.one_hot(tf.cast(y_true_squeezed, tf.int32), num_classes)
            
            # Apply label smoothing
            y_true_smooth = y_true_one_hot * (1.0 - Config.LABEL_SMOOTHING) + Config.LABEL_SMOOTHING / num_classes
            
            # Calculate categorical crossentropy with smoothed labels
            base_loss = tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
        else:
            # Standard sparse categorical crossentropy
            base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        if not Config.USE_CONFUSION_PENALTY:
            return tf.reduce_mean(base_loss)
        
        # Ensure y_true is properly shaped for penalty calculation
        y_true_flat = tf.squeeze(y_true)
        
        # Add penalty for saw-khaen confusion
        # Penalty when true class is saw but predicted khaen (and vice versa)
        saw_true_khaen_pred = tf.reduce_mean(
            y_pred[:, khaen_idx] * tf.cast(tf.equal(y_true_flat, saw_idx), tf.float32)
        )
        khaen_true_saw_pred = tf.reduce_mean(
            y_pred[:, saw_idx] * tf.cast(tf.equal(y_true_flat, khaen_idx), tf.float32)
        )
        
        penalty_loss = penalty_weight * (saw_true_khaen_pred + khaen_true_saw_pred)
        
        return tf.reduce_mean(base_loss) + penalty_loss
    
    return confusion_penalty_loss

def process_dataset():
    """Process the dataset with enhanced features"""
    print("Processing dataset with enhanced features...")
    
    features_list = []
    labels = []
    
    # Get all instrument folders
    instrument_folders = [d for d in os.listdir(Config.DATA_PATH) 
                         if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    # First pass to collect class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument is not None:
            class_names.add(instrument)
    
    class_names = list(class_names)
    print(f"Detected instrument classes: {class_names}")
    
    # Process each folder
    for folder in tqdm(instrument_folders, desc="Processing folders"):
        instrument = map_instrument_folder(folder, class_names)
        
        if instrument is None:
            print(f"Skipping folder: {folder} (not a target instrument)")
            continue
            
        folder_path = os.path.join(Config.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3'))]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {instrument}", leave=False):
            file_path = os.path.join(folder_path, audio_file)
            
            try:
                # Load audio
                audio, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
                
                # Skip very short files
                if len(audio) < sr * 0.5:
                    continue
                
                # Apply enhanced augmentation
                augmented_samples, aug_names = enhanced_augment_audio(audio, sr, instrument)
                
                for aug_audio, aug_name in zip(augmented_samples, aug_names):
                    # Extract best segment
                    best_segment = process_audio_with_best_segment(aug_audio, sr, Config.SEGMENT_DURATION)
                    
                    # Extract enhanced features
                    enhanced_features = extract_enhanced_features(best_segment, sr)
                    
                    # Add channel dimension for CNN
                    enhanced_features = np.expand_dims(enhanced_features, axis=-1)
                    
                    features_list.append(enhanced_features)
                    labels.append(instrument)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    # Print dataset summary
    print(f"\nEnhanced Dataset Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Feature shape: {X.shape}")
    if Config.USE_ENHANCED_FEATURES:
        print(f"Features: 128 mel + 11 discriminative = {Config.FEATURE_STACK_SIZE} total")
    else:
        print("Features: 128 mel spectrograms only")
    
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"{cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, class_names

def build_enhanced_cnn_model(input_shape, num_classes):
    """Build enhanced CNN model for discriminative features"""
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=input_shape),
        
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', 
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Third convolutional block with attention
        tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same',
                              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Global pooling and dense layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        # Additional dense layer for better discrimination
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REGULARIZATION)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_with_kfold(X, y, class_names):
    """Train the enhanced model with k-fold cross validation"""
    print("\nTraining enhanced model with K-fold cross validation...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_encoded = np.array([label_to_int[label] for label in y])
    
    # Initialize k-fold
    kf = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
    
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
        
        # Compute class weights
        class_weights = None
        if Config.USE_CLASS_WEIGHTS:
            weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(weights))
            print(f"Class weights: {class_weights}")
        
        # Build model
        input_shape = X_train.shape[1:]
        model = build_enhanced_cnn_model(input_shape, len(class_names))
        
        # Create custom loss with confusion penalty
        custom_loss = create_confusion_penalty_loss(class_names, Config.CONFUSION_PENALTY_WEIGHT)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss=custom_loss,
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
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
        test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nFold {fold+1} validation accuracy: {test_acc:.4f}")
        
        # Generate predictions and confusion matrix
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        
        # Calculate saw-khaen confusion specifically
        if 'saw' in class_names and 'khean' in class_names:
            saw_idx = class_names.index('saw')
            khaen_idx = class_names.index('khean')
            
            saw_total = np.sum(y_val == saw_idx)
            saw_as_khaen = cm[saw_idx, khaen_idx] if saw_total > 0 else 0
            khaen_total = np.sum(y_val == khaen_idx)
            khaen_as_saw = cm[khaen_idx, saw_idx] if khaen_total > 0 else 0
            
            if saw_total > 0:
                saw_confusion_rate = saw_as_khaen / saw_total * 100
                print(f"Saw confused as Khaen: {saw_as_khaen}/{saw_total} ({saw_confusion_rate:.1f}%)")
            if khaen_total > 0:
                khaen_confusion_rate = khaen_as_saw / khaen_total * 100
                print(f"Khaen confused as Saw: {khaen_as_saw}/{khaen_total} ({khaen_confusion_rate:.1f}%)")
        
        # Store results
        fold_accuracies.append(test_acc)
        fold_histories.append(history.history)
        fold_models.append(model)
        fold_cms.append(cm)
        
        # Print classification report
        print(f"\nClassification Report for Fold {fold+1}:")
        print(classification_report(y_val, y_pred, target_names=class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Enhanced Model - Confusion Matrix - Fold {fold+1}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, f'confusion_matrix_fold_{fold+1}.png'))
        plt.close()
    
    # Print summary
    print("\nEnhanced K-fold Cross Validation Results:")
    print(f"Accuracy for each fold: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Average accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard deviation: {np.std(fold_accuracies):.4f}")
    
    # Find best model
    best_model_idx = np.argmax(fold_accuracies)
    best_model = fold_models[best_model_idx]
    best_history = fold_histories[best_model_idx]
    
    print(f"\nBest enhanced model from fold {best_model_idx+1} with accuracy: {fold_accuracies[best_model_idx]:.4f}")
    
    return best_model, class_names, fold_accuracies, fold_cms

def main():
    """Main function for enhanced training"""
    print("Starting Enhanced Mel-Spectrogram CNN with Discriminative Features...")
    print(f"Enhanced features: {'Enabled' if Config.USE_ENHANCED_FEATURES else 'Disabled'}")
    print(f"Instrument-specific augmentation: {'Enabled' if Config.USE_INSTRUMENT_SPECIFIC_AUG else 'Disabled'}")
    print(f"Confusion penalty: {'Enabled' if Config.USE_CONFUSION_PENALTY else 'Disabled'}")
    
    # Process dataset with enhanced features
    X, y, class_names = process_dataset()
    
    # Train model
    if Config.USE_KFOLD:
        model, class_names, fold_accuracies, fold_cms = train_with_kfold(X, y, class_names)
        
        kfold_results = {
            'fold_accuracies': [float(acc) for acc in fold_accuracies],
            'mean_accuracy': float(np.mean(fold_accuracies)),
            'std_accuracy': float(np.std(fold_accuracies))
        }
    else:
        # Single split training (implement if needed)
        kfold_results = None
    
    # Save enhanced model metadata
    model_metadata = {
        'class_names': class_names,
        'feature_type': 'enhanced_mel_spectrogram',
        'enhanced_features': Config.USE_ENHANCED_FEATURES,
        'feature_stack_size': Config.FEATURE_STACK_SIZE if Config.USE_ENHANCED_FEATURES else Config.N_MELS,
        'instrument_specific_augmentation': Config.USE_INSTRUMENT_SPECIFIC_AUG,
        'confusion_penalty': Config.USE_CONFUSION_PENALTY,
        'confusion_penalty_weight': Config.CONFUSION_PENALTY_WEIGHT,
        'n_mels': Config.N_MELS,
        'sample_rate': Config.SAMPLE_RATE,
        'segment_duration': Config.SEGMENT_DURATION,
        'n_fft': Config.N_FFT,
        'hop_length': Config.HOP_LENGTH,
        'fmax': Config.FMAX,
        'input_shape': [Config.FEATURE_STACK_SIZE if Config.USE_ENHANCED_FEATURES else Config.N_MELS, None, 1],
        'kfold_cross_validation': Config.USE_KFOLD,
        'kfold_results': kfold_results,
        'model_version': 'enhanced_v1.0'
    }
    
    with open(os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    # Save model
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'enhanced_model.h5'))
    
    # Convert to ONNX
    try:
        input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input')]
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        onnx_path = os.path.join(Config.MODEL_SAVE_PATH, 'enhanced_mel_cnn_model_6sec.onnx')
        onnx.save_model(onnx_model, onnx_path)
        print(f"Enhanced ONNX model saved to {onnx_path}")
    except Exception as e:
        print(f"Warning: Could not convert to ONNX: {e}")
    
    print(f"\nEnhanced training complete! Model saved to {Config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()