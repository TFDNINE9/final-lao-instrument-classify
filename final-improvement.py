import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import tf2onnx
import onnx
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

# IMPROVED CONFIGURATION
class ImprovedConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 5.0  # Your preferred duration
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # CONSERVATIVE DATA AUGMENTATION
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.95, 1.05)  # Much more conservative
    PITCH_SHIFT_RANGE = (-0.5, 0.5)    # Reduced to preserve characteristics
    NOISE_FACTOR = 0.003               # Reduced noise
    
    # ANTI-OVERFITTING TRAINING PARAMETERS
    BATCH_SIZE = 8                     # Smaller batch size
    EPOCHS = 50                        # Reduced epochs
    LEARNING_RATE = 0.0001             # Lower learning rate
    EARLY_STOPPING_PATIENCE = 8        # Earlier stopping
    
    # AGGRESSIVE REGULARIZATION
    DROPOUT_RATE = 0.6                 # High dropout
    L2_REGULARIZATION = 0.005          # Increased L2
    LABEL_SMOOTHING = 0.15             # Increased label smoothing
    
    # K-FOLD PARAMETERS
    USE_KFOLD = True
    K_FOLDS = 3                        # Reduced for smaller datasets
    
    # SESSION-AWARE SPLITTING
    USE_SESSION_AWARE_SPLIT = True
    
    # MULTI-FEATURE MODEL
    USE_ENHANCED_FEATURES = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/improved_multi_feature_model_5sec"
    
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
os.makedirs(ImprovedConfig.MODEL_SAVE_PATH, exist_ok=True)

def map_instrument_folder(folder_name, class_names):
    """Map folder name to standard instrument name"""
    folder_lower = folder_name.lower()
    
    for standard_name, variants in ImprovedConfig.INSTRUMENT_MAPPING.items():
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
    """Extract comprehensive features that distinguish confused instruments"""
    features = {}
    
    # ATTACK AND ONSET CHARACTERISTICS
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
    if len(onset_frames) > 0:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        attack_characteristics = []
        
        for onset_time in onset_times[:3]:  # Analyze first 3 onsets
            start_sample = int(onset_time * sr)
            end_sample = min(start_sample + int(0.1 * sr), len(audio))
            
            if end_sample > start_sample:
                attack_segment = audio[start_sample:end_sample]
                attack_slope = np.diff(np.abs(attack_segment)).max() if len(attack_segment) > 1 else 0
                attack_characteristics.append(attack_slope)
        
        features['attack_slope_mean'] = np.mean(attack_characteristics) if attack_characteristics else 0
        features['onset_density'] = len(onset_frames) / (len(audio) / sr)
    else:
        features['attack_slope_mean'] = 0
        features['onset_density'] = 0
    
    # VIBRATO AND PITCH MODULATION
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'))
        f0_clean = f0[voiced_flag]
        
        if len(f0_clean) > 10:
            features['pitch_variation'] = np.std(f0_clean)
            f0_diff = np.diff(f0_clean)
            features['vibrato_rate'] = len(np.where(np.diff(np.sign(f0_diff)))[0]) / (len(f0_clean) / sr * 512)
        else:
            features['pitch_variation'] = 0
            features['vibrato_rate'] = 0
    except:
        features['pitch_variation'] = 0
        features['vibrato_rate'] = 0
    
    # ENHANCED HARMONIC ANALYSIS
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    total_energy = harmonic_energy + percussive_energy + 1e-8
    
    features['harmonic_ratio'] = harmonic_energy / total_energy
    features['percussive_ratio'] = percussive_energy / total_energy
    
    # SPECTRAL CHARACTERISTICS
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # ENERGY AND CONTINUITY
    rms_energy = librosa.feature.rms(y=audio)[0]
    features['energy_mean'] = np.mean(rms_energy)
    features['energy_std'] = np.std(rms_energy)
    features['energy_continuity'] = 1.0 / (np.std(rms_energy) + 1e-6)
    
    # TEXTURE AND TRANSIENTS
    zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zero_crossings)
    features['texture_consistency'] = 1.0 / (np.std(zero_crossings) + 1e-6)
    
    # Sharp transients detection
    audio_diff = np.diff(audio)
    sharp_transients = np.sum(np.abs(audio_diff) > np.std(audio_diff) * 2)
    features['transient_density'] = sharp_transients / len(audio)
    
    return features

def process_audio_with_best_segment(audio, sr, segment_duration=5.0):
    """Enhanced segment selection with multiple criteria"""
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Create overlapping segments
    hop_len = int(segment_len / 3)  # More overlap
    segments = []
    scores = []
    
    for start in range(0, len(audio) - segment_len + 1, hop_len):
        segment = audio[start:start + segment_len]
        
        # Multi-criteria scoring
        energy_score = np.sqrt(np.mean(segment**2))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Onset detection score
        try:
            onset_frames = librosa.onset.onset_detect(y=segment, sr=sr)
            onset_score = len(onset_frames) / (len(segment) / sr)
        except:
            onset_score = 0
        
        # Combined score with weights
        combined_score = (0.4 * energy_score + 
                         0.3 * spectral_contrast + 
                         0.3 * onset_score)
        
        segments.append(segment)
        scores.append(combined_score)
    
    # Return best segment
    best_idx = np.argmax(scores)
    return segments[best_idx]

def targeted_augmentation(audio, sr, instrument_type):
    """Apply instrument-specific conservative augmentation"""
    augmented_samples = []
    augmented_samples.append(audio)  # Original
    
    if ImprovedConfig.USE_AUGMENTATION:
        if instrument_type == 'saw':
            # For bowed instruments - preserve smooth characteristics
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.3)
            augmented_samples.append(pitch_shifted)
            
        elif instrument_type == 'pin':
            # For plucked instruments - preserve attack
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            augmented_samples.append(pitch_shifted)
            
        elif instrument_type == 'khean':
            # For wind instruments - add subtle breath noise
            breath_noise = np.random.normal(0, ImprovedConfig.NOISE_FACTOR, len(audio))
            noisy = audio + breath_noise
            augmented_samples.append(noisy)
            
        else:
            # Default conservative augmentation
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            augmented_samples.append(pitch_shifted)
    
    return augmented_samples

def session_aware_split(file_paths, labels, test_size=0.2):
    """Split data by recording session to prevent leakage"""
    session_data = []
    
    for file_path, label in zip(file_paths, labels):
        filename = os.path.basename(file_path)
        
        # Extract session ID from filename
        if '_' in filename:
            parts = filename.split('_')
            session_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        else:
            session_id = filename.split('.')[0][:10]
        
        session_data.append({
            'file_path': file_path,
            'label': label,
            'session_id': session_id
        })
    
    # Group by session
    sessions = {}
    for item in session_data:
        session_id = item['session_id']
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(item)
    
    # Split sessions
    session_ids = list(sessions.keys())
    session_labels = [sessions[sid][0]['label'] for sid in session_ids]
    
    try:
        train_sessions, test_sessions = train_test_split(
            session_ids, test_size=test_size, stratify=session_labels, random_state=42
        )
    except ValueError:
        train_sessions, test_sessions = train_test_split(
            session_ids, test_size=test_size, random_state=42
        )
    
    # Collect files
    train_files, train_labels = [], []
    test_files, test_labels = [], []
    
    for session_id in train_sessions:
        for item in sessions[session_id]:
            train_files.append(item['file_path'])
            train_labels.append(item['label'])
    
    for session_id in test_sessions:
        for item in sessions[session_id]:
            test_files.append(item['file_path'])
            test_labels.append(item['label'])
    
    print(f"Session-aware split: {len(train_sessions)} train sessions, {len(test_sessions)} test sessions")
    return train_files, test_files, train_labels, test_labels

def process_files_with_features(file_paths, labels, with_augmentation=True):
    """Process files and extract both mel-spectrograms and enhanced features"""
    mel_features = []
    enhanced_features = []
    processed_labels = []
    
    for file_path, label in tqdm(zip(file_paths, labels), 
                                desc=f"Processing ({'with' if with_augmentation else 'without'} augmentation)",
                                total=len(file_paths)):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=ImprovedConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.5:
                continue
            
            # Apply augmentation if requested
            if with_augmentation:
                augmented_samples = targeted_augmentation(audio, sr, label)
            else:
                augmented_samples = [audio]
            
            for aug_audio in augmented_samples:
                # Get best segment
                best_segment = process_audio_with_best_segment(aug_audio, sr, 
                                                             ImprovedConfig.SEGMENT_DURATION)
                
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=best_segment, sr=sr,
                    n_fft=ImprovedConfig.N_FFT,
                    hop_length=ImprovedConfig.HOP_LENGTH,
                    n_mels=ImprovedConfig.N_MELS,
                    fmax=ImprovedConfig.FMAX
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
                
                mel_features.append(mel_spec_with_channel)
                
                # Extract enhanced features
                if ImprovedConfig.USE_ENHANCED_FEATURES:
                    enhanced_feat = extract_enhanced_features(best_segment, sr)
                    # Convert to array
                    feat_array = np.array([
                        enhanced_feat['attack_slope_mean'],
                        enhanced_feat['onset_density'],
                        enhanced_feat['pitch_variation'],
                        enhanced_feat['vibrato_rate'],
                        enhanced_feat['harmonic_ratio'],
                        enhanced_feat['percussive_ratio'],
                        enhanced_feat['spectral_centroid_mean'],
                        enhanced_feat['spectral_centroid_std'],
                        enhanced_feat['spectral_bandwidth_mean'],
                        enhanced_feat['spectral_rolloff_mean'],
                        enhanced_feat['energy_mean'],
                        enhanced_feat['energy_std'],
                        enhanced_feat['energy_continuity'],
                        enhanced_feat['zcr_mean'],
                        enhanced_feat['texture_consistency'],
                        enhanced_feat['transient_density']
                    ])
                    enhanced_features.append(feat_array)
                else:
                    enhanced_features.append(np.zeros(16))  # Placeholder
                
                processed_labels.append(label)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(mel_features), np.array(enhanced_features), np.array(processed_labels)

def build_multi_feature_model(mel_input_shape, enhanced_feature_size, num_classes):
    """Build model that combines mel-spectrograms and enhanced features"""
    # Mel-spectrogram input (CNN branch)
    mel_input = tf.keras.layers.Input(shape=mel_input_shape, name='mel_input')
    
    # Simplified CNN to prevent overfitting
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(mel_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(ImprovedConfig.DROPOUT_RATE)(x)
    
    mel_features = tf.keras.layers.Dense(64, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(ImprovedConfig.L2_REGULARIZATION))(x)
    
    # Enhanced features input (Dense branch)
    enhanced_input = tf.keras.layers.Input(shape=(enhanced_feature_size,), name='enhanced_input')
    enhanced_branch = tf.keras.layers.Dense(32, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(ImprovedConfig.L2_REGULARIZATION))(enhanced_input)
    enhanced_branch = tf.keras.layers.Dropout(0.3)(enhanced_branch)
    enhanced_branch = tf.keras.layers.Dense(16, activation='relu')(enhanced_branch)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([mel_features, enhanced_branch])
    combined = tf.keras.layers.Dense(32, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(ImprovedConfig.L2_REGULARIZATION))(combined)
    combined = tf.keras.layers.Dropout(ImprovedConfig.DROPOUT_RATE)(combined)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs=[mel_input, enhanced_input], outputs=output)
    return model

def train_improved_model():
    """Main training function with all improvements"""
    print("🚀 Starting improved model training with all solutions...")
    
    # Step 1: Collect all raw files
    print("📁 Collecting raw files...")
    raw_files = []
    raw_labels = []
    
    instrument_folders = [d for d in os.listdir(ImprovedConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(ImprovedConfig.DATA_PATH, d))]
    
    # Get class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument is not None:
            class_names.add(instrument)
    class_names = list(class_names)
    print(f"Detected instruments: {class_names}")
    
    # Collect files
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, class_names)
        if instrument is None:
            continue
            
        folder_path = os.path.join(ImprovedConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            raw_files.append(file_path)
            raw_labels.append(instrument)
    
    print(f"Found {len(raw_files)} raw files across {len(class_names)} instruments")
    
    # Step 2: Session-aware split
    if ImprovedConfig.USE_SESSION_AWARE_SPLIT:
        train_files, test_files, train_labels, test_labels = session_aware_split(
            raw_files, raw_labels, test_size=0.2
        )
    else:
        train_files, test_files, train_labels, test_labels = train_test_split(
            raw_files, raw_labels, test_size=0.2, stratify=raw_labels, random_state=42
        )
    
    # Step 3: Process files
    print("🔄 Processing training files...")
    X_train_mel, X_train_enhanced, y_train = process_files_with_features(
        train_files, train_labels, with_augmentation=True
    )
    
    print("🔄 Processing test files...")
    X_test_mel, X_test_enhanced, y_test = process_files_with_features(
        test_files, test_labels, with_augmentation=False
    )
    
    # Normalize enhanced features
    if ImprovedConfig.USE_ENHANCED_FEATURES:
        scaler = StandardScaler()
        X_train_enhanced = scaler.fit_transform(X_train_enhanced)
        X_test_enhanced = scaler.transform(X_test_enhanced)
    
    # Step 4: Encode labels
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    print(f"Training samples: {len(X_train_mel)}")
    print(f"Test samples: {len(X_test_mel)}")
    
    # Step 5: Build model
    print("🏗️ Building multi-feature model...")
    mel_input_shape = X_train_mel.shape[1:]
    enhanced_feature_size = X_train_enhanced.shape[1]
    
    model = build_multi_feature_model(mel_input_shape, enhanced_feature_size, len(class_names))
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=ImprovedConfig.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Simple version without label smoothing
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Step 6: Class weights
    class_weights = None
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_encoded), 
        y=y_train_encoded
    )
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {class_weights}")
    
    # Step 7: Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=ImprovedConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ImprovedConfig.MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Step 8: Train model
    print("🎯 Training model...")
    history = model.fit(
        [X_train_mel, X_train_enhanced], y_train_encoded,
        validation_data=([X_test_mel, X_test_enhanced], y_test_encoded),
        epochs=ImprovedConfig.EPOCHS,
        batch_size=ImprovedConfig.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Step 9: Evaluate
    print("📊 Evaluating model...")
    test_loss, test_acc = model.evaluate([X_test_mel, X_test_enhanced], y_test_encoded, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate predictions and confusion matrix
    y_pred = np.argmax(model.predict([X_test_mel, X_test_enhanced]), axis=1)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training history
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Class-wise accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    ax4.bar(class_names, class_accuracy)
    ax4.set_title('Per-class Accuracy')
    ax4.set_ylabel('Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ImprovedConfig.MODEL_SAVE_PATH, 'training_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Save model and metadata
    model.save(os.path.join(ImprovedConfig.MODEL_SAVE_PATH, 'model.h5'))
    
    # Save scaler
    import joblib
    if ImprovedConfig.USE_ENHANCED_FEATURES:
        joblib.dump(scaler, os.path.join(ImprovedConfig.MODEL_SAVE_PATH, 'feature_scaler.pkl'))
    
    # Save metadata
    metadata = {
        'class_names': class_names,
        'test_accuracy': float(test_acc),
        'model_type': 'multi_feature_cnn',
        'enhanced_features': ImprovedConfig.USE_ENHANCED_FEATURES,
        'session_aware_split': ImprovedConfig.USE_SESSION_AWARE_SPLIT,
        'sample_rate': ImprovedConfig.SAMPLE_RATE,
        'segment_duration': ImprovedConfig.SEGMENT_DURATION,
        'n_mels': ImprovedConfig.N_MELS,
        'enhanced_feature_size': enhanced_feature_size,
        'mel_input_shape': list(mel_input_shape)
    }
    
    with open(os.path.join(ImprovedConfig.MODEL_SAVE_PATH, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Training complete! Model saved to {ImprovedConfig.MODEL_SAVE_PATH}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    return model, history, class_names

if __name__ == "__main__":
    train_improved_model()