import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import json
import tf2onnx
import onnx
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

class QuickFixConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # TARGETED augmentation for problematic classes
    USE_AUGMENTATION = True
    USE_TARGETED_AUGMENTATION = True  # NEW: Class-specific augmentation
    
    # Enhanced augmentation for khean (to fix low recall)
    KHEAN_AUGMENTATION_MULTIPLIER = 3.0  # More khean samples
    PIN_AUGMENTATION_MULTIPLIER = 2.5    # More pin samples  
    UNKNOWN_AUGMENTATION_MULTIPLIER = 2.0 # More unknown samples
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 60  # Slightly reduced for speed
    LEARNING_RATE = 0.0004  # Slightly higher for faster convergence
    EARLY_STOPPING_PATIENCE = 12
    
    # IMPROVED class weights based on confusion matrix analysis
    USE_SMART_CLASS_WEIGHTS = True
    CUSTOM_CLASS_WEIGHTS = {
        'khean': 2.0,      # BOOST heavily (low recall problem)
        'khong_vong': 1.0, # Good performance
        'pin': 1.8,        # BOOST moderately (pin-saw confusion)
        'ranad': 1.0,      # Good performance
        'saw': 0.7,        # REDUCE (stealing too many samples)
        'sing': 1.0,       # Excellent performance
        'unknown': 1.4     # BOOST slightly (conservative detection)
    }
    
    # Regularization
    DROPOUT_RATE = 0.4  # Slightly reduced
    L2_REGULARIZATION = 0.006
    LABEL_SMOOTHING = 0.05  # Reduced for better separation
    
    # Cross-validation
    K_FOLDS = 3
    USE_KFOLD = True
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/quick_fix_model"
    
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
model_path = f"{QuickFixConfig.MODEL_SAVE_PATH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_path, exist_ok=True)

def save_config():
    """Save configuration parameters to a JSON file"""
    config_dict = {key: value for key, value in QuickFixConfig.__dict__.items() 
                  if not key.startswith('__') and not callable(value)}
    
    for k, v in config_dict.items():
        if isinstance(v, tuple):
            config_dict[k] = list(v)
    
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

def map_instrument_folder(folder_name, class_names):
    """Map a folder name to the corresponding instrument class name"""
    folder_lower = folder_name.lower()
    
    if folder_lower.startswith('unknown-'):
        return 'unknown'
    
    for standard_name, variants in QuickFixConfig.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower

def enhanced_segment_selection(audio, sr, segment_duration=6.0, class_hint=None):
    """
    ENHANCED segment selection with class-specific optimization
    """
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    hop_len = segment_len // 2
    segments = []
    scores = []
    
    for start in range(0, len(audio) - segment_len + 1, hop_len):
        segment = audio[start:start + segment_len]
        
        # Basic metrics
        rms = np.sqrt(np.mean(segment**2))
        
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
            
            # CLASS-SPECIFIC scoring improvements
            if class_hint == 'khean':
                # Khean: Look for continuous, stable harmonic content
                # Pitch stability
                pitches = librosa.yin(segment, fmin=80, fmax=400, sr=sr)
                valid_pitches = pitches[pitches > 0]
                if len(valid_pitches) > 10:
                    pitch_stability = 1.0 / (1.0 + np.std(valid_pitches) / (np.mean(valid_pitches) + 1e-8))
                else:
                    pitch_stability = 0
                
                # Harmonic content
                harmonic, _ = librosa.effects.hpss(segment, margin=(2.0, 1.0))
                harmonic_ratio = np.sum(harmonic**2) / (np.sum(segment**2) + 1e-8)
                
                # Khean-specific score
                score = rms * 0.3 + pitch_stability * 0.4 + harmonic_ratio * 0.3
                
            elif class_hint == 'pin':
                # Pin: Look for attack-decay patterns
                onset_strength = np.max(librosa.onset.onset_strength(y=segment, sr=sr))
                
                # Check for decay pattern
                if len(segment) > sr // 2:  # At least 0.5 seconds
                    rms_envelope = librosa.feature.rms(y=segment, frame_length=1024)[0]
                    if len(rms_envelope) > 10:
                        decay_slope = np.polyfit(range(len(rms_envelope)), rms_envelope, 1)[0]
                        decay_score = 1.0 if decay_slope < -0.001 else 0.5
                    else:
                        decay_score = 0.5
                else:
                    decay_score = 0.5
                
                # Pin-specific score
                score = rms * 0.3 + onset_strength * 0.4 + decay_score * 0.3
                
            elif class_hint == 'unknown':
                # Unknown: Look for non-musical patterns
                # Low harmonicity, irregular patterns
                try:
                    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
                    harmonicity = np.max(np.mean(chroma, axis=1))
                    irregularity = 1.0 - harmonicity  # Higher for non-musical sounds
                except:
                    irregularity = 0.5
                
                # Zero crossing rate (higher for speech/noise)
                zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
                
                # Unknown-specific score
                score = rms * 0.3 + irregularity * 0.4 + zcr * 0.3
                
            else:
                # Default scoring for other instruments
                score = rms * 0.7 + (spectral_centroid / 4000) * 0.3
        
        except:
            # Fallback scoring
            score = rms
        
        segments.append(segment)
        scores.append(score)
    
    # Return best segment
    if not segments:
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    best_idx = np.argmax(scores)
    return segments[best_idx]

def extract_enhanced_features(audio, sr, class_hint=None):
    """Enhanced feature extraction with class-specific optimizations"""
    # Use enhanced segment selection
    best_segment = enhanced_segment_selection(audio, sr, QuickFixConfig.SEGMENT_DURATION, class_hint)
    
    # CLASS-SPECIFIC preprocessing
    if class_hint == 'khean':
        # Emphasize harmonic content for khean
        harmonic, percussive = librosa.effects.hpss(best_segment, margin=(3.0, 1.0))
        enhanced_audio = harmonic * 0.9 + percussive * 0.1
        
    elif class_hint == 'pin':
        # Emphasize attack characteristics for pin
        # Light harmonic separation but keep percussive elements
        harmonic, percussive = librosa.effects.hpss(best_segment, margin=(1.5, 2.0))
        enhanced_audio = harmonic * 0.7 + percussive * 0.3
        
    elif class_hint in ['saw', 'khong_vong', 'ranad', 'sing']:
        # Standard processing for well-performing classes
        harmonic, percussive = librosa.effects.hpss(best_segment, margin=(1.0, 2.0))
        enhanced_audio = harmonic * 0.8 + percussive * 0.2
        
    else:  # unknown or default
        # Minimal processing for unknown
        enhanced_audio = best_segment
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=enhanced_audio,
        sr=sr,
        n_fft=QuickFixConfig.N_FFT,
        hop_length=QuickFixConfig.HOP_LENGTH,
        n_mels=QuickFixConfig.N_MELS,
        fmax=QuickFixConfig.FMAX
    )
    
    # Convert to dB and normalize
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return mel_spec_normalized

def targeted_augmentation(audio, sr, class_name):
    """
    TARGETED augmentation based on class-specific needs
    """
    augmented = [audio]  # Original
    
    if not QuickFixConfig.USE_TARGETED_AUGMENTATION:
        return augmented
    
    if class_name == 'khean':
        # KHEAN: Generate more variations to fix low recall
        multiplier = QuickFixConfig.KHEAN_AUGMENTATION_MULTIPLIER
        
        # 1. Harmonic emphasis (strengthen khean characteristics)
        harmonic, _ = librosa.effects.hpss(audio, margin=(3.0, 1.0))
        harmonic_emphasized = harmonic * 1.1 + audio * 0.9
        augmented.append(harmonic_emphasized)
        
        # 2. Slight pitch variations (khean can have micro-tuning)
        for pitch_shift in [-0.3, 0.3]:
            try:
                pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
                augmented.append(pitched)
            except:
                pass
        
        # 3. Breath simulation (realistic khean playing)
        breath_noise = np.random.normal(0, 0.002, len(audio))
        with_breath = audio + breath_noise
        augmented.append(with_breath)
        
        # 4. Volume variations
        for vol_factor in [0.8, 1.2]:
            vol_varied = audio * vol_factor
            augmented.append(np.clip(vol_varied, -1.0, 1.0))
        
    elif class_name == 'pin':
        # PIN: Emphasize plucked characteristics to reduce pin-saw confusion
        multiplier = QuickFixConfig.PIN_AUGMENTATION_MULTIPLIER
        
        # 1. Attack enhancement
        try:
            onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='samples')
            attack_enhanced = audio.copy()
            for onset in onset_frames[:3]:
                if onset + 512 < len(audio):
                    attack_enhanced[onset:onset+256] *= 1.2
            augmented.append(attack_enhanced)
        except:
            pass
        
        # 2. Decay emphasis
        decay_envelope = np.exp(-np.arange(len(audio)) / (sr * 0.8))
        decay_emphasized = audio * (0.6 + 0.4 * decay_envelope)
        augmented.append(decay_emphasized)
        
        # 3. Slight time stretching to vary decay rate
        for stretch in [0.95, 1.05]:
            try:
                stretched = librosa.effects.time_stretch(audio, rate=stretch)
                if len(stretched) > len(audio):
                    stretched = stretched[:len(audio)]
                else:
                    stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
                augmented.append(stretched)
            except:
                pass
    
    elif class_name == 'unknown':
        # UNKNOWN: Add more variety to improve detection
        multiplier = QuickFixConfig.UNKNOWN_AUGMENTATION_MULTIPLIER
        
        # 1. Add various noise types
        noise_types = [
            np.random.normal(0, 0.01, len(audio)),      # Gaussian noise
            np.random.uniform(-0.008, 0.008, len(audio)), # Uniform noise
        ]
        for noise in noise_types:
            noisy = audio * 0.8 + noise * 0.2
            augmented.append(noisy)
        
        # 2. Percussive emphasis (non-harmonic)
        try:
            _, percussive = librosa.effects.hpss(audio, margin=(1.0, 3.0))
            percussive_emphasized = percussive * 1.3 + audio * 0.7
            augmented.append(percussive_emphasized)
        except:
            pass
        
        # 3. Spectral filtering
        filtered = librosa.effects.preemphasis(audio)
        augmented.append(filtered)
    
    else:
        # Standard light augmentation for well-performing classes
        try:
            # Just time stretch
            stretched = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.98, 1.02))
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            else:
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            augmented.append(stretched)
        except:
            pass
    
    return augmented

def build_improved_model(input_shape, num_classes):
    """
    Improved model architecture with better discrimination
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Multi-scale feature extraction (keep what works)
    # Fine scale (3x3) - for attack patterns
    x1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    
    # Medium scale (5x5) - for harmonic patterns  
    x2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Dropout(0.2)(x2)
    
    x2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    # NEW: Frequency-specific attention for better discrimination
    # Low frequency attention (khean, pin fundamentals)
    low_freq_attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(inputs)
    low_freq_slice = tf.keras.layers.Lambda(lambda x: x[:, :64, :, :])(inputs)  # Lower half
    low_freq_attended = tf.keras.layers.Multiply()([low_freq_slice, 
                                                   tf.keras.layers.Lambda(lambda x: x[:, :64, :, :])(low_freq_attention)])
    
    x3 = tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu')(low_freq_attended)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.MaxPooling2D((4, 4))(x3)
    x3 = tf.keras.layers.Dropout(0.3)(x3)
    
    # High frequency attention (bow noise, attacks, cymbal brightness)
    high_freq_slice = tf.keras.layers.Lambda(lambda x: x[:, 64:, :, :])(inputs)  # Upper half
    high_freq_attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(high_freq_slice)
    high_freq_attended = tf.keras.layers.Multiply()([high_freq_slice, high_freq_attention])
    
    x4 = tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu')(high_freq_attended)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.MaxPooling2D((4, 4))(x4)
    x4 = tf.keras.layers.Dropout(0.3)(x4)
    
    # Global pooling and concatenate
    g1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    g2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    g3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
    g4 = tf.keras.layers.GlobalAveragePooling2D()(x4)
    
    merged = tf.keras.layers.Concatenate()([g1, g2, g3, g4])
    
    # Enhanced dense layers
    x = tf.keras.layers.Dense(384, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(QuickFixConfig.L2_REGULARIZATION))(merged)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(QuickFixConfig.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(192, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(QuickFixConfig.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(QuickFixConfig.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def custom_label_smoothing_loss(y_true, y_pred, smoothing=0.05):
    """Custom label smoothing for TF 2.10 - working version"""
    if smoothing == 0.0:
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    
    y_true_one_hot = tf.one_hot(y_true, depth=tf.cast(num_classes, tf.int32))
    y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
    
    smoothing = tf.cast(smoothing, tf.float32)
    y_true_smooth = y_true_one_hot * (1.0 - smoothing) + smoothing / num_classes
    
    return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

def create_smart_class_weights(y_train, class_names):
    """Create smart class weights based on confusion matrix analysis"""
    if not QuickFixConfig.USE_SMART_CLASS_WEIGHTS:
        # Standard balanced weights
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
        return dict(enumerate(class_weights))
    
    # Custom weights based on your confusion matrix analysis
    label_to_int = {label: i for i, label in enumerate(class_names)}
    
    smart_weights = {}
    for i, class_name in enumerate(class_names):
        if class_name in QuickFixConfig.CUSTOM_CLASS_WEIGHTS:
            smart_weights[i] = QuickFixConfig.CUSTOM_CLASS_WEIGHTS[class_name]
        else:
            smart_weights[i] = 1.0
    
    print("Smart class weights applied:")
    for i, weight in smart_weights.items():
        print(f"  {class_names[i]}: {weight}")
    
    return smart_weights

def process_quick_fix_dataset():
    """Process dataset with quick fix improvements"""
    print("Processing dataset with QUICK FIX improvements...")
    
    # Collect files
    all_files = []
    all_labels = []
    
    instrument_folders = [d for d in os.listdir(QuickFixConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(QuickFixConfig.DATA_PATH, d))]
    
    # Get class names
    class_names = set()
    for folder in instrument_folders:
        instrument = map_instrument_folder(folder, [])
        if instrument:
            class_names.add(instrument)
    
    class_names = sorted(list(class_names))
    print(f"Detected classes: {class_names}")
    
    # Collect files
    for folder in tqdm(instrument_folders, desc="Collecting files"):
        instrument = map_instrument_folder(folder, class_names)
        if not instrument:
            continue
            
        folder_path = os.path.join(QuickFixConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            all_files.append(file_path)
            all_labels.append(instrument)
    
    print(f"Total files: {len(all_files)}")
    
    # Train-test split
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=QuickFixConfig.TEST_SIZE,
        random_state=QuickFixConfig.RANDOM_SEED,
        stratify=all_labels
    )
    
    print(f"Split: {len(train_files)} train, {len(test_files)} test files")
    
    # Process training files with TARGETED augmentation
    X_train = []
    y_train = []
    
    print("Processing training files with TARGETED augmentation...")
    for file_path, label in tqdm(zip(train_files, train_labels), total=len(train_files)):
        try:
            audio, sr = librosa.load(file_path, sr=QuickFixConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.8:
                continue
            
            # Enhanced segment selection with class hint
            segment = enhanced_segment_selection(audio, sr, QuickFixConfig.SEGMENT_DURATION, class_hint=label)
            
            # Apply targeted augmentation
            augmented_segments = targeted_augmentation(segment, sr, label)
            
            for aug_segment in augmented_segments:
                # Enhanced feature extraction with class hint
                mel_spec = extract_enhanced_features(aug_segment, sr, class_hint=label)
                mel_spec_with_channel = np.expand_dims(mel_spec, axis=-1)
                
                X_train.append(mel_spec_with_channel)
                y_train.append(label)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Process test files
    X_test = []
    y_test = []
    
    print("Processing test files...")
    for file_path, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        try:
            audio, sr = librosa.load(file_path, sr=QuickFixConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.8:
                continue
            
            # Enhanced processing for test too
            segment = enhanced_segment_selection(audio, sr, QuickFixConfig.SEGMENT_DURATION, class_hint=label)
            mel_spec = extract_enhanced_features(segment, sr, class_hint=label)
            mel_spec_with_channel = np.expand_dims(mel_spec, axis=-1)
            
            X_test.append(mel_spec_with_channel)
            y_test.append(label)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nFinal dataset:")
    print(f"Training: {len(X_train)} samples (from {len(train_files)} files)")
    print(f"Test: {len(X_test)} samples (from {len(test_files)} files)")
    print(f"Feature shape: {X_train.shape[1:]}")
    
    # Show augmentation effect per class
    print(f"\nAugmentation effect:")
    for class_name in class_names:
        class_count = sum(1 for label in y_train if label == class_name)
        original_count = sum(1 for label in train_labels if label == class_name)
        multiplier = class_count / original_count if original_count > 0 else 0
        print(f"  {class_name}: {original_count} → {class_count} ({multiplier:.1f}x)")
    
    return X_train, X_test, y_train, y_test, class_names

def train_quick_fix_model(X_train, y_train, X_test, y_test, class_names):
    """Train the quick fix model"""
    print(f"\nTraining QUICK FIX model with {QuickFixConfig.K_FOLDS}-fold CV...")
    
    # Convert labels
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_to_int, f, indent=4)
    
    # K-fold training
    kf = StratifiedKFold(n_splits=QuickFixConfig.K_FOLDS, shuffle=True, 
                        random_state=QuickFixConfig.RANDOM_SEED)
    
    fold_results = []
    best_model = None
    best_val_acc = 0
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_encoded)):
        print(f"\n--- Fold {fold+1}/{QuickFixConfig.K_FOLDS} ---")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
        
        print(f"Fold {fold+1}: {len(X_fold_train)} train, {len(X_fold_val)} val")
        
        # Smart class weights
        class_weight_dict = create_smart_class_weights(y_fold_train, class_names)
        
        # Build improved model
        model = build_improved_model(X_train.shape[1:], len(class_names))
        
        # Compile with custom loss
        def loss_fn(y_true, y_pred):
            return custom_label_smoothing_loss(y_true, y_pred, QuickFixConfig.LABEL_SMOOTHING)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=QuickFixConfig.LEARNING_RATE),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=QuickFixConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,
                min_lr=1e-7
            )
        ]
        
        # Train
        print(f"Training fold {fold+1} with smart class weights...")
        history = model.fit(
            X_fold_train, y_fold_train,
            validation_data=(X_fold_val, y_fold_val),
            epochs=QuickFixConfig.EPOCHS,
            batch_size=QuickFixConfig.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
        print(f"Fold {fold+1} accuracy: {val_acc:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': float(val_acc),
            'val_loss': float(val_loss)
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
            print(f"✓ New best model from fold {fold+1}")
    
    # CV summary
    fold_accuracies = [r['val_accuracy'] for r in fold_results]
    print(f"\nCV Results: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    
    with open(os.path.join(model_path, 'cv_results.json'), 'w') as f:
        json.dump(fold_results, f, indent=4)
    
    # Final test evaluation
    if best_model is not None:
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        
        test_loss, test_acc = best_model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Detailed predictions and analysis
        y_pred_probs = best_model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Quick Fix Model - Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(model_path, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=class_names))
        
        # Per-class analysis
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, y_pred, average=None, labels=range(len(class_names))
        )
        
        # Calculate improvements
        print("\n" + "="*60)
        print("IMPROVEMENT ANALYSIS vs ORIGINAL MODEL")
        print("="*60)
        
        # Original performance (from your results)
        original_performance = {
            'khean': {'precision': 1.000, 'recall': 0.559, 'f1': 0.717},
            'khong_vong': {'precision': 0.763, 'recall': 0.968, 'f1': 0.853},
            'pin': {'precision': 0.950, 'recall': 0.667, 'f1': 0.784},
            'ranad': {'precision': 0.722, 'recall': 0.950, 'f1': 0.820},
            'saw': {'precision': 0.655, 'recall': 0.983, 'f1': 0.786},
            'sing': {'precision': 1.000, 'recall': 0.949, 'f1': 0.974},
            'unknown': {'precision': 0.979, 'recall': 0.701, 'f1': 0.817}
        }
        
        print(f"{'Class':<12} {'Old F1':<8} {'New F1':<8} {'Change':<10} {'Status'}")
        print("-" * 50)
        
        improvements = []
        for i, class_name in enumerate(class_names):
            old_f1 = original_performance.get(class_name, {}).get('f1', 0)
            new_f1 = f1[i]
            change = new_f1 - old_f1
            
            status = "📈 BETTER" if change > 0.02 else "📊 SIMILAR" if abs(change) <= 0.02 else "📉 WORSE"
            
            print(f"{class_name:<12} {old_f1:<8.3f} {new_f1:<8.3f} {change:+.3f}     {status}")
            improvements.append(change)
        
        avg_improvement = np.mean(improvements)
        print(f"\nAverage F1 improvement: {avg_improvement:+.3f}")
        
        # Save detailed results
        test_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'average_f1_improvement': float(avg_improvement),
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i]),
                    'improvement_vs_original': float(improvements[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist(),
            'improvements_summary': {
                'khean_recall_improvement': float(recall[label_to_int['khean']] - 0.559),
                'pin_recall_improvement': float(recall[label_to_int['pin']] - 0.667),
                'unknown_recall_improvement': float(recall[label_to_int['unknown']] - 0.701),
                'overall_accuracy_improvement': float(test_acc - 0.825)
            }
        }
        
        with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Highlight key improvements
        print("\n🎯 KEY IMPROVEMENTS:")
        khean_recall_improvement = recall[label_to_int['khean']] - 0.559
        pin_recall_improvement = recall[label_to_int['pin']] - 0.667
        unknown_recall_improvement = recall[label_to_int['unknown']] - 0.701
        
        if khean_recall_improvement > 0.05:
            print(f"✅ Khean recall improved by {khean_recall_improvement:.1%} (was the biggest problem!)")
        
        if pin_recall_improvement > 0.05:
            print(f"✅ Pin recall improved by {pin_recall_improvement:.1%}")
            
        if unknown_recall_improvement > 0.05:
            print(f"✅ Unknown recall improved by {unknown_recall_improvement:.1%}")
        
        if test_acc > 0.825:
            print(f"✅ Overall accuracy improved by {(test_acc - 0.825):.1%}")
        
        # Save model
        try:
            best_model.save(os.path.join(model_path, 'quick_fix_model.h5'))
            print("✅ Model saved")
        except Exception as e:
            print(f"Model save error: {e}")
        
        # Convert to ONNX
        try:
            input_signature = [tf.TensorSpec(best_model.inputs[0].shape, tf.float32)]
            onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=input_signature)
            onnx.save_model(onnx_model, os.path.join(model_path, 'model.onnx'))
            print("✅ ONNX model saved")
        except Exception as e:
            print(f"ONNX conversion error: {e}")
        
        return best_model, test_results
    
    return None, None

def plot_training_history(history, fold_num=None, save_path=None):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title(f'Accuracy{"" if fold_num is None else f" - Fold {fold_num}"}', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title(f'Loss{"" if fold_num is None else f" - Fold {fold_num}"}', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def main():
    """Main function for quick fix training"""
    print("="*80)
    print("QUICK FIX LAO INSTRUMENT CLASSIFIER")
    print("="*80)
    print("🚀 Quick improvements targeting your specific issues:")
    print("   • Smart class weights (boost khean 2x, pin 1.8x, reduce saw 0.7x)")
    print("   • Targeted augmentation (more khean/pin samples)")
    print("   • Enhanced segment selection (class-specific optimization)")
    print("   • Improved architecture (frequency-specific attention)")
    print("   • Class-specific feature extraction")
    print("="*80)
    
    # Save config
    save_config()
    
    # Process dataset
    print("\n1. PROCESSING DATASET WITH QUICK FIXES...")
    start_time = datetime.now()
    
    X_train, X_test, y_train, y_test, class_names = process_quick_fix_dataset()
    
    if len(X_train) == 0:
        print("❌ No training data found!")
        return
    
    processing_time = datetime.now() - start_time
    print(f"✓ Dataset processed in {processing_time}")
    
    # Train model
    print("\n2. TRAINING IMPROVED MODEL...")
    train_start = datetime.now()
    
    best_model, results = train_quick_fix_model(X_train, y_train, X_test, y_test, class_names)
    
    training_time = datetime.now() - train_start
    total_time = datetime.now() - start_time
    
    print(f"\n✓ Training completed in {training_time}")
    print(f"✓ Total time: {total_time}")
    
    if results:
        print(f"\n📊 QUICK FIX RESULTS:")
        print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   Average F1 improvement: {results['average_f1_improvement']:+.3f}")
        print(f"   Model saved to: {model_path}")
        
        print(f"\n🎯 Specific improvements:")
        improvements = results['improvements_summary']
        for key, value in improvements.items():
            metric_name = key.replace('_', ' ').title()
            if value > 0.01:
                print(f"   ✅ {metric_name}: +{value:.1%}")
            elif value > -0.01:
                print(f"   📊 {metric_name}: {value:+.1%} (stable)")
            else:
                print(f"   📉 {metric_name}: {value:+.1%}")
        
        print(f"\n🎯 Per-class F1 improvements:")
        for cls, metrics in results['per_class'].items():
            improvement = metrics['improvement_vs_original']
            if improvement > 0.02:
                print(f"   ✅ {cls}: {metrics['f1_score']:.3f} ({improvement:+.3f})")
            elif improvement > -0.02:
                print(f"   📊 {cls}: {metrics['f1_score']:.3f} ({improvement:+.3f})")
            else:
                print(f"   📉 {cls}: {metrics['f1_score']:.3f} ({improvement:+.3f})")
    
    print("\n💡 Next steps:")
    print("   • Test the model with your real audio files")
    print("   • Use the enhanced Streamlit app with this model")
    print("   • Monitor the specific improvements in khean and pin detection")

if __name__ == "__main__":
    main()