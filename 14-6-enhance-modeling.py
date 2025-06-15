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

class EnhancedConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 4.0  # Shorter for better instrument focus
    OVERLAP_RATIO = 0.67    # More overlap for ensemble
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Advanced augmentation (conservative for background music)
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.95, 1.05)  # Very conservative
    PITCH_SHIFT_RANGE = (-0.5, 0.5)    # Minimal pitch shift
    NOISE_FACTOR = 0.003               # Very low noise
    
    # Training parameters
    BATCH_SIZE = 16         # Smaller for better convergence
    EPOCHS = 120            # More epochs with better regularization
    LEARNING_RATE = 0.0002  # Lower learning rate
    EARLY_STOPPING_PATIENCE = 20
    
    # Enhanced regularization
    DROPOUT_RATE = 0.6
    L2_REGULARIZATION = 0.01
    LABEL_SMOOTHING = 0.1   # Custom implementation for TF 2.10
    
    # Multi-segment ensemble
    SEGMENTS_PER_AUDIO = 3  # Multiple segments for ensemble
    
    # Cross-validation
    K_FOLDS = 5
    USE_KFOLD = True
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/enhanced_robust_model"
    
    # Instrument mapping (your existing mapping)
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
model_path = f"{EnhancedConfig.MODEL_SAVE_PATH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_path, exist_ok=True)

def save_config():
    """Save configuration parameters to a JSON file"""
    config_dict = {key: value for key, value in EnhancedConfig.__dict__.items() 
                  if not key.startswith('__') and not callable(value)}
    
    # Convert non-serializable types
    for k, v in config_dict.items():
        if isinstance(v, tuple):
            config_dict[k] = list(v)
    
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)

def map_instrument_folder(folder_name, class_names):
    """Map a folder name to the corresponding instrument class name"""
    folder_lower = folder_name.lower()
    
    # Special case for unknown folders
    if folder_lower.startswith('unknown-'):
        return 'unknown'
    
    for standard_name, variants in EnhancedConfig.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    # Try to match by name
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower

def intelligent_segment_selection_ensemble(audio, sr, segment_duration=4.0, n_segments=3):
    """
    Enhanced segment selection for ensemble prediction
    Focuses on segments where target instrument is most prominent
    """
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return [np.pad(audio, (0, segment_len - len(audio)), mode='constant')]
    
    # Create overlapping segments with high overlap for ensemble
    hop_len = int(segment_len * (1 - EnhancedConfig.OVERLAP_RATIO))
    segments = []
    segment_scores = []
    
    for start in range(0, len(audio) - segment_len + 1, hop_len):
        segment = audio[start:start + segment_len]
        
        # Score based on TOP discriminative features from your analysis
        
        # 1. Pitch stability (most important feature)
        try:
            pitches = librosa.yin(segment, fmin=80, fmax=2000, sr=sr)
            valid_pitches = pitches[pitches > 0]
            if len(valid_pitches) > 10:
                pitch_stability = 1.0 / (1.0 + np.std(valid_pitches) / (np.mean(valid_pitches) + 1e-8))
            else:
                pitch_stability = 0
        except:
            pitch_stability = 0
        
        # 2. Harmonic content strength
        try:
            harmonic, _ = librosa.effects.hpss(segment, margin=(1.0, 5.0))
            harmonic_strength = np.sum(harmonic**2) / (np.sum(segment**2) + 1e-8)
        except:
            harmonic_strength = 0
        
        # 3. High-frequency energy variability (2nd most important feature)
        try:
            stft = np.abs(librosa.stft(segment, n_fft=EnhancedConfig.N_FFT))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=EnhancedConfig.N_FFT)
            high_freq_mask = freqs > 2000
            if np.any(high_freq_mask):
                high_freq_energy = np.sum(stft[high_freq_mask]**2, axis=0)
                high_freq_variability = np.std(high_freq_energy) / (np.mean(high_freq_energy) + 1e-8)
            else:
                high_freq_variability = 0
        except:
            high_freq_variability = 0
        
        # 4. Zero crossing rate consistency (3rd most important)
        try:
            zcr = librosa.feature.zero_crossing_rate(segment)[0]
            zcr_consistency = 1.0 / (1.0 + np.std(zcr))
        except:
            zcr_consistency = 0
        
        # 5. Overall energy (avoid very quiet segments)
        rms_energy = np.sqrt(np.mean(segment**2))
        energy_score = min(rms_energy * 20, 1.0)  # Normalize to 0-1
        
        # Combined score emphasizing your most discriminative features
        score = (pitch_stability * 0.35 +          # Most important
                harmonic_strength * 0.25 +         # Instrument vs background
                high_freq_variability * 0.2 +      # Second most important
                zcr_consistency * 0.15 +           # Third most important
                energy_score * 0.05)               # Basic threshold
        
        segments.append(segment)
        segment_scores.append(score)
    
    # Return top N segments for ensemble
    if len(segments) <= n_segments:
        return segments
    
    top_indices = np.argsort(segment_scores)[-n_segments:]
    return [segments[i] for i in top_indices]

def extract_enhanced_features(audio, sr):
    """Extract enhanced mel spectrogram with focus on discriminative features"""
    # Use intelligent segment selection
    best_segments = intelligent_segment_selection_ensemble(audio, sr, 
                                                         segment_duration=EnhancedConfig.SEGMENT_DURATION,
                                                         n_segments=1)
    best_segment = best_segments[0]
    
    # Harmonic-percussive separation for cleaner features
    harmonic, percussive = librosa.effects.hpss(best_segment, margin=(1.0, 5.0))
    
    # Use harmonic component for tonal instruments
    # But keep some percussive for attack detection
    enhanced_audio = harmonic * 0.8 + percussive * 0.2
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=enhanced_audio,
        sr=sr,
        n_fft=EnhancedConfig.N_FFT,
        hop_length=EnhancedConfig.HOP_LENGTH,
        n_mels=EnhancedConfig.N_MELS,
        fmax=EnhancedConfig.FMAX
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Enhanced normalization
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    # Apply slight high-frequency emphasis (based on your feature analysis)
    freq_weights = np.linspace(0.8, 1.2, EnhancedConfig.N_MELS)  # Emphasize higher mels
    mel_spec_normalized = mel_spec_normalized * freq_weights.reshape(-1, 1)
    
    return mel_spec_normalized

def conservative_augmentation(audio, sr):
    """Apply very conservative augmentation suitable for background music scenarios"""
    augmented_samples = [audio]  # Original
    
    if not EnhancedConfig.USE_AUGMENTATION:
        return augmented_samples
    
    try:
        # Very conservative time stretching
        stretch_factor = np.random.uniform(*EnhancedConfig.TIME_STRETCH_RANGE)
        stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
        augmented_samples.append(stretched)
        
        # Minimal pitch shifting
        pitch_shift = np.random.uniform(*EnhancedConfig.PITCH_SHIFT_RANGE)
        pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
        augmented_samples.append(pitch_shifted)
        
        # Very low noise addition
        noise = np.random.normal(0, EnhancedConfig.NOISE_FACTOR, len(audio))
        noisy = audio + noise
        augmented_samples.append(noisy)
        
    except Exception as e:
        print(f"Augmentation error: {e}")
        # Return at least the original if augmentation fails
    
    return augmented_samples

def build_enhanced_robust_model(input_shape, num_classes):
    """
    Build enhanced model architecture focusing on discriminative features
    Compatible with TensorFlow 2.10
    """
    
    # Input layer
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Multi-scale feature extraction based on your analysis
    
    # Fine-scale features (3x3) - for attack patterns, pin characteristics
    fine_branch = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    fine_branch = tf.keras.layers.BatchNormalization()(fine_branch)
    fine_branch = tf.keras.layers.Activation('relu')(fine_branch)
    fine_branch = tf.keras.layers.MaxPooling2D((2, 2))(fine_branch)
    fine_branch = tf.keras.layers.Dropout(0.2)(fine_branch)
    
    fine_branch = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(fine_branch)
    fine_branch = tf.keras.layers.BatchNormalization()(fine_branch)
    fine_branch = tf.keras.layers.Activation('relu')(fine_branch)
    fine_branch = tf.keras.layers.MaxPooling2D((2, 2))(fine_branch)
    fine_branch = tf.keras.layers.Dropout(0.25)(fine_branch)
    
    # Medium-scale features (5x5) - for harmonic patterns
    medium_branch = tf.keras.layers.Conv2D(32, (5, 5), padding='same')(inputs)
    medium_branch = tf.keras.layers.BatchNormalization()(medium_branch)
    medium_branch = tf.keras.layers.Activation('relu')(medium_branch)
    medium_branch = tf.keras.layers.MaxPooling2D((2, 2))(medium_branch)
    medium_branch = tf.keras.layers.Dropout(0.2)(medium_branch)
    
    medium_branch = tf.keras.layers.Conv2D(64, (5, 5), padding='same')(medium_branch)
    medium_branch = tf.keras.layers.BatchNormalization()(medium_branch)
    medium_branch = tf.keras.layers.Activation('relu')(medium_branch)
    medium_branch = tf.keras.layers.MaxPooling2D((2, 2))(medium_branch)
    medium_branch = tf.keras.layers.Dropout(0.25)(medium_branch)
    
    # Coarse-scale features (7x7) - for overall spectral shape
    coarse_branch = tf.keras.layers.Conv2D(32, (7, 7), padding='same')(inputs)
    coarse_branch = tf.keras.layers.BatchNormalization()(coarse_branch)
    coarse_branch = tf.keras.layers.Activation('relu')(coarse_branch)
    coarse_branch = tf.keras.layers.MaxPooling2D((2, 2))(coarse_branch)
    coarse_branch = tf.keras.layers.Dropout(0.2)(coarse_branch)
    
    coarse_branch = tf.keras.layers.Conv2D(64, (7, 7), padding='same')(coarse_branch)
    coarse_branch = tf.keras.layers.BatchNormalization()(coarse_branch)
    coarse_branch = tf.keras.layers.Activation('relu')(coarse_branch)
    coarse_branch = tf.keras.layers.MaxPooling2D((2, 2))(coarse_branch)
    coarse_branch = tf.keras.layers.Dropout(0.25)(coarse_branch)
    
    # Frequency attention mechanism (emphasize high frequencies)
    attention_weights = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='freq_attention')(inputs)
    attended_input = tf.keras.layers.Multiply()([inputs, attention_weights])
    
    attention_branch = tf.keras.layers.Conv2D(48, (3, 3), padding='same')(attended_input)
    attention_branch = tf.keras.layers.BatchNormalization()(attention_branch)
    attention_branch = tf.keras.layers.Activation('relu')(attention_branch)
    attention_branch = tf.keras.layers.MaxPooling2D((4, 4))(attention_branch)
    attention_branch = tf.keras.layers.Dropout(0.3)(attention_branch)
    
    # Global pooling for each branch
    fine_global = tf.keras.layers.GlobalAveragePooling2D()(fine_branch)
    medium_global = tf.keras.layers.GlobalAveragePooling2D()(medium_branch)
    coarse_global = tf.keras.layers.GlobalAveragePooling2D()(coarse_branch)
    attention_global = tf.keras.layers.GlobalAveragePooling2D()(attention_branch)
    
    # Concatenate all features
    merged = tf.keras.layers.Concatenate()([fine_global, medium_global, coarse_global, attention_global])
    
    # Dense layers with heavy regularization
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(EnhancedConfig.L2_REGULARIZATION))(merged)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(EnhancedConfig.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(EnhancedConfig.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(EnhancedConfig.DROPOUT_RATE)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def custom_label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """
    Custom label smoothing implementation for TensorFlow 2.10 compatibility - Simplified version
    """
    if smoothing == 0.0:
        # No smoothing, use standard sparse categorical crossentropy
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    
    # Get number of classes
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    
    # Ensure y_true is the right shape and type
    y_true = tf.cast(tf.squeeze(y_true), tf.int32)
    
    # Convert to one-hot
    y_true_one_hot = tf.one_hot(y_true, depth=tf.cast(num_classes, tf.int32))
    y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
    
    # Apply label smoothing
    smoothing = tf.cast(smoothing, tf.float32)
    y_true_smooth = y_true_one_hot * (1.0 - smoothing) + smoothing / num_classes
    
    # Use categorical crossentropy
    return tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)

def process_enhanced_dataset():
    """Process dataset with enhanced multi-segment approach"""
    print("Processing dataset with enhanced multi-segment approach...")
    
    # Collect all files
    all_files = []
    all_labels = []
    
    instrument_folders = [d for d in os.listdir(EnhancedConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(EnhancedConfig.DATA_PATH, d))]
    
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
            
        folder_path = os.path.join(EnhancedConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            all_files.append(file_path)
            all_labels.append(instrument)
    
    print(f"Total files collected: {len(all_files)}")
    
    # Train-test split at FILE level (before processing)
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels,
        test_size=EnhancedConfig.TEST_SIZE,
        random_state=EnhancedConfig.RANDOM_SEED,
        stratify=all_labels
    )
    
    print(f"Split: {len(train_files)} training files, {len(test_files)} testing files")
    
    # Process training files with multi-segment ensemble approach
    X_train = []
    y_train = []
    
    print("Processing training files with multi-segment ensemble...")
    for file_path, label in tqdm(zip(train_files, train_labels), total=len(train_files)):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=EnhancedConfig.SAMPLE_RATE)
            
            # Skip very short files
            if len(audio) < sr * 1.0:  # Less than 1 second
                continue
            
            # Get multiple segments for ensemble training
            segments = intelligent_segment_selection_ensemble(
                audio, sr, 
                segment_duration=EnhancedConfig.SEGMENT_DURATION,
                n_segments=EnhancedConfig.SEGMENTS_PER_AUDIO
            )
            
            # Process each segment
            for segment in segments:
                # Apply conservative augmentation
                augmented_segments = conservative_augmentation(segment, sr)
                
                for aug_segment in augmented_segments:
                    # Extract enhanced features
                    mel_spec = extract_enhanced_features(aug_segment, sr)
                    mel_spec_with_channel = np.expand_dims(mel_spec, axis=-1)
                    
                    X_train.append(mel_spec_with_channel)
                    y_train.append(label)
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Process test files (single best segment, no augmentation)
    X_test = []
    y_test = []
    
    print("Processing test files...")
    for file_path, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        try:
            audio, sr = librosa.load(file_path, sr=EnhancedConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 1.0:
                continue
            
            # Get single best segment for testing
            segments = intelligent_segment_selection_ensemble(audio, sr, n_segments=1)
            segment = segments[0]
            
            # Extract features (no augmentation for test)
            mel_spec = extract_enhanced_features(segment, sr)
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
    print(f"Training samples: {len(X_train)} (from {len(train_files)} files)")
    print(f"Test samples: {len(X_test)} (from {len(test_files)} files)")
    print(f"Feature shape: {X_train.shape[1:]}")
    print(f"Augmentation factor: ~{len(X_train)/len(train_files):.1f}x")
    
    return X_train, X_test, y_train, y_test, class_names

def plot_training_history(history, fold_num=None, save_path=None):
    """Plot training history with accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'Model Accuracy{"" if fold_num is None else f" - Fold {fold_num}"}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title(f'Model Loss{"" if fold_num is None else f" - Fold {fold_num}"}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history saved to: {save_path}")
    
    plt.show()
    return fig

def plot_comprehensive_results(test_results, class_names, save_path=None):
    """Create comprehensive visualization of results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Per-class F1 scores
    f1_scores = [test_results['per_class_metrics'][cls]['f1_score'] for cls in class_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(class_names)))
    
    bars1 = ax1.bar(class_names, f1_scores, color=colors)
    ax1.set_title('F1 Score by Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('F1 Score')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Precision vs Recall scatter plot
    precisions = [test_results['per_class_metrics'][cls]['precision'] for cls in class_names]
    recalls = [test_results['per_class_metrics'][cls]['recall'] for cls in class_names]
    
    scatter = ax2.scatter(precisions, recalls, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision vs Recall by Class', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    
    # Add class labels to points
    for i, cls in enumerate(class_names):
        ax2.annotate(cls, (precisions[i], recalls[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # 3. Support (number of samples) by class
    supports = [test_results['per_class_metrics'][cls]['support'] for cls in class_names]
    bars3 = ax3.bar(class_names, supports, color=colors)
    ax3.set_title('Test Samples by Class', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Samples')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, support in zip(bars3, supports):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{support}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Performance summary metrics
    metrics_names = ['Precision', 'Recall', 'F1-Score']
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    
    metrics_values = [avg_precision, avg_recall, avg_f1]
    bars4 = ax4.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Average Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars4, metrics_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add overall accuracy as text
    ax4.text(0.5, 0.5, f'Overall Accuracy\n{test_results["test_accuracy"]:.3f}', 
            transform=ax4.transAxes, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comprehensive results saved to: {save_path}")
    
    plt.show()
    return fig

def train_enhanced_model_kfold(X_train, y_train, X_test, y_test, class_names):
    """Train enhanced model with k-fold cross validation"""
    print(f"\nTraining Enhanced Robust Model with {EnhancedConfig.K_FOLDS}-fold CV...")
    
    # Convert labels to integers
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_to_int, f, indent=4)
    
    # K-fold cross validation
    if EnhancedConfig.USE_KFOLD:
        kf = StratifiedKFold(n_splits=EnhancedConfig.K_FOLDS, shuffle=True, 
                           random_state=EnhancedConfig.RANDOM_SEED)
        
        fold_results = []
        best_model = None
        best_val_acc = 0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_encoded)):
            print(f"\n--- Training Fold {fold+1}/{EnhancedConfig.K_FOLDS} ---")
            
            # Split data for this fold
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            print(f"Fold {fold+1}: {len(X_fold_train)} train, {len(X_fold_val)} val samples")
            
            # Class weights for this fold
            if EnhancedConfig.USE_CLASS_WEIGHTS:
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_fold_train),
                    y=y_fold_train
                )
                class_weight_dict = dict(enumerate(class_weights))
                print("Class weights:", {class_names[i]: f"{w:.3f}" for i, w in enumerate(class_weights)})
            else:
                class_weight_dict = None
            
            # Build model
            model = build_enhanced_robust_model(X_train.shape[1:], len(class_names))
            
            # Compile with custom loss function for TF 2.10
            def loss_fn(y_true, y_pred):
                return custom_label_smoothing_loss(y_true, y_pred, EnhancedConfig.LABEL_SMOOTHING)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=EnhancedConfig.LEARNING_RATE),
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=EnhancedConfig.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_path, f'fold_{fold+1}_best.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=EnhancedConfig.EPOCHS,
                batch_size=EnhancedConfig.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Evaluate fold
            val_loss, val_acc = model.evaluate(X_fold_val, y_fold_val, verbose=0)
            print(f"Fold {fold+1} validation accuracy: {val_acc:.4f}")
            
            # Store results
            fold_results.append({
                'fold': fold + 1,
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss)
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                print(f"New best model from fold {fold+1}")
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title(f'Fold {fold+1} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title(f'Fold {fold+1} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_path, f'fold_{fold+1}_history.png'), dpi=150)
            plt.close()
        
        # Print CV results
        fold_accuracies = [r['val_accuracy'] for r in fold_results]
        print(f"\nK-Fold CV Results:")
        print(f"Mean accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        
        # Save fold results
        with open(os.path.join(model_path, 'kfold_results.json'), 'w') as f:
            json.dump(fold_results, f, indent=4)
    
    else:
        # Single train without k-fold
        print("\nTraining single model...")
        best_model = build_enhanced_robust_model(X_train.shape[1:], len(class_names))
        
        # Compile model
        def loss_fn(y_true, y_pred):
            return custom_label_smoothing_loss(y_true, y_pred, EnhancedConfig.LABEL_SMOOTHING)
        
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=EnhancedConfig.LEARNING_RATE),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Class weights
        if EnhancedConfig.USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_encoded),
                y=y_train_encoded
            )
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=EnhancedConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(model_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        # Train
        history = best_model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=EnhancedConfig.EPOCHS,
            batch_size=EnhancedConfig.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
    
    # Final evaluation on test set
    if best_model is not None:
        print("\n" + "="*50)
        print("FINAL EVALUATION ON TEST SET")
        print("="*50)
        
        test_loss, test_acc = best_model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Generate predictions
        y_test_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_test_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Final Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(model_path, 'final_confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_test_pred, target_names=class_names))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, y_test_pred, average=None, labels=range(len(class_names))
        )
        
        # Save test results
        test_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist()
        }
        
        with open(os.path.join(model_path, 'final_test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Save the best model
        try:
            best_model.save(os.path.join(model_path, 'final_best_model.h5'))
            print("✓ Best model saved as final_best_model.h5")
        except Exception as e:
            print(f"Error saving model: {e}")
            try:
                best_model.save_weights(os.path.join(model_path, 'final_best_model_weights.h5'))
                print("✓ Model weights saved as fallback")
            except Exception as e2:
                print(f"Error saving weights: {e2}")
        
        # Convert to ONNX
        convert_to_onnx_enhanced(best_model, model_path, label_to_int)
        
        return best_model, test_results
    
    return None, None

def convert_to_onnx_enhanced(model, output_path, label_mapping):
    """Enhanced ONNX conversion with better error handling"""
    print("\nConverting model to ONNX format...")
    try:
        # Get model input signature
        input_signature = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name='mel_spectrogram')]
        
        # Convert to ONNX with specific opset version for better compatibility
        onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
        
        # Save ONNX model
        onnx_path = os.path.join(output_path, 'model.onnx')
        onnx.save_model(onnx_model, onnx_path)
        
        # Save label mapping
        with open(os.path.join(output_path, 'label_mapping.json'), 'w') as f:
            json.dump(label_mapping, f, indent=4)
        
        print(f"✓ ONNX model saved: {onnx_path}")
        print(f"✓ Label mapping saved: {os.path.join(output_path, 'label_mapping.json')}")
        
        # Test ONNX model
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path)
            print("✓ ONNX model validation successful")
            
            # Print input/output info
            print("ONNX Model Info:")
            for input_meta in session.get_inputs():
                print(f"  Input: {input_meta.name}, shape: {input_meta.shape}, type: {input_meta.type}")
            for output_meta in session.get_outputs():
                print(f"  Output: {output_meta.name}, shape: {output_meta.shape}, type: {output_meta.type}")
                
        except ImportError:
            print("⚠ onnxruntime not available for validation, but ONNX model saved")
        except Exception as e:
            print(f"⚠ ONNX model validation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ ONNX conversion failed: {e}")
        print("The TensorFlow model is still saved and functional")
        return False

def create_ensemble_inference_script():
    """Create a separate script for ensemble inference"""
    ensemble_script = '''
import numpy as np
import librosa
import onnxruntime as ort
import json
import os

class EnsembleInference:
    """
    Enhanced inference class that uses ensemble prediction
    from multiple segments for robust real-world performance
    """
    
    def __init__(self, model_path, label_mapping_path):
        self.session = ort.InferenceSession(model_path)
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.idx_to_label = {int(idx): label for label, idx in self.label_mapping.items()}
        
        # Audio parameters (must match training)
        self.SAMPLE_RATE = 44100
        self.SEGMENT_DURATION = 4.0
        self.N_FFT = 2048
        self.HOP_LENGTH = 512
        self.N_MELS = 128
        self.FMAX = 8000
    
    def intelligent_segment_selection(self, audio, sr, n_segments=3):
        """Same segment selection as training"""
        segment_len = int(self.SEGMENT_DURATION * sr)
        
        if len(audio) <= segment_len:
            return [np.pad(audio, (0, segment_len - len(audio)), mode='constant')]
        
        hop_len = int(segment_len * 0.33)  # 67% overlap
        segments = []
        scores = []
        
        for start in range(0, len(audio) - segment_len + 1, hop_len):
            segment = audio[start:start + segment_len]
            
            # Score segment quality
            try:
                # Pitch stability
                pitches = librosa.yin(segment, fmin=80, fmax=2000, sr=sr)
                valid_pitches = pitches[pitches > 0]
                if len(valid_pitches) > 10:
                    pitch_stability = 1.0 / (1.0 + np.std(valid_pitches) / (np.mean(valid_pitches) + 1e-8))
                else:
                    pitch_stability = 0
                
                # Harmonic content
                harmonic, _ = librosa.effects.hpss(segment, margin=(1.0, 5.0))
                harmonic_strength = np.sum(harmonic**2) / (np.sum(segment**2) + 1e-8)
                
                # High-frequency variability
                stft = np.abs(librosa.stft(segment, n_fft=self.N_FFT))
                freqs = librosa.fft_frequencies(sr=sr, n_fft=self.N_FFT)
                high_freq_mask = freqs > 2000
                if np.any(high_freq_mask):
                    high_freq_energy = np.sum(stft[high_freq_mask]**2, axis=0)
                    high_freq_var = np.std(high_freq_energy) / (np.mean(high_freq_energy) + 1e-8)
                else:
                    high_freq_var = 0
                
                # Energy
                rms_energy = np.sqrt(np.mean(segment**2))
                energy_score = min(rms_energy * 20, 1.0)
                
                # Combined score
                score = (pitch_stability * 0.35 + harmonic_strength * 0.25 + 
                        high_freq_var * 0.2 + energy_score * 0.2)
                
            except:
                score = np.sqrt(np.mean(segment**2))  # Fallback to RMS
            
            segments.append(segment)
            scores.append(score)
        
        # Return top segments
        if len(segments) <= n_segments:
            return segments
        
        top_indices = np.argsort(scores)[-n_segments:]
        return [segments[i] for i in top_indices]
    
    def extract_features(self, audio, sr):
        """Extract features matching training"""
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio, margin=(1.0, 5.0))
        enhanced_audio = harmonic * 0.8 + percussive * 0.2
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=enhanced_audio, sr=sr,
            n_fft=self.N_FFT, hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS, fmax=self.FMAX
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # High-frequency emphasis
        freq_weights = np.linspace(0.8, 1.2, self.N_MELS)
        mel_spec_normalized = mel_spec_normalized * freq_weights.reshape(-1, 1)
        
        return mel_spec_normalized
    
    def predict(self, audio_path_or_array, sr=None, confidence_threshold=0.4):
        """
        Make ensemble prediction with confidence estimation
        
        Args:
            audio_path_or_array: File path or audio array
            sr: Sample rate (if audio_array provided)
            confidence_threshold: Minimum confidence for definitive prediction
            
        Returns:
            dict with prediction results including uncertainty estimation
        """
        # Load audio
        if isinstance(audio_path_or_array, str):
            audio, sr = librosa.load(audio_path_or_array, sr=self.SAMPLE_RATE)
        else:
            audio = audio_path_or_array
            if sr != self.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        
        # Get multiple segments
        segments = self.intelligent_segment_selection(audio, self.SAMPLE_RATE, n_segments=3)
        
        predictions = []
        confidences = []
        
        # Predict on each segment
        for segment in segments:
            mel_spec = self.extract_features(segment, self.SAMPLE_RATE)
            features_batch = np.expand_dims(np.expand_dims(mel_spec, axis=-1), axis=0).astype(np.float32)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: features_batch})
            probabilities = outputs[0][0]
            
            predictions.append(probabilities)
            confidences.append(np.max(probabilities))
        
        # Ensemble prediction (confidence-weighted average)
        if len(predictions) > 1:
            weights = np.array(confidences) / (np.sum(confidences) + 1e-8)
            ensemble_probs = np.average(predictions, axis=0, weights=weights)
        else:
            ensemble_probs = predictions[0]
        
        # Final results
        max_prob_idx = np.argmax(ensemble_probs)
        max_prob = ensemble_probs[max_prob_idx]
        instrument = self.idx_to_label[max_prob_idx]
        
        # Calculate uncertainty
        entropy = -np.sum(ensemble_probs * np.log2(ensemble_probs + 1e-10)) / np.log2(len(ensemble_probs))
        is_uncertain = entropy > 0.6 or max_prob < confidence_threshold
        
        return {
            'instrument': instrument,
            'confidence': float(max_prob),
            'entropy': float(entropy),
            'is_uncertain': is_uncertain,
            'probabilities': {self.idx_to_label[i]: float(prob) for i, prob in enumerate(ensemble_probs)},
            'segments_used': len(predictions),
            'individual_confidences': confidences,
            'ensemble_std': float(np.std([np.max(p) for p in predictions]))
        }

# Example usage
if __name__ == "__main__":
    # Initialize ensemble inference
    predictor = EnsembleInference('model.onnx', 'label_mapping.json')
    
    # Make prediction
    result = predictor.predict('test_audio.wav')
    
    print(f"Predicted: {result['instrument']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Uncertain: {result['is_uncertain']}")
    print("All probabilities:")
    for instrument, prob in result['probabilities'].items():
        print(f"  {instrument}: {prob:.3f}")
'''
    
    # Save the ensemble inference script
    with open(os.path.join(model_path, 'ensemble_inference.py'), 'w') as f:
        f.write(ensemble_script)
    
    print(f"✓ Ensemble inference script saved: {os.path.join(model_path, 'ensemble_inference.py')}")

def main():
    """Main training function"""
    print("="*80)
    print("ENHANCED LAO INSTRUMENT CLASSIFIER - BACKGROUND MUSIC ROBUST")
    print("="*80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model will be saved to: {model_path}")
    
    # Save configuration
    save_config()
    print("✓ Configuration saved")
    
    # Process dataset
    print("\n1. PROCESSING DATASET...")
    X_train, X_test, y_train, y_test, class_names = process_enhanced_dataset()
    
    if len(X_train) == 0:
        print("❌ No training data found! Check your dataset path and structure.")
        return
    
    print("✓ Dataset processed successfully")
    
    # Train model
    print("\n2. TRAINING MODEL...")
    best_model, test_results = train_enhanced_model_kfold(X_train, y_train, X_test, y_test, class_names)
    
    if best_model is None:
        print("❌ Training failed!")
        return
    
    print("✓ Model training completed")
    
    # Create ensemble inference script
    print("\n3. CREATING INFERENCE TOOLS...")
    create_ensemble_inference_script()
    
    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*80)
    
    if test_results:
        print(f"📊 Final test accuracy: {test_results['test_accuracy']:.4f}")
        print(f"📁 Model saved to: {model_path}")
        print(f"🎯 Classes: {', '.join(class_names)}")
        
        print("\n📈 Per-class performance:")
        for class_name, metrics in test_results['per_class_metrics'].items():
            print(f"   {class_name}: F1={metrics['f1_score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")
    
    print("\n🚀 KEY IMPROVEMENTS IN THIS MODEL:")
    print("   ✓ Multi-segment ensemble prediction")
    print("   ✓ Focus on top discriminative features (pitch, high-freq energy, ZCR)")
    print("   ✓ Background music robustness")
    print("   ✓ Conservative augmentation")
    print("   ✓ Enhanced regularization")
    print("   ✓ Confidence-based uncertainty estimation")
    
    print("\n📝 FILES CREATED:")
    print(f"   • final_best_model.h5 - TensorFlow model")
    print(f"   • model.onnx - ONNX model for deployment")
    print(f"   • label_mapping.json - Class mappings")
    print(f"   • ensemble_inference.py - Enhanced inference script")
    print(f"   • final_test_results.json - Detailed results")
    print(f"   • config.json - Training configuration")
    
    print("\n💡 NEXT STEPS:")
    print("   1. Test the model using ensemble_inference.py")
    print("   2. If performance is still not satisfactory, consider:")
    print("      - Collecting more solo instrument recordings")
    print("      - Using the ensemble inference for better real-world performance")
    print("      - Implementing confidence thresholding in your application")
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Results saved to: {model_path}")

if __name__ == "__main__":
    main()