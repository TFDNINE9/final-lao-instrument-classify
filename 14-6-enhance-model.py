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

class FastConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # FAST augmentation (minimal for speed)
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.98, 1.02)  # Very minimal
    PITCH_SHIFT_RANGE = (-0.3, 0.3)    # Very minimal
    NOISE_FACTOR = 0.002               # Very low
    
    # Training parameters
    BATCH_SIZE = 32         # Larger batch for speed
    EPOCHS = 80             # Reduced epochs
    LEARNING_RATE = 0.0003  
    EARLY_STOPPING_PATIENCE = 15
    
    # Regularization
    DROPOUT_RATE = 0.5
    L2_REGULARIZATION = 0.008
    LABEL_SMOOTHING = 0.1
    
    # SPEED OPTIMIZATIONS
    SINGLE_SEGMENT_TRAINING = True    # Only use 1 segment per file for training
    FAST_SEGMENT_SELECTION = True     # Simplified segment selection
    AUGMENTATION_RATIO = 2.0          # Limit augmentation multiplier
    
    # Cross-validation
    K_FOLDS = 3                       # Reduced folds for speed
    USE_KFOLD = True
    
    # Class balancing
    USE_CLASS_WEIGHTS = True
    
    # Train/test split
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/fast_robust_model"
    
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
model_path = f"{FastConfig.MODEL_SAVE_PATH}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(model_path, exist_ok=True)

def save_config():
    """Save configuration parameters to a JSON file"""
    config_dict = {key: value for key, value in FastConfig.__dict__.items() 
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
    
    for standard_name, variants in FastConfig.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower

def fast_segment_selection(audio, sr, segment_duration=6.0):
    """
    FAST segment selection - much simpler but still effective
    Uses only basic metrics for speed
    """
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Create fewer segments with less overlap for speed
    hop_len = segment_len // 2  # 50% overlap instead of 67%
    segments = []
    scores = []
    
    for start in range(0, len(audio) - segment_len + 1, hop_len):
        segment = audio[start:start + segment_len]
        
        # SIMPLIFIED scoring for speed - only use fast metrics
        rms = np.sqrt(np.mean(segment**2))
        
        # Basic spectral centroid (faster than full analysis)
        try:
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
            score = rms * 0.7 + (spectral_centroid / 4000) * 0.3  # Normalize centroid
        except:
            score = rms  # Fallback to just RMS
        
        segments.append(segment)
        scores.append(score)
    
    # Return best segment (single segment for training speed)
    if not segments:
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    best_idx = np.argmax(scores)
    return segments[best_idx]

def extract_fast_features(audio, sr):
    """Extract features quickly with minimal processing"""
    # Use fast segment selection
    best_segment = fast_segment_selection(audio, sr, FastConfig.SEGMENT_DURATION)
    
    # OPTIONAL: Light harmonic separation (can be disabled for even more speed)
    if FastConfig.FAST_SEGMENT_SELECTION:
        # Skip harmonic separation for maximum speed
        enhanced_audio = best_segment
    else:
        # Light harmonic separation
        harmonic, percussive = librosa.effects.hpss(best_segment, margin=(1.0, 2.0))
        enhanced_audio = harmonic * 0.9 + percussive * 0.1
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=enhanced_audio,
        sr=sr,
        n_fft=FastConfig.N_FFT,
        hop_length=FastConfig.HOP_LENGTH,
        n_mels=FastConfig.N_MELS,
        fmax=FastConfig.FMAX
    )
    
    # Convert to dB and normalize
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return mel_spec_normalized

def fast_augmentation(audio, sr):
    """Very fast augmentation with minimal operations"""
    if not FastConfig.USE_AUGMENTATION:
        return [audio]
    
    augmented = [audio]  # Original
    
    # Limit augmentation for speed
    num_augmentations = int(FastConfig.AUGMENTATION_RATIO) - 1
    
    try:
        if num_augmentations >= 1:
            # Time stretch (minimal)
            stretch_factor = np.random.uniform(*FastConfig.TIME_STRETCH_RANGE)
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            else:
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), mode='constant')
            augmented.append(stretched)
        
        if num_augmentations >= 2:
            # Add noise (very minimal)
            noise = np.random.normal(0, FastConfig.NOISE_FACTOR, len(audio))
            noisy = audio + noise
            augmented.append(noisy)
            
    except Exception as e:
        print(f"Augmentation warning: {e}")
    
    return augmented

def build_fast_model(input_shape, num_classes):
    """
    Build a FASTER model - fewer layers but still effective
    Based on your diagnostic findings but optimized for speed
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # Simplified architecture for speed
    # Still multi-scale but fewer operations
    
    # Branch 1: Fine features (3x3)
    x1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.2)(x1)
    
    x1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.MaxPooling2D((2, 2))(x1)
    x1 = tf.keras.layers.Dropout(0.3)(x1)
    
    # Branch 2: Coarse features (5x5) 
    x2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Dropout(0.2)(x2)
    
    x2 = tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling2D((2, 2))(x2)
    x2 = tf.keras.layers.Dropout(0.3)(x2)
    
    # Simple attention for high frequencies
    attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(inputs)
    attended = tf.keras.layers.Multiply()([inputs, attention])
    
    x3 = tf.keras.layers.Conv2D(48, (3, 3), padding='same', activation='relu')(attended)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.MaxPooling2D((4, 4))(x3)
    x3 = tf.keras.layers.Dropout(0.3)(x3)
    
    # Global pooling and concatenate
    g1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    g2 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    g3 = tf.keras.layers.GlobalAveragePooling2D()(x3)
    
    merged = tf.keras.layers.Concatenate()([g1, g2, g3])
    
    # Simplified dense layers
    x = tf.keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(FastConfig.L2_REGULARIZATION))(merged)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(FastConfig.DROPOUT_RATE)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(FastConfig.L2_REGULARIZATION))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(FastConfig.DROPOUT_RATE)(x)
    
    # Output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def custom_label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    """Custom label smoothing for TF 2.10 - Simplified version using sparse categorical"""
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

def process_fast_dataset():
    """Process dataset with SPEED optimizations"""
    print("Processing dataset with FAST approach...")
    
    # Collect files
    all_files = []
    all_labels = []
    
    instrument_folders = [d for d in os.listdir(FastConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(FastConfig.DATA_PATH, d))]
    
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
            
        folder_path = os.path.join(FastConfig.DATA_PATH, folder)
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
        test_size=FastConfig.TEST_SIZE,
        random_state=FastConfig.RANDOM_SEED,
        stratify=all_labels
    )
    
    print(f"Split: {len(train_files)} train, {len(test_files)} test files")
    
    # Process training files - SINGLE SEGMENT ONLY for speed
    X_train = []
    y_train = []
    
    print("Processing training files (FAST mode)...")
    for file_path, label in tqdm(zip(train_files, train_labels), total=len(train_files)):
        try:
            audio, sr = librosa.load(file_path, sr=FastConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.8:  # Skip very short files
                continue
            
            # SINGLE segment only (not multiple)
            segment = fast_segment_selection(audio, sr, FastConfig.SEGMENT_DURATION)
            
            # Apply fast augmentation
            augmented_segments = fast_augmentation(segment, sr)
            
            for aug_segment in augmented_segments:
                mel_spec = extract_fast_features(aug_segment, sr)
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
            audio, sr = librosa.load(file_path, sr=FastConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.8:
                continue
            
            # Single segment, no augmentation
            segment = fast_segment_selection(audio, sr, FastConfig.SEGMENT_DURATION)
            mel_spec = extract_fast_features(segment, sr)
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
    print(f"Speed optimization: ~{len(X_train)/len(train_files):.1f}x per file (vs 3-6x before)")
    
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
    f1_scores = [test_results['per_class'][cls]['f1_score'] for cls in class_names]
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
    precisions = [test_results['per_class'][cls]['precision'] for cls in class_names]
    recalls = [test_results['per_class'][cls]['recall'] for cls in class_names]
    
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
    supports = [test_results['per_class'][cls]['support'] for cls in class_names]
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

def train_fast_model(X_train, y_train, X_test, y_test, class_names):
    """Train model with speed optimizations"""
    print(f"\nTraining FAST model with {FastConfig.K_FOLDS}-fold CV...")
    
    # Convert labels
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    # Save label mapping
    with open(os.path.join(model_path, 'label_mapping.json'), 'w') as f:
        json.dump(label_to_int, f, indent=4)
    
    if FastConfig.USE_KFOLD:
        kf = StratifiedKFold(n_splits=FastConfig.K_FOLDS, shuffle=True, 
                           random_state=FastConfig.RANDOM_SEED)
        
        fold_results = []
        best_model = None
        best_val_acc = 0
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train_encoded)):
            print(f"\n--- Fold {fold+1}/{FastConfig.K_FOLDS} ---")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            
            print(f"Fold {fold+1}: {len(X_fold_train)} train, {len(X_fold_val)} val")
            
            # Class weights
            if FastConfig.USE_CLASS_WEIGHTS:
                class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_fold_train),
                    y=y_fold_train
                )
                class_weight_dict = dict(enumerate(class_weights))
            else:
                class_weight_dict = None
            
            # Build model
            model = build_fast_model(X_train.shape[1:], len(class_names))
            
            # Compile
            def loss_fn(y_true, y_pred):
                return custom_label_smoothing_loss(y_true, y_pred, FastConfig.LABEL_SMOOTHING)
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=FastConfig.LEARNING_RATE),
                loss=loss_fn,
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=FastConfig.EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=6,  # Faster LR reduction
                    min_lr=1e-7
                )
            ]
            
            # Train
            print(f"Training fold {fold+1}...")
            history = model.fit(
                X_fold_train, y_fold_train,
                validation_data=(X_fold_val, y_fold_val),
                epochs=FastConfig.EPOCHS,
                batch_size=FastConfig.BATCH_SIZE,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Plot and save training history for this fold
            history_plot_path = os.path.join(model_path, f'fold_{fold+1}_training_history.png')
            plot_training_history(history, fold_num=fold+1, save_path=history_plot_path)
            
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
    
    else:
        # Single model training
        print("Training single model...")
        best_model = build_fast_model(X_train.shape[1:], len(class_names))
        
        def loss_fn(y_true, y_pred):
            return custom_label_smoothing_loss(y_true, y_pred, FastConfig.LABEL_SMOOTHING)
        
        best_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=FastConfig.LEARNING_RATE),
            loss=loss_fn,
            metrics=['accuracy']
        )
        
        # Split for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
        )
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=FastConfig.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7
            )
        ]
        
        # Class weights
        if FastConfig.USE_CLASS_WEIGHTS:
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_encoded),
                y=y_train_encoded
            )
            class_weight_dict = dict(enumerate(class_weights))
        else:
            class_weight_dict = None
        
        history = best_model.fit(
            X_train_split, y_train_split,
            validation_data=(X_val_split, y_val_split),
            epochs=FastConfig.EPOCHS,
            batch_size=FastConfig.BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Plot training history for single model
        history_plot_path = os.path.join(model_path, 'training_history.png')
        plot_training_history(history, save_path=history_plot_path)
    
    # Final test evaluation
    if best_model is not None:
        print("\n" + "="*50)
        print("FINAL TEST EVALUATION")
        print("="*50)
        
        test_loss, test_acc = best_model.evaluate(X_test, y_test_encoded, verbose=0)
        print(f"Test accuracy: {test_acc:.4f}")
        
        # Predictions and confusion matrix
        y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Test Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(model_path, 'confusion_matrix.png'), dpi=150)
        plt.close()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=class_names))
        
        # Save results
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test_encoded, y_pred, average=None, labels=range(len(class_names))
        )
        
        test_results = {
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'per_class': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            }
        }
        
        with open(os.path.join(model_path, 'test_results.json'), 'w') as f:
            json.dump(test_results, f, indent=4)
        
        # Create comprehensive results visualization
        results_plot_path = os.path.join(model_path, 'comprehensive_results.png')
        plot_comprehensive_results(test_results, class_names, save_path=results_plot_path)
        
        # Save model
        try:
            best_model.save(os.path.join(model_path, 'fast_model.h5'))
            print("✓ Model saved")
        except Exception as e:
            print(f"Model save error: {e}")
        
        # Convert to ONNX
        try:
            input_signature = [tf.TensorSpec(best_model.inputs[0].shape, tf.float32)]
            onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=input_signature)
            onnx.save_model(onnx_model, os.path.join(model_path, 'model.onnx'))
            print("✓ ONNX model saved")
        except Exception as e:
            print(f"ONNX conversion error: {e}")
        
        return best_model, test_results
    
    return None, None

def main():
    """Main training function optimized for SPEED"""
    print("="*60)
    print("FAST LAO INSTRUMENT CLASSIFIER")
    print("="*60)
    print("🚀 Speed optimizations enabled:")
    print("   • Single segment per file (vs 3 segments)")
    print("   • Simplified feature extraction")
    print("   • Minimal augmentation")
    print("   • Faster model architecture")
    print("   • Reduced k-folds (3 vs 5)")
    print("="*60)
    
    # Save config
    save_config()
    
    # Process dataset
    print("\n1. PROCESSING DATASET...")
    start_time = datetime.now()
    
    X_train, X_test, y_train, y_test, class_names = process_fast_dataset()
    
    if len(X_train) == 0:
        print("❌ No training data found!")
        return
    
    processing_time = datetime.now() - start_time
    print(f"✓ Dataset processed in {processing_time}")
    
    # Train model
    print("\n2. TRAINING MODEL...")
    train_start = datetime.now()
    
    best_model, results = train_fast_model(X_train, y_train, X_test, y_test, class_names)
    
    training_time = datetime.now() - train_start
    total_time = datetime.now() - start_time
    
    print(f"\n✓ Training completed in {training_time}")
    print(f"✓ Total time: {total_time}")
    
    if results:
        print(f"\n📊 Final Results:")
        print(f"   Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"   Model saved to: {model_path}")
        
        print("\n🎯 Per-class F1 scores:")
        for cls, metrics in results['per_class'].items():
            print(f"   {cls}: {metrics['f1_score']:.3f}")
    
    print("\n💡 Next steps:")
    print("   • Test with your real audio files")
    print("   • Use ensemble inference for better accuracy") 
    print("   • The ensemble script will still work with this faster model!")

if __name__ == "__main__":
    main()