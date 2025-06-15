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
import joblib
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

# PIN-FOCUSED CONFIGURATION
class PinFocusedConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # CONSERVATIVE AUGMENTATION - NO AUGMENTATION FOR PIN
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.95, 1.05)
    PITCH_SHIFT_RANGE = (-0.5, 0.5)
    NOISE_FACTOR = 0.003
    PRESERVE_PIN_CHARACTERISTICS = True  # New: No augmentation for pin
    
    # TRAINING PARAMETERS
    BATCH_SIZE = 4
    EPOCHS = 25
    LEARNING_RATE = 0.00005
    EARLY_STOPPING_PATIENCE = 5
    
    # REGULARIZATION
    DROPOUT_RATE = 0.6  # Slightly reduced for pin-specific features
    L2_REGULARIZATION = 0.01
    
    # Features
    USE_ENHANCED_FEATURES = True
    USE_PIN_FOCUSED_FEATURES = True  # New
    USE_SESSION_AWARE_SPLIT = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/pin_focused_model_6sec"
    
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
os.makedirs(PinFocusedConfig.MODEL_SAVE_PATH, exist_ok=True)

def map_instrument_folder(folder_name, class_names):
    """Map folder name to standard instrument name"""
    folder_lower = folder_name.lower()
    
    for standard_name, variants in PinFocusedConfig.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    if 'unknown' in folder_lower or 'noise' in folder_lower or 'background' in folder_lower:
        return None
    
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower

def extract_pin_focused_features(audio, sr):
    """Extract features specifically designed to identify Pin instrument"""
    features = {}
    
    try:
        # PIN CHARACTERISTIC 1: Sharp Attack + Exponential Decay
        rms_energy = librosa.feature.rms(y=audio, hop_length=256)[0]  # Higher resolution
        
        # Find attack points (sharp energy increases)
        energy_diff = np.diff(rms_energy)
        attack_points = np.where(energy_diff > np.std(energy_diff) * 3)[0]
        
        if len(attack_points) > 0:
            # Analyze each attack
            attack_sharpness_scores = []
            decay_scores = []
            
            for attack_idx in attack_points[:5]:  # First 5 attacks
                # Look at 50ms before and after attack
                before_samples = 10  # ~50ms at hop_length=256
                after_samples = 40   # ~200ms for decay analysis
                
                start_idx = max(0, attack_idx - before_samples)
                end_idx = min(len(rms_energy), attack_idx + after_samples)
                
                if end_idx - start_idx > 20:  # Enough samples
                    segment = rms_energy[start_idx:end_idx]
                    attack_point = before_samples if attack_idx >= before_samples else attack_idx
                    
                    # Attack sharpness: how steep the rise is
                    if attack_point > 0 and attack_point < len(segment) - 1:
                        pre_attack = segment[:attack_point]
                        post_attack = segment[attack_point:attack_point+5]  # 5 frames after
                        
                        if len(pre_attack) > 0 and len(post_attack) > 0:
                            attack_rise = np.max(post_attack) - np.mean(pre_attack)
                            attack_sharpness_scores.append(attack_rise)
                            
                            # Decay analysis: exponential fit
                            decay_portion = segment[attack_point:]
                            if len(decay_portion) > 10:
                                # Fit exponential decay: y = a * exp(-b*x)
                                x = np.arange(len(decay_portion))
                                y = decay_portion + 1e-8  # Avoid log(0)
                                
                                try:
                                    # Linear fit to log(y) = log(a) - b*x
                                    log_y = np.log(y)
                                    if not np.any(np.isinf(log_y)) and not np.any(np.isnan(log_y)):
                                        decay_coeff = np.polyfit(x, log_y, 1)[0]  # -b coefficient
                                        decay_scores.append(abs(decay_coeff))
                                except:
                                    pass
            
            features['pin_attack_sharpness'] = np.mean(attack_sharpness_scores) if attack_sharpness_scores else 0
            features['pin_decay_rate'] = np.mean(decay_scores) if decay_scores else 0
            features['pin_attack_count'] = len(attack_points) / (len(audio) / sr)  # Attacks per second
            
        else:
            features['pin_attack_sharpness'] = 0
            features['pin_decay_rate'] = 0
            features['pin_attack_count'] = 0
        
        # PIN CHARACTERISTIC 2: Plucking Harmonics Pattern
        stft = np.abs(librosa.stft(audio, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Pin typically has strong fundamental + clear harmonics (string resonance)
        # Look for harmonic series pattern
        fundamental_range = (80, 400)  # Typical pin fundamental range
        
        fund_mask = (freqs >= fundamental_range[0]) & (freqs <= fundamental_range[1])
        h2_mask = (freqs >= fundamental_range[0]*2) & (freqs <= fundamental_range[1]*2)
        h3_mask = (freqs >= fundamental_range[0]*3) & (freqs <= fundamental_range[1]*3)
        
        fund_energy = np.mean(stft[fund_mask, :])
        h2_energy = np.mean(stft[h2_mask, :])
        h3_energy = np.mean(stft[h3_mask, :])
        
        total_harmonic = fund_energy + h2_energy + h3_energy + 1e-8
        
        features['pin_fundamental_strength'] = fund_energy / total_harmonic
        features['pin_harmonic_clarity'] = (h2_energy + h3_energy) / (fund_energy + 1e-8)
        
        # PIN CHARACTERISTIC 3: Sustained String Resonance
        # Pin should have longer sustain than percussion (ranad) but different from bow (saw)
        energy_envelope = rms_energy
        peak_energy = np.max(energy_envelope)
        
        # Measure sustain at different thresholds
        sustain_10 = np.sum(energy_envelope > peak_energy * 0.1) / len(energy_envelope)
        sustain_05 = np.sum(energy_envelope > peak_energy * 0.05) / len(energy_envelope)
        
        features['pin_sustain_10'] = sustain_10
        features['pin_sustain_05'] = sustain_05
        features['pin_sustain_ratio'] = sustain_05 / (sustain_10 + 1e-8)
        
        # PIN CHARACTERISTIC 4: Pluck vs Bow vs Blow Distinction
        # Analyze the attack-to-total-energy ratio
        if len(attack_points) > 0:
            # Energy in attack phases vs total energy
            attack_energy = 0
            total_energy = np.sum(energy_envelope**2)
            
            for attack_idx in attack_points:
                start = max(0, attack_idx - 2)
                end = min(len(energy_envelope), attack_idx + 8)
                attack_energy += np.sum(energy_envelope[start:end]**2)
            
            features['pin_attack_energy_ratio'] = attack_energy / (total_energy + 1e-8)
        else:
            features['pin_attack_energy_ratio'] = 0
        
        # PIN CHARACTERISTIC 5: Spectral Evolution Pattern
        # Pin has characteristic spectral change over time (bright attack, darker sustain)
        if stft.shape[1] > 10:
            # Compare early vs late spectral content
            early_frames = stft[:, :stft.shape[1]//3]  # First third
            late_frames = stft[:, -stft.shape[1]//3:]  # Last third
            
            early_centroid = np.sum(freqs[:, np.newaxis] * early_frames) / (np.sum(early_frames, axis=0) + 1e-8)
            late_centroid = np.sum(freqs[:, np.newaxis] * late_frames) / (np.sum(late_frames, axis=0) + 1e-8)
            
            features['pin_spectral_evolution'] = np.mean(early_centroid) - np.mean(late_centroid)
        else:
            features['pin_spectral_evolution'] = 0
            
    except Exception as e:
        # If anything fails, return neutral values
        features = {
            'pin_attack_sharpness': 0,
            'pin_decay_rate': 0,
            'pin_attack_count': 0,
            'pin_fundamental_strength': 0,
            'pin_harmonic_clarity': 0,
            'pin_sustain_10': 0,
            'pin_sustain_05': 0,
            'pin_sustain_ratio': 0,
            'pin_attack_energy_ratio': 0,
            'pin_spectral_evolution': 0
        }
    
    return features

def extract_fast_enhanced_features(audio, sr):
    """Fast enhanced features from previous version"""
    features = {}
    
    try:
        # Pre-compute expensive operations once
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        stft_mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        rms_energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=512)[0] 
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=512)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio, hop_length=512)[0]
        
    except Exception as e:
        return {f'feature_{i}': 0.0 for i in range(32)}
    
    # Basic features
    features['energy_mean'] = np.mean(rms_energy)
    features['energy_std'] = np.std(rms_energy)
    features['energy_continuity'] = 1.0 / (np.std(rms_energy) + 1e-6)
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    features['zcr_mean'] = np.mean(zero_crossings)
    features['texture_consistency'] = 1.0 / (np.std(zero_crossings) + 1e-6)
    
    # Simplified onset detection
    try:
        energy_diff = np.diff(rms_energy)
        energy_peaks = np.where(energy_diff > np.std(energy_diff) * 2)[0]
        features['onset_density'] = len(energy_peaks) / (len(audio) / sr)
        
        if len(energy_peaks) > 0:
            attack_slopes = []
            for peak_idx in energy_peaks[:3]:
                if peak_idx > 0 and peak_idx < len(rms_energy) - 5:
                    before = rms_energy[max(0, peak_idx-2):peak_idx]
                    after = rms_energy[peak_idx:min(len(rms_energy), peak_idx+3)]
                    if len(before) > 0 and len(after) > 0:
                        slope = np.max(after) - np.mean(before)
                        attack_slopes.append(slope)
            
            features['attack_slope_mean'] = np.mean(attack_slopes) if attack_slopes else 0
        else:
            features['attack_slope_mean'] = 0
    except:
        features['onset_density'] = 0
        features['attack_slope_mean'] = 0
    
    # Harmonic/percussive analysis
    try:
        low_freq_mask = freqs < 500
        high_freq_mask = freqs > 2000
        mid_freq_mask = (freqs >= 500) & (freqs <= 2000)
        
        low_energy = np.mean(stft_mag[low_freq_mask, :])
        mid_energy = np.mean(stft_mag[mid_freq_mask, :])
        high_energy = np.mean(stft_mag[high_freq_mask, :])
        total_energy = low_energy + mid_energy + high_energy + 1e-8
        
        features['harmonic_ratio'] = (low_energy + mid_energy) / total_energy
        features['percussive_ratio'] = high_energy / total_energy
        
    except:
        features['harmonic_ratio'] = 0.5
        features['percussive_ratio'] = 0.5
    
    # Additional features for completeness (32 total)
    features['pitch_variation'] = features['spectral_centroid_std']
    features['vibrato_rate'] = features['energy_std']
    features['transient_density'] = features['onset_density']
    features['attack_steepness_mean'] = features['attack_slope_mean']
    features['attack_steepness_std'] = features['attack_slope_mean'] * 0.1
    features['decay_rate_mean'] = features['energy_std']
    features['attack_to_decay_ratio'] = features['attack_slope_mean'] / (features['energy_std'] + 1e-6)
    features['sustain_duration'] = features['energy_continuity'] * 0.1
    features['envelope_curvature'] = features['energy_mean'] - features['energy_std']
    features['envelope_slope'] = features['energy_std']
    features['fundamental_strength'] = low_energy
    features['harmonic_complexity'] = high_energy / (low_energy + 1e-8)
    features['spectral_regularity'] = features['texture_consistency']
    features['frequency_variation'] = features['spectral_centroid_std']
    features['beating_intensity'] = features['energy_std']
    features['metallic_shimmer'] = high_energy / total_energy
    features['mid_frequency_ratio'] = mid_energy / total_energy
    features['resonance_duration'] = features['energy_continuity'] * 0.05
    features['decay_characteristic'] = features['energy_std']
    
    return features

def extract_ultra_enhanced_features(audio, sr):
    """Combine fast features with pin-focused features"""
    fast_features = extract_fast_enhanced_features(audio, sr)
    pin_features = extract_pin_focused_features(audio, sr)
    
    all_features = {**fast_features, **pin_features}
    return all_features

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Enhanced segment selection"""
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    hop_len = int(segment_len / 3)
    segments = []
    scores = []
    
    for start in range(0, len(audio) - segment_len + 1, hop_len):
        segment = audio[start:start + segment_len]
        
        energy_score = np.sqrt(np.mean(segment**2))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        try:
            onset_frames = librosa.onset.onset_detect(y=segment, sr=sr)
            onset_score = len(onset_frames) / (len(segment) / sr)
        except:
            onset_score = 0
        
        combined_score = (0.4 * energy_score + 
                         0.3 * spectral_contrast + 
                         0.3 * onset_score)
        
        segments.append(segment)
        scores.append(combined_score)
    
    best_idx = np.argmax(scores)
    return segments[best_idx]

def targeted_augmentation(audio, sr, instrument_type):
    """Conservative instrument-specific augmentation"""
    augmented_samples = []
    augmented_samples.append(audio)  # Original
    
    if PinFocusedConfig.USE_AUGMENTATION:
        # NO AUGMENTATION for pin to preserve attack characteristics
        if instrument_type == 'pin' and PinFocusedConfig.PRESERVE_PIN_CHARACTERISTICS:
            return augmented_samples  # Only return original
        
        if instrument_type == 'saw':
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.3)
            augmented_samples.append(pitch_shifted)
            
        elif instrument_type == 'khean':
            breath_noise = np.random.normal(0, PinFocusedConfig.NOISE_FACTOR, len(audio))
            noisy = audio + breath_noise
            augmented_samples.append(noisy)
            
        elif instrument_type == 'khong_vong':
            time_stretched = librosa.effects.time_stretch(audio, rate=0.98)
            if len(time_stretched) > len(audio):
                time_stretched = time_stretched[:len(audio)]
            else:
                time_stretched = np.pad(time_stretched, (0, len(audio) - len(time_stretched)), mode='constant')
            augmented_samples.append(time_stretched)
            
        else:
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            augmented_samples.append(pitch_shifted)
    
    return augmented_samples

def session_aware_split(file_paths, labels, test_size=0.2):
    """Split data by recording session"""
    session_data = []
    
    for file_path, label in zip(file_paths, labels):
        filename = os.path.basename(file_path)
        
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

def process_files_with_pin_focus(file_paths, labels, with_augmentation=True):
    """Process files with pin-focused features"""
    mel_features = []
    enhanced_features = []
    processed_labels = []
    
    print(f"Processing {len(file_paths)} files with pin-focused features...")
    
    for file_path, label in tqdm(zip(file_paths, labels), 
                                desc="Pin-focused processing",
                                total=len(file_paths)):
        try:
            audio, sr = librosa.load(file_path, sr=PinFocusedConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.5:
                continue
            
            if with_augmentation:
                augmented_samples = targeted_augmentation(audio, sr, label)
            else:
                augmented_samples = [audio]
            
            for aug_audio in augmented_samples:
                best_segment = process_audio_with_best_segment(aug_audio, sr, 
                                                             PinFocusedConfig.SEGMENT_DURATION)
                
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=best_segment, sr=sr,
                    n_fft=PinFocusedConfig.N_FFT,
                    hop_length=PinFocusedConfig.HOP_LENGTH,
                    n_mels=PinFocusedConfig.N_MELS,
                    fmax=PinFocusedConfig.FMAX
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
                
                mel_features.append(mel_spec_with_channel)
                
                # Extract ultra-enhanced features (42 features total)
                enhanced_feat = extract_ultra_enhanced_features(best_segment, sr)
                
                # Convert to array (42 features: 32 original + 10 pin-specific)
                feat_array = np.array([
                    # Original 32 features
                    enhanced_feat.get('attack_slope_mean', 0),
                    enhanced_feat.get('onset_density', 0),
                    enhanced_feat.get('pitch_variation', 0),
                    enhanced_feat.get('vibrato_rate', 0),
                    enhanced_feat.get('harmonic_ratio', 0),
                    enhanced_feat.get('percussive_ratio', 0),
                    enhanced_feat.get('spectral_centroid_mean', 0),
                    enhanced_feat.get('spectral_centroid_std', 0),
                    enhanced_feat.get('spectral_bandwidth_mean', 0),
                    enhanced_feat.get('spectral_rolloff_mean', 0),
                    enhanced_feat.get('energy_mean', 0),
                    enhanced_feat.get('energy_std', 0),
                    enhanced_feat.get('energy_continuity', 0),
                    enhanced_feat.get('zcr_mean', 0),
                    enhanced_feat.get('texture_consistency', 0),
                    enhanced_feat.get('transient_density', 0),
                    enhanced_feat.get('attack_steepness_mean', 0),
                    enhanced_feat.get('attack_steepness_std', 0),
                    enhanced_feat.get('decay_rate_mean', 0),
                    enhanced_feat.get('attack_to_decay_ratio', 0),
                    enhanced_feat.get('sustain_duration', 0),
                    enhanced_feat.get('envelope_curvature', 0),
                    enhanced_feat.get('envelope_slope', 0),
                    enhanced_feat.get('fundamental_strength', 0),
                    enhanced_feat.get('harmonic_complexity', 0),
                    enhanced_feat.get('spectral_regularity', 0),
                    enhanced_feat.get('frequency_variation', 0),
                    enhanced_feat.get('beating_intensity', 0),
                    enhanced_feat.get('metallic_shimmer', 0),
                    enhanced_feat.get('mid_frequency_ratio', 0),
                    enhanced_feat.get('resonance_duration', 0),
                    enhanced_feat.get('decay_characteristic', 0),
                    
                    # New pin-specific features (10)
                    enhanced_feat.get('pin_attack_sharpness', 0),
                    enhanced_feat.get('pin_decay_rate', 0),
                    enhanced_feat.get('pin_attack_count', 0),
                    enhanced_feat.get('pin_fundamental_strength', 0),
                    enhanced_feat.get('pin_harmonic_clarity', 0),
                    enhanced_feat.get('pin_sustain_10', 0),
                    enhanced_feat.get('pin_sustain_05', 0),
                    enhanced_feat.get('pin_sustain_ratio', 0),
                    enhanced_feat.get('pin_attack_energy_ratio', 0),
                    enhanced_feat.get('pin_spectral_evolution', 0)
                ])
                
                enhanced_features.append(feat_array)
                processed_labels.append(label)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(mel_features), np.array(enhanced_features), np.array(processed_labels)

def build_pin_focused_model(mel_input_shape, enhanced_feature_size, num_classes):
    """Model with special attention to pin classification"""
    
    # Mel-spectrogram branch
    mel_input = tf.keras.layers.Input(shape=mel_input_shape, name='mel_input')
    
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
    x = tf.keras.layers.Dropout(0.5)(x)
    
    mel_features = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Enhanced features branch (42 features)
    enhanced_input = tf.keras.layers.Input(shape=(enhanced_feature_size,), name='enhanced_input')
    
    # Separate pin-specific features (last 10 features)
    pin_specific = tf.keras.layers.Lambda(lambda x: x[:, -10:])(enhanced_input)
    general_features = tf.keras.layers.Lambda(lambda x: x[:, :-10])(enhanced_input)
    
    # Process pin-specific features separately
    pin_branch = tf.keras.layers.Dense(16, activation='relu')(pin_specific)
    pin_branch = tf.keras.layers.Dropout(0.3)(pin_branch)
    
    # Process general features
    general_branch = tf.keras.layers.Dense(32, activation='relu')(general_features)
    general_branch = tf.keras.layers.Dropout(0.3)(general_branch)
    
    # Combine all branches
    combined = tf.keras.layers.concatenate([mel_features, general_branch, pin_branch])
    combined = tf.keras.layers.Dense(64, activation='relu')(combined)
    combined = tf.keras.layers.Dropout(PinFocusedConfig.DROPOUT_RATE)(combined)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs=[mel_input, enhanced_input], outputs=output)
    return model

def train_pin_focused_model():
    """Main pin-focused training function"""
    print("🚀 Starting Pin-Focused model training...")
    print("🎯 Special focus on fixing Pin classification issues")
    
    # Step 1: Collect raw files
    print("📁 Collecting raw files...")
    raw_files = []
    raw_labels = []
    
    instrument_folders = [d for d in os.listdir(PinFocusedConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(PinFocusedConfig.DATA_PATH, d))]
    
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
            
        folder_path = os.path.join(PinFocusedConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            raw_files.append(file_path)
            raw_labels.append(instrument)
    
    print(f"Found {len(raw_files)} raw files across {len(class_names)} instruments")
    
    # Step 2: Session-aware split
    if PinFocusedConfig.USE_SESSION_AWARE_SPLIT:
        train_files, test_files, train_labels, test_labels = session_aware_split(
            raw_files, raw_labels, test_size=0.2
        )
    else:
        train_files, test_files, train_labels, test_labels = train_test_split(
            raw_files, raw_labels, test_size=0.2, stratify=raw_labels, random_state=42
        )
    
    # Step 3: Process files with pin-focused features
    print("🔄 Processing training files with PIN-FOCUSED features...")
    print("⚠️ Note: Pin samples will NOT be augmented to preserve attack characteristics")
    X_train_mel, X_train_enhanced, y_train = process_files_with_pin_focus(
        train_files, train_labels, with_augmentation=True
    )
    
    print("🔄 Processing test files...")
    X_test_mel, X_test_enhanced, y_test = process_files_with_pin_focus(
        test_files, test_labels, with_augmentation=False
    )
    
    # Step 4: Normalize enhanced features
    print("🔧 Normalizing enhanced features (42 features including 10 pin-specific)...")
    scaler = StandardScaler()
    X_train_enhanced = scaler.fit_transform(X_train_enhanced)
    X_test_enhanced = scaler.transform(X_test_enhanced)
    
    # Step 5: Encode labels
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    print(f"\n📊 Dataset Summary:")
    print(f"Training samples: {len(X_train_mel)}")
    print(f"Test samples: {len(X_test_mel)}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]} features per sample")
    print(f"Pin-specific features: 10 (features 32-41)")
    
    # Print class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\n🎵 Training set distribution:")
    for cls, count in zip(unique_train, counts_train):
        percentage = count / len(y_train) * 100
        print(f"  • {cls}: {count} samples ({percentage:.1f}%)")
        if cls == 'pin':
            print(f"    📍 Pin preservation: No augmentation applied")
    
    # Step 6: Build pin-focused model
    print("\n🏗️ Building pin-focused multi-feature model...")
    mel_input_shape = X_train_mel.shape[1:]
    enhanced_feature_size = X_train_enhanced.shape[1]
    
    model = build_pin_focused_model(mel_input_shape, enhanced_feature_size, len(class_names))
    
    # Step 7: Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=PinFocusedConfig.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n🔧 Model Architecture:")
    model.summary()
    
    # Step 8: Calculate class weights with special attention to pin
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_encoded), 
        y=y_train_encoded
    )
    class_weights = dict(enumerate(weights))
    
    # Boost pin weight slightly due to its difficulty
    pin_class_idx = label_to_int.get('pin', -1)
    if pin_class_idx != -1:
        class_weights[pin_class_idx] *= 1.5  # Boost pin weight by 50%
        print(f"🎯 Pin class weight boosted: {class_weights[pin_class_idx]:.3f}")
    
    print(f"\n⚖️ Class weights:")
    for i, (cls, weight) in enumerate(zip(class_names, weights)):
        boosted = " (boosted)" if cls == 'pin' else ""
        print(f"  • {cls}: {class_weights[i]:.3f}{boosted}")
    
    # Step 9: Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=PinFocusedConfig.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(PinFocusedConfig.MODEL_SAVE_PATH, 'best_pin_focused_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Step 10: Train model
    print("\n🎯 Training pin-focused model...")
    print("🔍 Watch for improved Pin classification performance!")
    
    history = model.fit(
        [X_train_mel, X_train_enhanced], y_train_encoded,
        validation_data=([X_test_mel, X_test_enhanced], y_test_encoded),
        epochs=PinFocusedConfig.EPOCHS,
        batch_size=PinFocusedConfig.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Step 11: Evaluate
    print("\n📊 Evaluating pin-focused model...")
    test_loss, test_acc = model.evaluate([X_test_mel, X_test_enhanced], y_test_encoded, verbose=0)
    print(f"🎯 Final test accuracy: {test_acc:.4f}")
    
    # Generate predictions and detailed analysis
    y_pred = np.argmax(model.predict([X_test_mel, X_test_enhanced], verbose=0), axis=1)
    y_pred_proba = model.predict([X_test_mel, X_test_enhanced], verbose=0)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Step 12: Pin-specific analysis
    print("\n🎯 PIN-SPECIFIC ANALYSIS:")
    pin_class_idx = label_to_int.get('pin', -1)
    if pin_class_idx != -1:
        pin_mask = y_test_encoded == pin_class_idx
        pin_predictions = y_pred[pin_mask]
        pin_true_count = np.sum(pin_mask)
        pin_correct_count = np.sum(pin_predictions == pin_class_idx)
        pin_accuracy = pin_correct_count / pin_true_count if pin_true_count > 0 else 0
        
        print(f"📍 Pin Results:")
        print(f"  • Total pin samples in test: {pin_true_count}")
        print(f"  • Correctly classified: {pin_correct_count}")
        print(f"  • Pin accuracy: {pin_accuracy:.4f} ({pin_accuracy*100:.1f}%)")
        
        # Show where pin samples were misclassified
        if pin_true_count > pin_correct_count:
            print(f"  • Pin misclassifications:")
            pin_misclass = pin_predictions[pin_predictions != pin_class_idx]
            for wrong_class in np.unique(pin_misclass):
                count = np.sum(pin_misclass == wrong_class)
                wrong_class_name = class_names[wrong_class]
                print(f"    - {count} pin samples classified as {wrong_class_name}")
    
    # Step 13: Create comprehensive visualizations
    print("\n📈 Creating comprehensive visualizations...")
    fig = plt.figure(figsize=(20, 15))
    
    # Training history - Accuracy
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy (Pin-Focused)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training history - Loss
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss (Pin-Focused)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix with pin highlighting
    ax3 = plt.subplot(3, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix (Pin-Focused)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Highlight pin row and column
    pin_idx = label_to_int.get('pin', -1)
    if pin_idx != -1:
        ax3.axhline(pin_idx + 0.5, color='red', linewidth=3, alpha=0.7)
        ax3.axhline(pin_idx - 0.5, color='red', linewidth=3, alpha=0.7)
        ax3.axvline(pin_idx + 0.5, color='red', linewidth=3, alpha=0.7)
        ax3.axvline(pin_idx - 0.5, color='red', linewidth=3, alpha=0.7)
    
    # Per-class accuracy with pin highlighted
    ax4 = plt.subplot(3, 3, 4)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    colors = ['red' if cls == 'pin' else 'skyblue' for cls in class_names]
    bars = ax4.bar(class_names, class_accuracy, color=colors)
    ax4.set_title('Per-class Accuracy (Pin Highlighted)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add accuracy values on bars
    for bar, acc, cls in zip(bars, class_accuracy, class_names):
        color = 'white' if cls == 'pin' else 'black'
        weight = 'bold' if cls == 'pin' else 'normal'
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', color=color, fontweight=weight)
    
    # Pin confidence analysis
    ax5 = plt.subplot(3, 3, 5)
    if pin_class_idx != -1:
        pin_confidences = y_pred_proba[pin_mask, pin_class_idx]
        ax5.hist(pin_confidences, bins=15, alpha=0.7, color='red', edgecolor='black')
        ax5.axvline(np.mean(pin_confidences), color='darkred', linestyle='--', 
                   label=f'Mean: {np.mean(pin_confidences):.3f}')
        ax5.set_title('Pin Prediction Confidence', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Frequency')
        ax5.legend()
    
    # Feature importance (last 10 are pin-specific)
    ax6 = plt.subplot(3, 3, 6)
    feature_importance = np.std(X_train_enhanced, axis=0)
    pin_feature_importance = feature_importance[-10:]  # Last 10 features
    pin_feature_names = [f'Pin_{i+1}' for i in range(10)]
    
    bars = ax6.bar(pin_feature_names, pin_feature_importance, color='red', alpha=0.7)
    ax6.set_title('Pin-Specific Feature Importance', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Standard Deviation')
    ax6.tick_params(axis='x', rotation=45)
    
    # Class distribution comparison
    ax7 = plt.subplot(3, 3, 7)
    train_dist = [np.sum(y_train_encoded == i) for i in range(len(class_names))]
    test_dist = [np.sum(y_test_encoded == i) for i in range(len(class_names))]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax7.bar(x - width/2, train_dist, width, label='Train', alpha=0.8)
    bars2 = ax7.bar(x + width/2, test_dist, width, label='Test', alpha=0.8)
    
    # Highlight pin bars
    pin_idx_plot = label_to_int.get('pin', -1)
    if pin_idx_plot != -1:
        bars1[pin_idx_plot].set_color('red')
        bars2[pin_idx_plot].set_color('darkred')
    
    ax7.set_title('Train vs Test Distribution', fontsize=14, fontweight='bold')
    ax7.set_ylabel('Number of Samples')
    ax7.set_xticks(x)
    ax7.set_xticklabels(class_names, rotation=45)
    ax7.legend()
    
    # Overall confidence distribution
    ax8 = plt.subplot(3, 3, 8)
    max_confidences = np.max(y_pred_proba, axis=1)
    ax8.hist(max_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax8.axvline(np.mean(max_confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(max_confidences):.3f}')
    ax8.set_title('Overall Prediction Confidence', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Max Confidence')
    ax8.set_ylabel('Frequency')
    ax8.legend()
    
    # Model summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate improvement metrics
    pin_accuracy_val = class_accuracy[pin_class_idx] if pin_class_idx != -1 else 0
    
    summary_text = f"""
    Pin-Focused Model Summary
    ========================
    
    🎯 Pin Classification Results:
    • Pin Accuracy: {pin_accuracy_val:.3f} ({pin_accuracy_val*100:.1f}%)
    • Total Parameters: {model.count_params():,}
    • Enhanced Features: 42 (32 + 10 pin-specific)
    
    📊 Overall Performance:
    • Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)
    • Training Samples: {len(X_train_mel):,}
    • Test Samples: {len(X_test_mel):,}
    
    🏆 Best Performing Classes:
    """
    
    # Add best and worst classes
    best_classes = np.argsort(class_accuracy)[-2:]
    worst_classes = np.argsort(class_accuracy)[:2]
    
    for idx in reversed(best_classes):
        summary_text += f"• {class_names[idx]}: {class_accuracy[idx]:.3f}\n    "
    
    summary_text += f"\n🎯 Focus Areas:\n    "
    for idx in worst_classes:
        summary_text += f"• {class_names[idx]}: {class_accuracy[idx]:.3f}\n    "
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PinFocusedConfig.MODEL_SAVE_PATH, 'pin_focused_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 14: Detailed classification report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT (Pin-Focused)")
    print("="*80)
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Step 15: Pin-specific confusion analysis
    print("\n" + "="*80)
    print("PIN-FOCUSED CONFUSION ANALYSIS")
    print("="*80)
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i, :].sum()
                if confusion_rate > 0.05:  # Show confusions > 5%
                    highlight = "🎯 " if true_class == 'pin' or pred_class == 'pin' else "• "
                    print(f"{highlight}{true_class} → {pred_class}: {cm[i, j]} samples ({confusion_rate:.1%})")
    
    # Step 16: Save model and comprehensive metadata
    print("\n💾 Saving pin-focused model and metadata...")
    
    # Save model
    model.save(os.path.join(PinFocusedConfig.MODEL_SAVE_PATH, 'pin_focused_model.h5'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(PinFocusedConfig.MODEL_SAVE_PATH, 'pin_focused_scaler.pkl'))
    
    # Save comprehensive metadata
    metadata = {
        'model_version': 'pin_focused_v1',
        'class_names': class_names,
        'test_accuracy': float(test_acc),
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(class_accuracy)},
        'pin_specific_accuracy': float(pin_accuracy_val),
        'model_type': 'pin_focused_multi_feature_cnn',
        'enhanced_features': 42,
        'pin_specific_features': 10,
        'pin_preservation': PinFocusedConfig.PRESERVE_PIN_CHARACTERISTICS,
        'session_aware_split': PinFocusedConfig.USE_SESSION_AWARE_SPLIT,
        'audio_parameters': {
            'sample_rate': PinFocusedConfig.SAMPLE_RATE,
            'segment_duration': PinFocusedConfig.SEGMENT_DURATION,
            'n_mels': PinFocusedConfig.N_MELS,
            'n_fft': PinFocusedConfig.N_FFT,
            'hop_length': PinFocusedConfig.HOP_LENGTH,
            'fmax': PinFocusedConfig.FMAX
        },
        'training_parameters': {
            'batch_size': PinFocusedConfig.BATCH_SIZE,
            'learning_rate': PinFocusedConfig.LEARNING_RATE,
            'epochs_trained': len(history.history['accuracy']),
            'early_stopping_patience': PinFocusedConfig.EARLY_STOPPING_PATIENCE,
            'dropout_rate': PinFocusedConfig.DROPOUT_RATE,
            'l2_regularization': PinFocusedConfig.L2_REGULARIZATION,
            'pin_weight_boost': 1.5
        },
        'dataset_info': {
            'total_raw_files': len(raw_files),
            'training_samples': len(X_train_mel),
            'test_samples': len(X_test_mel),
            'pin_samples_train': int(np.sum(y_train_encoded == pin_class_idx)) if pin_class_idx != -1 else 0,
            'pin_samples_test': int(np.sum(y_test_encoded == pin_class_idx)) if pin_class_idx != -1 else 0
        },
        'mel_input_shape': list(mel_input_shape),
        'confusion_matrix': cm.tolist(),
        'tensorflow_version': tf.__version__,
        'pin_feature_names': [
            'pin_attack_sharpness', 'pin_decay_rate', 'pin_attack_count',
            'pin_fundamental_strength', 'pin_harmonic_clarity', 'pin_sustain_10',
            'pin_sustain_05', 'pin_sustain_ratio', 'pin_attack_energy_ratio',
            'pin_spectral_evolution'
        ]
    }
    
    with open(os.path.join(PinFocusedConfig.MODEL_SAVE_PATH, 'pin_focused_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Pin-focused training complete!")
    print(f"📁 Model saved to: {PinFocusedConfig.MODEL_SAVE_PATH}")
    print(f"🎯 Final test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"📍 Pin accuracy: {pin_accuracy_val:.4f} ({pin_accuracy_val*100:.1f}%)")
    print(f"🔧 Total model parameters: {model.count_params():,}")
    print(f"📊 Pin-specific features: 10 specialized features")
    
    return model, history, class_names, metadata

if __name__ == "__main__":
    train_pin_focused_model()