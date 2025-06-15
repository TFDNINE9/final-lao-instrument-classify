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

# ENHANCED CONFIGURATION - Version 2
class EnhancedConfig:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0  # Back to 6.0 seconds
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # CONSERVATIVE AUGMENTATION
    USE_AUGMENTATION = True
    TIME_STRETCH_RANGE = (0.95, 1.05)
    PITCH_SHIFT_RANGE = (-0.5, 0.5)
    NOISE_FACTOR = 0.003
    
    # OPTIMIZED TRAINING PARAMETERS (based on your results)
    BATCH_SIZE = 4               # Smaller batches for better generalization
    EPOCHS = 25                  # Reduced since you were getting good results at epoch 15
    LEARNING_RATE = 0.00005      # Even lower learning rate
    EARLY_STOPPING_PATIENCE = 5  # Stop earlier to prevent overfitting
    
    # INCREASED REGULARIZATION
    DROPOUT_RATE = 0.7           # Higher dropout
    L2_REGULARIZATION = 0.01     # Higher L2 regularization
    
    # Enhanced features
    USE_ENHANCED_FEATURES = True
    USE_SESSION_AWARE_SPLIT = True
    
    # Paths
    DATA_PATH = "dataset"
    MODEL_SAVE_PATH = "models/enhanced_v2_model_6sec"  # Updated path name
    
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
os.makedirs(EnhancedConfig.MODEL_SAVE_PATH, exist_ok=True)

def map_instrument_folder(folder_name, class_names):
    """Map folder name to standard instrument name"""
    folder_lower = folder_name.lower()
    
    for standard_name, variants in EnhancedConfig.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    if 'unknown' in folder_lower or 'noise' in folder_lower or 'background' in folder_lower:
        return None
    
    for cls in class_names:
        if cls.lower() in folder_lower:
            return cls
    
    return folder_lower


def extract_essential_features_only(audio, sr):
    """Extract only the most important features for speed"""
    features = {}
    
    try:
        # Only compute the most essential features
        rms_energy = librosa.feature.rms(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Basic features (8 features)
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['zcr_mean'] = np.mean(zero_crossings)
        
        # Pin-specific (4 features)
        peak_idx = np.argmax(rms_energy)
        if peak_idx > 0 and peak_idx < len(rms_energy) - 1:
            attack_portion = rms_energy[:peak_idx+1]
            decay_portion = rms_energy[peak_idx:]
            features['attack_steepness'] = np.max(np.diff(attack_portion)) if len(attack_portion) > 1 else 0
            features['decay_rate'] = decay_portion[-1] / (decay_portion[0] + 1e-8) if len(decay_portion) > 0 else 0
        else:
            features['attack_steepness'] = 0
            features['decay_rate'] = 0
        
        # Khong_vong specific (4 features)
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        high_freq_mask = freqs > 4000
        features['metallic_shimmer'] = np.mean(stft[high_freq_mask, :]) / (np.mean(stft) + 1e-8)
        features['frequency_variation'] = np.std(spectral_centroids)
        
        # Fill remaining slots with zeros or duplicates
        for i in range(16):  # Total 16 features instead of 32
            if f'feature_{i}' not in features:
                features[f'feature_{i}'] = list(features.values())[i % len(features)]
        
    except:
        # If anything fails, return zeros
        features = {f'feature_{i}': 0.0 for i in range(16)}
    
    return features


def extract_pin_specific_features(audio, sr):
    """Extract features that specifically distinguish pin from ranad and khean"""
    features = {}
    
    # PIN-SPECIFIC: Plucking attack characteristics
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
    
    if len(onset_frames) > 0:
        attack_times = []
        decay_rates = []
        
        for onset_frame in onset_frames[:5]:
            onset_sample = librosa.frames_to_samples(onset_frame)
            
            attack_window = int(0.05 * sr)  # 50ms
            decay_window = int(0.5 * sr)    # 500ms
            
            if onset_sample + decay_window < len(audio):
                attack_segment = audio[onset_sample:onset_sample + attack_window]
                decay_segment = audio[onset_sample + attack_window:onset_sample + decay_window]
                
                # Attack steepness
                if len(attack_segment) > 1:
                    attack_steepness = np.max(np.diff(np.abs(attack_segment)))
                    attack_times.append(attack_steepness)
                
                # Decay rate
                if len(decay_segment) > 10:
                    decay_envelope = np.abs(decay_segment)
                    if decay_envelope[0] > 0:
                        decay_rate = decay_envelope[-1] / decay_envelope[0]
                        decay_rates.append(decay_rate)
        
        features['attack_steepness_mean'] = np.mean(attack_times) if attack_times else 0
        features['attack_steepness_std'] = np.std(attack_times) if len(attack_times) > 1 else 0
        features['decay_rate_mean'] = np.mean(decay_rates) if decay_rates else 0
        features['attack_to_decay_ratio'] = (np.mean(attack_times) / (np.mean(decay_rates) + 1e-6)) if attack_times and decay_rates else 0
    else:
        features['attack_steepness_mean'] = 0
        features['attack_steepness_std'] = 0
        features['decay_rate_mean'] = 0
        features['attack_to_decay_ratio'] = 0
    
    # Sustain characteristics
    rms_energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    
    energy_threshold = np.max(rms_energy) * 0.1
    sustain_frames = np.sum(rms_energy > energy_threshold)
    features['sustain_duration'] = sustain_frames / len(rms_energy)
    
    # Energy envelope shape
    if len(rms_energy) > 10:
        time_points = np.arange(len(rms_energy))
        poly_coeffs = np.polyfit(time_points, rms_energy, deg=2)
        features['envelope_curvature'] = poly_coeffs[0]
        features['envelope_slope'] = poly_coeffs[1]
    else:
        features['envelope_curvature'] = 0
        features['envelope_slope'] = 0
    
    # Harmonic structure
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(audio, margin=3.0)
        stft = np.abs(librosa.stft(y_harmonic))
        freqs = librosa.fft_frequencies(sr=sr)
        
        fundamental_mask = (freqs >= 80) & (freqs <= 800)
        harmonic_mask = (freqs >= 800) & (freqs <= 3200)
        
        fundamental_energy = np.mean(stft[fundamental_mask, :])
        harmonic_energy = np.mean(stft[harmonic_mask, :])
        
        features['fundamental_strength'] = fundamental_energy
        features['harmonic_complexity'] = harmonic_energy / (fundamental_energy + 1e-8)
        
        # Spectral regularity
        spectral_peaks = []
        for frame in range(min(10, stft.shape[1])):
            frame_spectrum = stft[:, frame]
            peaks = np.where(frame_spectrum > np.percentile(frame_spectrum, 90))[0]
            if len(peaks) > 0:
                spectral_peaks.extend(freqs[peaks])
        
        if spectral_peaks:
            peak_diffs = np.diff(sorted(spectral_peaks)[:10])
            features['spectral_regularity'] = 1.0 / (np.std(peak_diffs) + 1e-6) if len(peak_diffs) > 1 else 0
        else:
            features['spectral_regularity'] = 0
            
    except:
        features['fundamental_strength'] = 0
        features['harmonic_complexity'] = 0
        features['spectral_regularity'] = 0
    
    return features

def extract_khong_vong_features(audio, sr):
    """Extract features specific to khong_vong (gong-like characteristics)"""
    features = {}
    
    # Frequency modulation and beating
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, fmin=50, fmax=2000)
        f0_clean = f0[voiced_flag]
        
        if len(f0_clean) > 20:
            f0_variation = np.std(f0_clean)
            f0_diff = np.diff(f0_clean)
            beating_measure = np.sum(np.abs(f0_diff) > np.std(f0_diff)) / len(f0_diff)
            
            features['frequency_variation'] = f0_variation
            features['beating_intensity'] = beating_measure
        else:
            features['frequency_variation'] = 0
            features['beating_intensity'] = 0
    except:
        features['frequency_variation'] = 0
        features['beating_intensity'] = 0
    
    # Metallic timbre
    stft = np.abs(librosa.stft(audio))
    freqs = librosa.fft_frequencies(sr=sr)
    
    high_freq_mask = freqs > 4000
    mid_freq_mask = (freqs >= 1000) & (freqs <= 4000)
    low_freq_mask = freqs < 1000
    
    high_freq_energy = np.mean(stft[high_freq_mask, :])
    mid_freq_energy = np.mean(stft[mid_freq_mask, :])
    low_freq_energy = np.mean(stft[low_freq_mask, :])
    
    total_energy = high_freq_energy + mid_freq_energy + low_freq_energy + 1e-8
    
    features['metallic_shimmer'] = high_freq_energy / total_energy
    features['mid_frequency_ratio'] = mid_freq_energy / total_energy
    
    # Resonance characteristics
    rms_energy = librosa.feature.rms(y=audio)[0]
    peak_energy = np.max(rms_energy)
    resonance_threshold = peak_energy * 0.05
    
    resonance_samples = np.sum(rms_energy > resonance_threshold)
    features['resonance_duration'] = resonance_samples / len(rms_energy)
    
    # Decay characteristics
    if len(rms_energy) > 20:
        peak_idx = np.argmax(rms_energy)
        if peak_idx < len(rms_energy) - 10:
            decay_portion = rms_energy[peak_idx:]
            if len(decay_portion) > 5:
                log_decay = np.log(decay_portion + 1e-8)
                decay_slope = np.polyfit(range(len(log_decay)), log_decay, 1)[0]
                features['decay_characteristic'] = abs(decay_slope)
            else:
                features['decay_characteristic'] = 0
        else:
            features['decay_characteristic'] = 0
    else:
        features['decay_characteristic'] = 0
    
    return features

def extract_base_features(audio, sr):
    """Extract base features from previous version"""
    features = {}
    
    # Attack and onset characteristics
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='frames')
    if len(onset_frames) > 0:
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        attack_characteristics = []
        
        for onset_time in onset_times[:3]:
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
    
    # Pitch characteristics
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
    
    # Harmonic/percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    total_energy = harmonic_energy + percussive_energy + 1e-8
    
    features['harmonic_ratio'] = harmonic_energy / total_energy
    features['percussive_ratio'] = percussive_energy / total_energy
    
    # Spectral characteristics
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # Energy and texture
    rms_energy = librosa.feature.rms(y=audio)[0]
    features['energy_mean'] = np.mean(rms_energy)
    features['energy_std'] = np.std(rms_energy)
    features['energy_continuity'] = 1.0 / (np.std(rms_energy) + 1e-6)
    
    zero_crossings = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zero_crossings)
    features['texture_consistency'] = 1.0 / (np.std(zero_crossings) + 1e-6)
    
    # Transients
    audio_diff = np.diff(audio)
    sharp_transients = np.sum(np.abs(audio_diff) > np.std(audio_diff) * 2)
    features['transient_density'] = sharp_transients / len(audio)
    
    return features

def extract_all_enhanced_features(audio, sr):
    """Combine all feature extraction methods"""
    # Get base features
    base_features = extract_base_features(audio, sr)
    
    # Add pin-specific features
    pin_features = extract_pin_specific_features(audio, sr)
    
    # Add khong_vong features
    khong_vong_features = extract_khong_vong_features(audio, sr)
    
    # Combine all features
    all_features = {**base_features, **pin_features, **khong_vong_features}
    
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
    
    if EnhancedConfig.USE_AUGMENTATION:
        if instrument_type == 'saw':
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.3)
            augmented_samples.append(pitch_shifted)
            
        elif instrument_type == 'pin':
            pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=0.5)
            augmented_samples.append(pitch_shifted)
            
        elif instrument_type == 'khean':
            breath_noise = np.random.normal(0, EnhancedConfig.NOISE_FACTOR, len(audio))
            noisy = audio + breath_noise
            augmented_samples.append(noisy)
            
        elif instrument_type == 'khong_vong':
            # For metallic instruments, preserve resonance
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

def process_files_with_enhanced_features(file_paths, labels, with_augmentation=True):
    """Process files with all enhanced features"""
    mel_features = []
    enhanced_features = []
    processed_labels = []
    
    for file_path, label in tqdm(zip(file_paths, labels), 
                                desc=f"Processing with enhanced features",
                                total=len(file_paths)):
        try:
            audio, sr = librosa.load(file_path, sr=EnhancedConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.5:
                continue
            
            if with_augmentation:
                augmented_samples = targeted_augmentation(audio, sr, label)
            else:
                augmented_samples = [audio]
            
            for aug_audio in augmented_samples:
                best_segment = process_audio_with_best_segment(aug_audio, sr, 
                                                             EnhancedConfig.SEGMENT_DURATION)
                
                # Extract mel-spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=best_segment, sr=sr,
                    n_fft=EnhancedConfig.N_FFT,
                    hop_length=EnhancedConfig.HOP_LENGTH,
                    n_mels=EnhancedConfig.N_MELS,
                    fmax=EnhancedConfig.FMAX
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
                
                mel_features.append(mel_spec_with_channel)
                
                # Extract all enhanced features
                enhanced_feat = extract_all_enhanced_features(best_segment, sr)
                
                # Convert to array (32 features total)
                feat_array = np.array([
                    # Base features (16)
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
                    
                    # Pin-specific features (10)
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
                    
                    # Khong_vong features (6)
                    enhanced_feat.get('frequency_variation', 0),
                    enhanced_feat.get('beating_intensity', 0),
                    enhanced_feat.get('metallic_shimmer', 0),
                    enhanced_feat.get('mid_frequency_ratio', 0),
                    enhanced_feat.get('resonance_duration', 0),
                    enhanced_feat.get('decay_characteristic', 0)
                ])
                
                enhanced_features.append(feat_array)
                processed_labels.append(label)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(mel_features), np.array(enhanced_features), np.array(processed_labels)



def extract_fast_enhanced_features(audio, sr):
    """Fast enhanced features - optimized for speed"""
    features = {}
    
    # PRE-COMPUTE EXPENSIVE OPERATIONS ONCE
    try:
        # Compute STFT once and reuse
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        stft_mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Compute other expensive features once
        rms_energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=512)[0] 
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=512)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio, hop_length=512)[0]
        
    except Exception as e:
        # If any computation fails, return zeros
        return {f'feature_{i}': 0.0 for i in range(32)}
    
    # BASIC FEATURES (fast)
    features['energy_mean'] = np.mean(rms_energy)
    features['energy_std'] = np.std(rms_energy)
    features['energy_continuity'] = 1.0 / (np.std(rms_energy) + 1e-6)
    
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    features['zcr_mean'] = np.mean(zero_crossings)
    features['texture_consistency'] = 1.0 / (np.std(zero_crossings) + 1e-6)
    
    # SIMPLIFIED ONSET DETECTION (much faster)
    try:
        # Use simple energy-based onset detection instead of complex onset_detect
        energy_diff = np.diff(rms_energy)
        energy_peaks = np.where(energy_diff > np.std(energy_diff) * 2)[0]
        features['onset_density'] = len(energy_peaks) / (len(audio) / sr)
        
        # Simple attack measure
        if len(energy_peaks) > 0:
            attack_slopes = []
            for peak_idx in energy_peaks[:3]:  # First 3 peaks only
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
    
    # SIMPLIFIED HARMONIC/PERCUSSIVE (faster version)
    try:
        # Use spectral analysis instead of expensive HPSS
        # Low frequencies (< 500Hz) vs high frequencies (> 2000Hz)
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
    
    # FAST PIN-SPECIFIC FEATURES
    try:
        # Simplified attack analysis using energy envelope
        if len(rms_energy) > 10:
            # Energy envelope shape analysis
            time_points = np.arange(len(rms_energy))
            poly_coeffs = np.polyfit(time_points, rms_energy, deg=2)
            features['envelope_curvature'] = poly_coeffs[0]
            features['envelope_slope'] = poly_coeffs[1]
            
            # Sustain vs attack ratio
            peak_idx = np.argmax(rms_energy)
            if peak_idx > 0 and peak_idx < len(rms_energy) - 1:
                attack_portion = rms_energy[:peak_idx+1]
                decay_portion = rms_energy[peak_idx:]
                
                if len(attack_portion) > 0 and len(decay_portion) > 1:
                    attack_steepness = np.max(np.diff(attack_portion)) if len(attack_portion) > 1 else 0
                    decay_rate = decay_portion[-1] / (decay_portion[0] + 1e-8)
                    
                    features['attack_steepness_mean'] = attack_steepness
                    features['decay_rate_mean'] = decay_rate
                    features['attack_to_decay_ratio'] = attack_steepness / (decay_rate + 1e-6)
                else:
                    features['attack_steepness_mean'] = 0
                    features['decay_rate_mean'] = 0
                    features['attack_to_decay_ratio'] = 0
            else:
                features['attack_steepness_mean'] = 0
                features['decay_rate_mean'] = 0
                features['attack_to_decay_ratio'] = 0
        else:
            features['envelope_curvature'] = 0
            features['envelope_slope'] = 0
            features['attack_steepness_mean'] = 0
            features['decay_rate_mean'] = 0
            features['attack_to_decay_ratio'] = 0
        
        # Sustain duration (simplified)
        energy_threshold = np.max(rms_energy) * 0.1
        sustain_frames = np.sum(rms_energy > energy_threshold)
        features['sustain_duration'] = sustain_frames / len(rms_energy)
        
    except:
        features['envelope_curvature'] = 0
        features['envelope_slope'] = 0
        features['attack_steepness_mean'] = 0
        features['decay_rate_mean'] = 0
        features['attack_to_decay_ratio'] = 0
        features['sustain_duration'] = 0
    
    # FAST SPECTRAL FEATURES
    try:
        # Fundamental vs harmonics (simplified)
        fundamental_mask = (freqs >= 80) & (freqs <= 800)
        harmonic_mask = (freqs > 800) & (freqs <= 3200)
        
        fundamental_energy = np.mean(stft_mag[fundamental_mask, :])
        harmonic_energy = np.mean(stft_mag[harmonic_mask, :])
        
        features['fundamental_strength'] = fundamental_energy
        features['harmonic_complexity'] = harmonic_energy / (fundamental_energy + 1e-8)
        
        # Spectral regularity (simplified)
        # Use spectral centroid variation as proxy
        features['spectral_regularity'] = 1.0 / (np.std(spectral_centroids) + 1e-6)
        
    except:
        features['fundamental_strength'] = 0
        features['harmonic_complexity'] = 0
        features['spectral_regularity'] = 0
    
    # FAST KHONG_VONG FEATURES
    try:
        # High frequency content (metallic shimmer)
        very_high_freq_mask = freqs > 4000
        metallic_energy = np.mean(stft_mag[very_high_freq_mask, :])
        total_spectral_energy = np.mean(stft_mag) + 1e-8
        
        features['metallic_shimmer'] = metallic_energy / total_spectral_energy
        features['mid_frequency_ratio'] = mid_energy / total_energy
        
        # Resonance duration (using energy)
        peak_energy = np.max(rms_energy)
        resonance_threshold = peak_energy * 0.05
        resonance_samples = np.sum(rms_energy > resonance_threshold)
        features['resonance_duration'] = resonance_samples / len(rms_energy)
        
        # Simple frequency variation
        features['frequency_variation'] = np.std(spectral_centroids)
        
        # Simple beating detection using energy modulation
        if len(rms_energy) > 10:
            energy_smooth = np.convolve(rms_energy, np.ones(5)/5, mode='same')
            energy_modulation = np.std(rms_energy - energy_smooth)
            features['beating_intensity'] = energy_modulation
        else:
            features['beating_intensity'] = 0
        
        # Decay characteristic (simplified)
        if len(rms_energy) > 20:
            peak_idx = np.argmax(rms_energy)
            if peak_idx < len(rms_energy) - 10:
                decay_portion = rms_energy[peak_idx:]
                # Simple exponential fit approximation
                if len(decay_portion) > 5 and decay_portion[0] > 0:
                    decay_ratio = decay_portion[-1] / decay_portion[0]
                    features['decay_characteristic'] = abs(np.log(decay_ratio + 1e-8))
                else:
                    features['decay_characteristic'] = 0
            else:
                features['decay_characteristic'] = 0
        else:
            features['decay_characteristic'] = 0
            
    except:
        features['metallic_shimmer'] = 0
        features['mid_frequency_ratio'] = 0
        features['resonance_duration'] = 0
        features['frequency_variation'] = 0
        features['beating_intensity'] = 0
        features['decay_characteristic'] = 0
    
    # REMAINING FEATURES (placeholders for consistency)
    features['attack_steepness_std'] = features['attack_steepness_mean'] * 0.1  # Approximation
    features['pitch_variation'] = features['frequency_variation']  # Reuse
    features['vibrato_rate'] = features['beating_intensity']  # Reuse
    features['transient_density'] = features['onset_density']  # Reuse
    
    return features

def process_files_with_fast_features(file_paths, labels, with_augmentation=True):
    """Optimized processing with fast feature extraction"""
    mel_features = []
    enhanced_features = []
    processed_labels = []
    
    # Pre-allocate arrays for better performance
    estimated_samples = len(file_paths) * (2 if with_augmentation else 1)
    
    print(f"Processing {len(file_paths)} files with fast enhanced features...")
    
    for file_path, label in tqdm(zip(file_paths, labels), 
                                desc=f"Fast processing ({'with' if with_augmentation else 'without'} augmentation)",
                                total=len(file_paths)):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=EnhancedConfig.SAMPLE_RATE)
            
            if len(audio) < sr * 0.5:
                continue
            
            if with_augmentation:
                augmented_samples = targeted_augmentation(audio, sr, label)
            else:
                augmented_samples = [audio]
            
            for aug_audio in augmented_samples:
                best_segment = process_audio_with_best_segment(aug_audio, sr, 
                                                             EnhancedConfig.SEGMENT_DURATION)
                
                # Extract mel-spectrogram (this is already optimized)
                mel_spec = librosa.feature.melspectrogram(
                    y=best_segment, sr=sr,
                    n_fft=EnhancedConfig.N_FFT,
                    hop_length=EnhancedConfig.HOP_LENGTH,
                    n_mels=EnhancedConfig.N_MELS,
                    fmax=EnhancedConfig.FMAX
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
                mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
                
                mel_features.append(mel_spec_with_channel)
                
                # Extract FAST enhanced features
                enhanced_feat = extract_fast_enhanced_features(best_segment, sr)
                
                # Convert to array (32 features - same as before)
                feat_array = np.array([
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
                    
                    # Pin-specific features
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
                    
                    # Khong_vong features
                    enhanced_feat.get('frequency_variation', 0),
                    enhanced_feat.get('beating_intensity', 0),
                    enhanced_feat.get('metallic_shimmer', 0),
                    enhanced_feat.get('mid_frequency_ratio', 0),
                    enhanced_feat.get('resonance_duration', 0),
                    enhanced_feat.get('decay_characteristic', 0)
                ])
                
                enhanced_features.append(feat_array)
                processed_labels.append(label)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return np.array(mel_features), np.array(enhanced_features), np.array(processed_labels)

def build_enhanced_multi_feature_model(mel_input_shape, enhanced_feature_size, num_classes):
    """Build enhanced multi-feature model"""
    # Mel-spectrogram input (CNN branch)
    mel_input = tf.keras.layers.Input(shape=mel_input_shape, name='mel_input')
    
    # Simplified CNN
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
    x = tf.keras.layers.Dropout(EnhancedConfig.DROPOUT_RATE)(x)
    
    mel_features = tf.keras.layers.Dense(64, activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(EnhancedConfig.L2_REGULARIZATION))(x)
    
    # Enhanced features input (Dense branch)
    enhanced_input = tf.keras.layers.Input(shape=(enhanced_feature_size,), name='enhanced_input')
    enhanced_branch = tf.keras.layers.Dense(64, activation='relu',
                                           kernel_regularizer=tf.keras.regularizers.l2(EnhancedConfig.L2_REGULARIZATION))(enhanced_input)
    enhanced_branch = tf.keras.layers.Dropout(0.4)(enhanced_branch)
    enhanced_branch = tf.keras.layers.Dense(32, activation='relu')(enhanced_branch)
    enhanced_branch = tf.keras.layers.Dropout(0.3)(enhanced_branch)
    
    # Combine branches
    combined = tf.keras.layers.concatenate([mel_features, enhanced_branch])
    combined = tf.keras.layers.Dense(64, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(EnhancedConfig.L2_REGULARIZATION))(combined)
    combined = tf.keras.layers.Dropout(EnhancedConfig.DROPOUT_RATE)(combined)
    
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(combined)
    
    model = tf.keras.Model(inputs=[mel_input, enhanced_input], outputs=output)
    return model

def train_enhanced_model():
    """Main enhanced training function"""
    print("🚀 Starting Enhanced V2 model training...")
    
    # Step 1: Collect raw files
    print("📁 Collecting raw files...")
    raw_files = []
    raw_labels = []
    
    instrument_folders = [d for d in os.listdir(EnhancedConfig.DATA_PATH) 
                         if os.path.isdir(os.path.join(EnhancedConfig.DATA_PATH, d))]
    
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
            
        folder_path = os.path.join(EnhancedConfig.DATA_PATH, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            raw_files.append(file_path)
            raw_labels.append(instrument)
    
    print(f"Found {len(raw_files)} raw files across {len(class_names)} instruments")
    
    # Step 2: Session-aware split
    if EnhancedConfig.USE_SESSION_AWARE_SPLIT:
        train_files, test_files, train_labels, test_labels = session_aware_split(
            raw_files, raw_labels, test_size=0.2
        )
    else:
        train_files, test_files, train_labels, test_labels = train_test_split(
            raw_files, raw_labels, test_size=0.2, stratify=raw_labels, random_state=42
        )
    
    # Step 3: Process files with FAST enhanced features
    print("🔄 Processing training files with FAST enhanced features...")
    X_train_mel, X_train_enhanced, y_train = process_files_with_fast_features(
        train_files, train_labels, with_augmentation=True
    )
    
    print("🔄 Processing test files...")
    X_test_mel, X_test_enhanced, y_test = process_files_with_fast_features(
        test_files, test_labels, with_augmentation=False
    )
    
    # Step 4: Normalize enhanced features
    if EnhancedConfig.USE_ENHANCED_FEATURES:
        print("🔧 Normalizing enhanced features...")
        scaler = StandardScaler()
        X_train_enhanced = scaler.fit_transform(X_train_enhanced)
        X_test_enhanced = scaler.transform(X_test_enhanced)
    
    # Step 5: Encode labels
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_train_encoded = np.array([label_to_int[label] for label in y_train])
    y_test_encoded = np.array([label_to_int[label] for label in y_test])
    
    print(f"Training samples: {len(X_train_mel)}")
    print(f"Test samples: {len(X_test_mel)}")
    print(f"Enhanced features: {X_train_enhanced.shape[1]} features per sample")
    
    # Print class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\nTraining distribution:")
    for cls, count in zip(unique_train, counts_train):
        print(f"  {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Step 6: Build enhanced model
    print("🏗️ Building enhanced multi-feature model...")
    mel_input_shape = X_train_mel.shape[1:]
    enhanced_feature_size = X_train_enhanced.shape[1]
    
    model = build_enhanced_multi_feature_model(mel_input_shape, enhanced_feature_size, len(class_names))
    
    # Step 7: Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=EnhancedConfig.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Step 8: Class weights
    weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train_encoded), 
        y=y_train_encoded
    )
    class_weights = dict(enumerate(weights))
    print(f"\nClass weights: {class_weights}")
    
    # Step 9: Callbacks
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
            patience=3,  # Reduced patience
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Step 10: Train model
    print("🎯 Training enhanced model...")
    history = model.fit(
        [X_train_mel, X_train_enhanced], y_train_encoded,
        validation_data=([X_test_mel, X_test_enhanced], y_test_encoded),
        epochs=EnhancedConfig.EPOCHS,
        batch_size=EnhancedConfig.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Step 11: Evaluate
    print("📊 Evaluating enhanced model...")
    test_loss, test_acc = model.evaluate([X_test_mel, X_test_enhanced], y_test_encoded, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Generate predictions and confusion matrix
    y_pred = np.argmax(model.predict([X_test_mel, X_test_enhanced], verbose=0), axis=1)
    cm = confusion_matrix(y_test_encoded, y_pred)
    
    # Step 12: Create comprehensive visualizations
    print("📈 Creating visualizations...")
    fig = plt.figure(figsize=(20, 15))
    
    # Training history - Accuracy
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training history - Loss
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix
    ax3 = plt.subplot(3, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names, ax=ax3)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    
    # Per-class accuracy
    ax4 = plt.subplot(3, 3, 4)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    bars = ax4.bar(class_names, class_accuracy, color=plt.cm.Set3(np.arange(len(class_names))))
    ax4.set_title('Per-class Accuracy', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, class_accuracy):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Class distribution in training set
    ax5 = plt.subplot(3, 3, 5)
    train_dist = [np.sum(y_train_encoded == i) for i in range(len(class_names))]
    ax5.bar(class_names, train_dist, color=plt.cm.Set2(np.arange(len(class_names))))
    ax5.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Number of Samples')
    ax5.tick_params(axis='x', rotation=45)
    
    # Class distribution in test set
    ax6 = plt.subplot(3, 3, 6)
    test_dist = [np.sum(y_test_encoded == i) for i in range(len(class_names))]
    ax6.bar(class_names, test_dist, color=plt.cm.Set1(np.arange(len(class_names))))
    ax6.set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Number of Samples')
    ax6.tick_params(axis='x', rotation=45)
    
    # Prediction confidence distribution
    ax7 = plt.subplot(3, 3, 7)
    y_pred_proba = model.predict([X_test_mel, X_test_enhanced], verbose=0)
    max_confidences = np.max(y_pred_proba, axis=1)
    ax7.hist(max_confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax7.axvline(np.mean(max_confidences), color='red', linestyle='--', 
               label=f'Mean: {np.mean(max_confidences):.3f}')
    ax7.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Max Confidence')
    ax7.set_ylabel('Frequency')
    ax7.legend()
    
    # Learning rate history (if available)
    ax8 = plt.subplot(3, 3, 8)
    if 'lr' in history.history:
        ax8.plot(history.history['lr'], linewidth=2, color='orange')
        ax8.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Learning Rate')
        ax8.set_yscale('log')
    else:
        ax8.text(0.5, 0.5, 'Learning Rate\nHistory\nNot Available', 
                ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    # Model comparison summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary text
    summary_text = f"""
    Enhanced Model V2 Summary
    ========================
    
    Total Parameters: {model.count_params():,}
    
    Final Results:
    • Test Accuracy: {test_acc:.4f}
    • Training Samples: {len(X_train_mel):,}
    • Test Samples: {len(X_test_mel):,}
    • Enhanced Features: {enhanced_feature_size}
    
    Best Performing Classes:
    """
    
    # Add best and worst classes
    best_classes = np.argsort(class_accuracy)[-2:]  # Top 2
    worst_classes = np.argsort(class_accuracy)[:2]   # Bottom 2
    
    for idx in reversed(best_classes):
        summary_text += f"• {class_names[idx]}: {class_accuracy[idx]:.3f}\n    "
    
    summary_text += f"\nChallenging Classes:\n    "
    for idx in worst_classes:
        summary_text += f"• {class_names[idx]}: {class_accuracy[idx]:.3f}\n    "
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'enhanced_training_results.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 13: Detailed classification report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(classification_report(y_test_encoded, y_pred, target_names=class_names))
    
    # Step 14: Analyze specific confusions
    print("\n" + "="*80)
    print("CONFUSION ANALYSIS")
    print("="*80)
    
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i, :].sum()
                if confusion_rate > 0.1:  # Show confusions > 10%
                    print(f"• {true_class} → {pred_class}: {cm[i, j]} samples ({confusion_rate:.1%})")
    
    # Step 15: Save model and metadata
    print("\n💾 Saving model and metadata...")
    
    # Save model
    model.save(os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'enhanced_model.h5'))
    
    # Save scaler
    if EnhancedConfig.USE_ENHANCED_FEATURES:
        joblib.dump(scaler, os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'enhanced_feature_scaler.pkl'))
    
    # Save comprehensive metadata
    metadata = {
        'model_version': 'enhanced_v2',
        'class_names': class_names,
        'test_accuracy': float(test_acc),
        'per_class_accuracy': {class_names[i]: float(acc) for i, acc in enumerate(class_accuracy)},
        'model_type': 'multi_feature_cnn_enhanced',
        'enhanced_features': EnhancedConfig.USE_ENHANCED_FEATURES,
        'enhanced_feature_count': enhanced_feature_size,
        'session_aware_split': EnhancedConfig.USE_SESSION_AWARE_SPLIT,
        'audio_parameters': {
            'sample_rate': EnhancedConfig.SAMPLE_RATE,
            'segment_duration': EnhancedConfig.SEGMENT_DURATION,
            'n_mels': EnhancedConfig.N_MELS,
            'n_fft': EnhancedConfig.N_FFT,
            'hop_length': EnhancedConfig.HOP_LENGTH,
            'fmax': EnhancedConfig.FMAX
        },
        'training_parameters': {
            'batch_size': EnhancedConfig.BATCH_SIZE,
            'learning_rate': EnhancedConfig.LEARNING_RATE,
            'epochs_trained': len(history.history['accuracy']),
            'early_stopping_patience': EnhancedConfig.EARLY_STOPPING_PATIENCE,
            'dropout_rate': EnhancedConfig.DROPOUT_RATE,
            'l2_regularization': EnhancedConfig.L2_REGULARIZATION
        },
        'dataset_info': {
            'total_raw_files': len(raw_files),
            'training_samples': len(X_train_mel),
            'test_samples': len(X_test_mel),
            'augmentation_factor': len(X_train_mel) / len(train_files)
        },
        'mel_input_shape': list(mel_input_shape),
        'confusion_matrix': cm.tolist(),
        'tensorflow_version': tf.__version__,
        'feature_names': [
            # Base features
            'attack_slope_mean', 'onset_density', 'pitch_variation', 'vibrato_rate',
            'harmonic_ratio', 'percussive_ratio', 'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_rolloff_mean', 'energy_mean', 'energy_std',
            'energy_continuity', 'zcr_mean', 'texture_consistency', 'transient_density',
            # Pin-specific features
            'attack_steepness_mean', 'attack_steepness_std', 'decay_rate_mean', 'attack_to_decay_ratio',
            'sustain_duration', 'envelope_curvature', 'envelope_slope', 'fundamental_strength',
            'harmonic_complexity', 'spectral_regularity',
            # Khong_vong features
            'frequency_variation', 'beating_intensity', 'metallic_shimmer', 'mid_frequency_ratio',
            'resonance_duration', 'decay_characteristic'
        ]
    }
    
    with open(os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'enhanced_model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Step 16: Generate feature importance analysis
    print("\n🔍 Analyzing feature importance...")
    
    # Simple feature importance based on standard deviation (features with more variation are potentially more important)
    feature_importance = np.std(X_train_enhanced, axis=0)
    feature_names = metadata['feature_names']
    
    # Sort by importance
    importance_indices = np.argsort(feature_importance)[::-1]
    
    print("\nTop 10 Most Variable Features:")
    for i, idx in enumerate(importance_indices[:10]):
        print(f"{i+1:2d}. {feature_names[idx]:<25}: {feature_importance[idx]:.4f}")
    
    # Save feature importance plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(feature_importance)), feature_importance[importance_indices])
    plt.title('Feature Importance (by Standard Deviation)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Index (sorted by importance)')
    plt.ylabel('Standard Deviation')
    plt.xticks(range(0, len(feature_importance), 5), 
               [feature_names[idx][:10] + '...' for idx in importance_indices[::5]], 
               rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(EnhancedConfig.MODEL_SAVE_PATH, 'feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Enhanced training complete!")
    print(f"📁 Model saved to: {EnhancedConfig.MODEL_SAVE_PATH}")
    print(f"🎯 Final test accuracy: {test_acc:.4f}")
    print(f"📊 Enhanced features used: {enhanced_feature_size}")
    print(f"🔧 Total model parameters: {model.count_params():,}")
    
    return model, history, class_names, metadata

if __name__ == "__main__":
    train_enhanced_model()