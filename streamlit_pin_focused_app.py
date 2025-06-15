import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import onnxruntime as ort
import json
import os
import tempfile
import soundfile as sf
import time
import wave
import pyaudio
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Lao Instrument Classifier (Pin-Focused)",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model configuration and parameters
@st.cache_resource
def load_pin_focused_model(model_dir='models/pin_focused_model_6sec'):
    """Load pin-focused model configuration and parameters"""
    model_path = os.path.join(model_dir, 'pin_focused_model.onnx')
    metadata_path = os.path.join(model_dir, 'pin_focused_metadata.json')
    scaler_path = os.path.join(model_dir, 'pin_focused_scaler.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None, None
    
    if not os.path.exists(metadata_path):
        st.error(f"Metadata file not found: {metadata_path}")
        return None, None, None
    
    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found: {scaler_path}")
        return None, None, None
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Create ONNX inference session
    try:
        session = ort.InferenceSession(model_path)
        return session, metadata, scaler
    except Exception as e:
        st.error(f"Error loading ONNX model: {str(e)}")
        return None, None, None

# Load model and metadata
model_session, model_metadata, feature_scaler = load_pin_focused_model()

# Extract parameters from loaded config
if model_metadata is not None:
    SAMPLE_RATE = model_metadata['audio_parameters']['sample_rate']
    N_MELS = model_metadata['audio_parameters']['n_mels']
    N_FFT = model_metadata['audio_parameters']['n_fft']
    HOP_LENGTH = model_metadata['audio_parameters']['hop_length']
    SEGMENT_DURATION = model_metadata['audio_parameters']['segment_duration']
    FMAX = model_metadata['audio_parameters']['fmax']
    CLASS_LABELS = model_metadata['class_names']
    ENHANCED_FEATURES = model_metadata['enhanced_features']
    PIN_FEATURES = model_metadata.get('pin_specific_features', 10)
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_DURATION = 6.0
    FMAX = 8000
    ENHANCED_FEATURES = 42
    PIN_FEATURES = 10
    CLASS_LABELS = ['khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']

# Recording configuration
RECORD_SECONDS = 12  # Extra time to ensure 6-second segments
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_RATE = 44100

# Enhanced instrument information
INSTRUMENT_INFO = {
    'khean': {
        'name': 'Khaen (ແຄນ)',
        'description': 'A mouth organ made of bamboo pipes, each with a metal reed. UNESCO recognized it as part of Lao\'s intangible cultural heritage in 2017.',
        'sound_characteristics': 'Continuous, buzzing sound with multiple simultaneous notes and drone-like quality',
        'cultural_significance': 'Central to Lao cultural identity, featured in ceremonies and traditional performances.',
        'model_performance': 'Good accuracy (82.1%)'
    },
    'khong_vong': {
        'name': 'Khong Wong (ຄ້ອງວົງ)',
        'description': 'Circular arrangement of small gongs in a wooden frame, used in ceremonial music.',
        'sound_characteristics': 'Clear, resonant metallic tones with long sustain and beating patterns',
        'cultural_significance': 'Central instrument in traditional Lao ensembles.',
        'model_performance': 'Good accuracy (71.9%)'
    },
    'pin': {
        'name': 'Pin (ພິນ)',
        'description': 'Plucked string instrument with coconut shell or hardwood resonator, similar to a lute.',
        'sound_characteristics': 'Sharp plucking attack followed by exponential decay, deep warm tones with prominent bass',
        'cultural_significance': 'Featured in folk music and storytelling traditions.',
        'model_performance': 'Challenging instrument (57.1% accuracy) - improved with pin-focused features'
    },
    'ranad': {
        'name': 'Ranad (ລະນາດ)',
        'description': 'Wooden xylophone with bamboo resonators underneath.',
        'sound_characteristics': 'Bright, percussive wooden tones with moderate resonance and clear attacks',
        'cultural_significance': 'Important role in traditional Lao folk music ensembles.',
        'model_performance': 'Excellent accuracy (93.3%)'
    },
    'saw': {
        'name': 'So U (ຊໍອູ້)',
        'description': 'Bowed string instrument with coconut shell resonator.',
        'sound_characteristics': 'Lyrical, singing quality with continuous sustained tones and vibrato',
        'cultural_significance': 'Often imitates human voice, used for emotional expression in music.',
        'model_performance': 'Excellent accuracy (93.1%)'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'Small cymbal-like percussion instrument used in ensembles.',
        'sound_characteristics': 'Sharp, bright metallic sound with quick decay and shimmer',
        'cultural_significance': 'Provides rhythmic structure in traditional ensembles.',
        'model_performance': 'Perfect accuracy (100%)'
    }
}

def extract_pin_focused_features(audio, sr):
    """Extract pin-focused features (same as training code)"""
    features = {}
    
    try:
        # PIN-SPECIFIC FEATURES
        rms_energy = librosa.feature.rms(y=audio, hop_length=256)[0]
        
        # Find attack points
        energy_diff = np.diff(rms_energy)
        attack_points = np.where(energy_diff > np.std(energy_diff) * 3)[0]
        
        if len(attack_points) > 0:
            attack_sharpness_scores = []
            decay_scores = []
            
            for attack_idx in attack_points[:5]:
                before_samples = 10
                after_samples = 40
                
                start_idx = max(0, attack_idx - before_samples)
                end_idx = min(len(rms_energy), attack_idx + after_samples)
                
                if end_idx - start_idx > 20:
                    segment = rms_energy[start_idx:end_idx]
                    attack_point = before_samples if attack_idx >= before_samples else attack_idx
                    
                    if attack_point > 0 and attack_point < len(segment) - 1:
                        pre_attack = segment[:attack_point]
                        post_attack = segment[attack_point:attack_point+5]
                        
                        if len(pre_attack) > 0 and len(post_attack) > 0:
                            attack_rise = np.max(post_attack) - np.mean(pre_attack)
                            attack_sharpness_scores.append(attack_rise)
                            
                            # Decay analysis
                            decay_portion = segment[attack_point:]
                            if len(decay_portion) > 10:
                                x = np.arange(len(decay_portion))
                                y = decay_portion + 1e-8
                                
                                try:
                                    log_y = np.log(y)
                                    if not np.any(np.isinf(log_y)) and not np.any(np.isnan(log_y)):
                                        decay_coeff = np.polyfit(x, log_y, 1)[0]
                                        decay_scores.append(abs(decay_coeff))
                                except:
                                    pass
            
            features['pin_attack_sharpness'] = np.mean(attack_sharpness_scores) if attack_sharpness_scores else 0
            features['pin_decay_rate'] = np.mean(decay_scores) if decay_scores else 0
            features['pin_attack_count'] = len(attack_points) / (len(audio) / sr)
        else:
            features['pin_attack_sharpness'] = 0
            features['pin_decay_rate'] = 0
            features['pin_attack_count'] = 0
        
        # Harmonic analysis
        stft = np.abs(librosa.stft(audio, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr)
        
        fundamental_range = (80, 400)
        fund_mask = (freqs >= fundamental_range[0]) & (freqs <= fundamental_range[1])
        h2_mask = (freqs >= fundamental_range[0]*2) & (freqs <= fundamental_range[1]*2)
        h3_mask = (freqs >= fundamental_range[0]*3) & (freqs <= fundamental_range[1]*3)
        
        fund_energy = np.mean(stft[fund_mask, :])
        h2_energy = np.mean(stft[h2_mask, :])
        h3_energy = np.mean(stft[h3_mask, :])
        
        total_harmonic = fund_energy + h2_energy + h3_energy + 1e-8
        
        features['pin_fundamental_strength'] = fund_energy / total_harmonic
        features['pin_harmonic_clarity'] = (h2_energy + h3_energy) / (fund_energy + 1e-8)
        
        # Sustain analysis
        energy_envelope = rms_energy
        peak_energy = np.max(energy_envelope)
        
        sustain_10 = np.sum(energy_envelope > peak_energy * 0.1) / len(energy_envelope)
        sustain_05 = np.sum(energy_envelope > peak_energy * 0.05) / len(energy_envelope)
        
        features['pin_sustain_10'] = sustain_10
        features['pin_sustain_05'] = sustain_05
        features['pin_sustain_ratio'] = sustain_05 / (sustain_10 + 1e-8)
        
        # Attack energy ratio
        if len(attack_points) > 0:
            attack_energy = 0
            total_energy = np.sum(energy_envelope**2)
            
            for attack_idx in attack_points:
                start = max(0, attack_idx - 2)
                end = min(len(energy_envelope), attack_idx + 8)
                attack_energy += np.sum(energy_envelope[start:end]**2)
            
            features['pin_attack_energy_ratio'] = attack_energy / (total_energy + 1e-8)
        else:
            features['pin_attack_energy_ratio'] = 0
        
        # Spectral evolution
        if stft.shape[1] > 10:
            early_frames = stft[:, :stft.shape[1]//3]
            late_frames = stft[:, -stft.shape[1]//3:]
            
            early_centroid = np.sum(freqs[:, np.newaxis] * early_frames) / (np.sum(early_frames, axis=0) + 1e-8)
            late_centroid = np.sum(freqs[:, np.newaxis] * late_frames) / (np.sum(late_frames, axis=0) + 1e-8)
            
            features['pin_spectral_evolution'] = np.mean(early_centroid) - np.mean(late_centroid)
        else:
            features['pin_spectral_evolution'] = 0
            
    except Exception as e:
        # Return neutral values if extraction fails
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
    """Extract fast enhanced features (32 features)"""
    features = {}
    
    try:
        # Pre-compute operations
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        stft_mag = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        rms_energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=512)[0] 
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=512)[0]
        zero_crossings = librosa.feature.zero_crossing_rate(audio, hop_length=512)[0]
        
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
        
        # Onset detection
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
        
        # Frequency analysis
        low_freq_mask = freqs < 500
        high_freq_mask = freqs > 2000
        mid_freq_mask = (freqs >= 500) & (freqs <= 2000)
        
        low_energy = np.mean(stft_mag[low_freq_mask, :])
        mid_energy = np.mean(stft_mag[mid_freq_mask, :])
        high_energy = np.mean(stft_mag[high_freq_mask, :])
        total_energy = low_energy + mid_energy + high_energy + 1e-8
        
        features['harmonic_ratio'] = (low_energy + mid_energy) / total_energy
        features['percussive_ratio'] = high_energy / total_energy
        
        # Fill remaining features (32 total)
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
        
    except Exception as e:
        # Return zeros if extraction fails
        features = {f'feature_{i}': 0.0 for i in range(32)}
    
    return features

def extract_ultra_enhanced_features(audio, sr):
    """Extract all 42 features (32 + 10 pin-specific)"""
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

def predict_instrument(audio_data, sr):
    """Process audio and make prediction using pin-focused ONNX model"""
    if model_session is None or feature_scaler is None:
        st.error("Model not loaded properly!")
        return None
    
    try:
        # Ensure correct sample rate
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Get best segment
        best_segment = process_audio_with_best_segment(audio_data, sr, SEGMENT_DURATION)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=best_segment, sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmax=FMAX
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
        mel_spec_batch = np.expand_dims(mel_spec_with_channel, axis=0).astype(np.float32)
        
        # Extract enhanced features
        enhanced_feat = extract_ultra_enhanced_features(best_segment, sr)
        
        # Convert to array (42 features)
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
            
            # Pin-specific features (10)
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
        
        # Normalize features
        feat_normalized = feature_scaler.transform(feat_array.reshape(1, -1))
        feat_batch = feat_normalized.astype(np.float32)
        
        # Get input names
        input_names = [input.name for input in model_session.get_inputs()]
        
        # Run inference
        outputs = model_session.run(None, {
            input_names[0]: mel_spec_batch,
            input_names[1]: feat_batch
        })
        
        # Process results
        probabilities = outputs[0][0]  # First output, first batch item
        
        # Create result dictionary
        result = {
            'probabilities': {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'mel_spectrogram': mel_spec_db,
            'enhanced_features': feat_array
        }
        
        # Find the most likely instrument
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        instrument = CLASS_LABELS[max_prob_idx]
        
        # Calculate entropy as uncertainty measure
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log2(probabilities + epsilon)) / np.log2(len(probabilities))
        
        # Determine if prediction is uncertain
        is_uncertain = entropy > 0.6 or max_prob < 0.4
        
        result.update({
            'instrument': instrument,
            'confidence': float(max_prob),
            'entropy': float(entropy),
            'is_uncertain': is_uncertain
        })
        
        return result
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def plot_mel_spectrogram(mel_spec, sr, hop_length):
    """Create visualization of the mel spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    img = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=hop_length,
        fmax=FMAX,
        ax=ax
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(f'Mel Spectrogram ({SEGMENT_DURATION}s Best Segment)')
    
    plt.tight_layout()
    return fig

def plot_waveform(audio, sr):
    """Create visualization of the audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 2))
    
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Audio Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    return fig

def plot_classification_probabilities(result):
    """Create bar chart of classification probabilities with pin highlighting"""
    if not result or 'probabilities' not in result:
        return None
    
    probs = result['probabilities']
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    instruments = [INSTRUMENT_INFO.get(label, {}).get('name', label) for label, _ in sorted_probs]
    values = [prob * 100 for _, prob in sorted_probs]
    
    # Create plot with pin highlighting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars - highlight pin in red
    colors = ['red' if label == 'pin' else 'skyblue' for label, _ in sorted_probs]
    bars = ax.barh(instruments, values, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_color = 'white' if sorted_probs[i][0] == 'pin' else 'black'
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                ha='left', va='center', fontweight='bold', color=label_color)
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Pin-Focused Model: Instrument Classification Results')
    
    # Add model info
    ax.text(0.02, 0.98, f'Model: Pin-Focused (42 features)\nTotal Parameters: 36,758', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_pin_features(enhanced_features):
    """Plot pin-specific features"""
    if enhanced_features is None or len(enhanced_features) < 42:
        return None
    
    # Last 10 features are pin-specific
    pin_features = enhanced_features[-10:]
    pin_feature_names = [
        'Attack Sharpness', 'Decay Rate', 'Attack Count',
        'Fundamental Strength', 'Harmonic Clarity', 'Sustain 10%',
        'Sustain 5%', 'Sustain Ratio', 'Attack Energy Ratio',
        'Spectral Evolution'
    ]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(pin_feature_names, pin_features, color='red', alpha=0.7)
    ax.set_title('Pin-Specific Features (Normalized Values)')
    ax.set_ylabel('Feature Value')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, pin_features):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig

def display_instrument_info(instrument_id, confidence):
    """Display enhanced information about a specific instrument"""
    if instrument_id not in INSTRUMENT_INFO:
        st.warning(f"Information not available for instrument: {instrument_id}")
        return
    
    info = INSTRUMENT_INFO[instrument_id]
    
    st.markdown(f"<div class='instrument-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display model performance badge
        if instrument_id == 'pin':
            st.markdown("🎯 **PIN-FOCUSED MODEL**")
            st.markdown("*Specialized features for plucked strings*")
        
        st.write(f"## {info['name']}")
    
    with col2:
        st.markdown(f"**Confidence**: {confidence*100:.1f}%")
        st.markdown(f"**Description**: {info['description']}")
        st.markdown(f"**Sound Characteristics**: {info['sound_characteristics']}")
        st.markdown(f"**Cultural Significance**: {info['cultural_significance']}")
        st.markdown(f"**Model Performance**: {info['model_performance']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def record_audio_with_progress():
    """Record audio with progress bar"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    recording_info = st.info(f"""
    📢 **Recording Tips for Pin-Focused Model:**
    - For **Pin (ພິນ)**: Pluck strings clearly with sharp attacks
    - Recording for {RECORD_SECONDS} seconds to capture multiple plucks/notes
    - Hold instrument close to microphone
    - Play continuously during recording
    - The model has specialized features for detecting plucking patterns
    """)
    
    # Create PyAudio instance
    audio = pyaudio.PyAudio()
    
    # List available input devices
    available_devices = []
    default_device_index = None
    
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            available_devices.append((i, info.get('name')))
            if default_device_index is None:
                default_device_index = i
    
    # Device selection
    if len(available_devices) > 1:
        device_options = {device_name: idx for idx, device_name in available_devices}
        selected_device_name = st.selectbox("Select input device:", list(device_options.keys()))
        selected_device_index = device_options[selected_device_name]
    elif len(available_devices) == 1:
        selected_device_index = available_devices[0][0]
        st.write(f"Using audio input device: {available_devices[0][1]}")
    else:
        st.error("No input devices found!")
        return None, None
    
    try:
        # Open stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECORD_RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=selected_device_index
        )
        
        frames = []
        start_time = time.time()
        
        status_placeholder.text("🔴 Recording...")
        
        for i in range(0, int(RECORD_RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                st.error(f"Error reading from microphone: {e}")
                break
            
            # Update progress
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / RECORD_SECONDS)
            remaining = max(0, RECORD_SECONDS - elapsed)
            progress_placeholder.progress(progress)
            status_placeholder.text(f"🔴 Recording... {remaining:.1f} seconds remaining")
        
        # Stop and close
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        status_placeholder.text("Processing recording...")
        
        # Create audio file
        audio_data = b''.join(frames)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RECORD_RATE)
                wf.writeframes(audio_data)
        
        # Load with librosa
        audio_array, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        
        # Clear placeholders
        progress_placeholder.empty()
        status_placeholder.empty()
        recording_info.empty()
        
        # Check audio levels
        if np.abs(audio_array).max() < 0.05:
            st.warning("⚠️ Very low audio levels detected. Try recording closer to the microphone.")
        
        return audio_array, tmp_path
        
    except Exception as e:
        if 'audio' in locals():
            audio.terminate()
        status_placeholder.empty()
        progress_placeholder.empty()
        recording_info.empty()
        raise Exception(f"Error recording audio: {str(e)}")

def display_results(audio_data, sr, result):
    """Display analysis results with pin-focused information"""
    st.subheader("Pin-Focused Model Analysis Results")
    
    # Display prediction result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if result['is_uncertain']:
            st.warning(f"⚠️ Uncertain classification (Confidence: {result['confidence']*100:.1f}%)")
            st.write("The model is not confident about its prediction.")
        else:
            instrument_id = result['instrument']
            info = INSTRUMENT_INFO.get(instrument_id, {})
            
            if instrument_id == 'pin':
                st.success(f"🎯 **PIN DETECTED**: {info.get('name', instrument_id)}")
                st.write("*Pin-focused model with specialized plucking detection*")
            else:
                st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
            
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
    
    with col2:
        # Plot probabilities
        prob_fig = plot_classification_probabilities(result)
        if prob_fig:
            st.pyplot(prob_fig)
    
    # Pin-specific features (if pin was detected or had reasonable probability)
    if result['instrument'] == 'pin' or result['probabilities'].get('pin', 0) > 0.1:
        st.subheader("🎯 Pin-Specific Feature Analysis")
        pin_fig = plot_pin_features(result.get('enhanced_features'))
        if pin_fig:
            st.pyplot(pin_fig)
            st.markdown("""
            <div class='info-box'>
            These 10 specialized features help distinguish Pin from other instruments:
            • **Attack Sharpness**: Detects sharp plucking attacks
            • **Decay Rate**: Measures exponential string decay
            • **Harmonic Clarity**: Analyzes string harmonic series
            • **Sustain Patterns**: String resonance characteristics
            </div>
            """, unsafe_allow_html=True)
    
    # Audio visualizations
    st.subheader("Audio Visualization")
    
    viz_tab1, viz_tab2 = st.tabs(["Mel Spectrogram", "Waveform"])
    
    with viz_tab1:
        mel_fig = plot_mel_spectrogram(result['mel_spectrogram'], sr, HOP_LENGTH)
        st.pyplot(mel_fig)
        st.markdown(f"""
        <div class='info-box'>
        The Mel Spectrogram shows the frequency content over time for the best {SEGMENT_DURATION}-second segment.
        The pin-focused model uses this along with 42 enhanced features for classification.
        </div>
        """, unsafe_allow_html=True)
    
    with viz_tab2:
        waveform_fig = plot_waveform(audio_data, sr)
        st.pyplot(waveform_fig)
    
    # Show detailed instrument information
    if not result['is_uncertain']:
        st.subheader("Instrument Information")
        display_instrument_info(result['instrument'], result['confidence'])

def main():
    # Initialize session state
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
        st.session_state.recorded_audio_path = None
        st.session_state.recording_result = None
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Pin-Focused Lao Instrument Classifier")
        st.markdown("---")
        
        st.subheader("About This Model")
        st.markdown(f"""
        This **pin-focused model** is specifically enhanced to better classify the **Pin (ພິນ)** instrument
        using specialized features for plucked strings.
        
        **Model Specifications:**
        - **42 enhanced features** (32 general + 10 pin-specific)
        - **{SEGMENT_DURATION}-second** segment analysis
        - **Specialized plucking detection**
        - **Multi-branch neural network**
        """)
        
        st.subheader("Supported Instruments:")
        for label in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(label, {})
            name = info.get('name', label)
            perf = info.get('model_performance', 'Good performance')
            if label == 'pin':
                st.markdown(f"🎯 **{name}** - *{perf}*")
            else:
                st.markdown(f"• **{name}** - {perf}")
        
        st.markdown("---")
        
        st.subheader("Technical Details")
        if model_metadata:
            st.markdown(f"""
            - **Total Parameters**: {model_metadata.get('dataset_info', {}).get('training_samples', 'N/A')} training samples
            - **Test Accuracy**: {model_metadata.get('test_accuracy', 0)*100:.1f}%
            - **Pin Accuracy**: {model_metadata.get('pin_specific_accuracy', 0)*100:.1f}%
            - **Enhanced Features**: {ENHANCED_FEATURES}
            - **Pin Features**: {PIN_FEATURES}
            """)
        
        st.markdown(f"""
        - **Sample Rate**: {SAMPLE_RATE} Hz
        - **Recording Duration**: {RECORD_SECONDS} seconds
        - **Analysis Duration**: {SEGMENT_DURATION} seconds
        """)
    
    # Main content
    st.markdown(f"""
    <div class='title-area'>
        <h1>🎯 Pin-Focused Lao Instrument Classifier</h1>
        <p>Enhanced model with specialized features for Pin (ພິນ) detection and 42 total acoustic features.</p>
        <p><strong>Upload an audio file or record live audio of a Lao musical instrument for classification.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status
    if model_metadata:
        st.success(f"✅ Pin-focused model loaded successfully! Test accuracy: {model_metadata.get('test_accuracy', 0)*100:.1f}%")
    else:
        st.error("❌ Model not loaded. Please check the model files.")
        return
    
    # Create tabs for input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            with st.spinner("Processing audio with pin-focused model..."):
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)
                    
                    result = predict_instrument(audio_data, sr)
                    
                    if result:
                        display_results(audio_data, sr, result)
                    else:
                        st.error("Failed to process the audio. Please try another file.")
                
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
    
    with tab2:
        st.subheader("Record Audio")
        st.markdown(f"""
        Record {RECORD_SECONDS} seconds of audio for analysis. The pin-focused model will analyze 
        the best {SEGMENT_DURATION}-second segment with specialized plucking detection.
        """)
        
        if st.button("🎙️ Start Recording", key="record_button"):
            try:
                audio_data, audio_path = record_audio_with_progress()
                
                st.session_state.recorded_audio = audio_data
                st.session_state.recorded_audio_path = audio_path
                
                result = predict_instrument(audio_data, SAMPLE_RATE)
                st.session_state.recording_result = result
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
        
        # Display recorded audio results
        if st.session_state.recorded_audio is not None and st.session_state.recorded_audio_path is not None:
            st.subheader("Recorded Audio Analysis")
            st.audio(st.session_state.recorded_audio_path)
            
            result = st.session_state.recording_result
            if result:
                display_results(st.session_state.recorded_audio, SAMPLE_RATE, result)
            else:
                st.error("Failed to process the recording. Please try again.")
    
    # Information about instruments
    st.markdown("---")
    with st.expander("Learn About Lao Musical Instruments & Model Performance"):
        for instrument_id in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(instrument_id, {})
            
            st.markdown(f"<div class='instrument-card'>", unsafe_allow_html=True)
            
            if instrument_id == 'pin':
                st.markdown(f"### 🎯 {info.get('name', instrument_id)} **(Pin-Focused Model)**")
            else:
                st.markdown(f"### {info.get('name', instrument_id)}")
            
            st.markdown(f"**Description**: {info.get('description', 'No description available')}")
            st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
            st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
            st.markdown(f"**Model Performance**: {info.get('model_performance', 'Not available')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")

# Custom CSS
def local_css():
    st.markdown("""
    <style>
    .title-area {
        text-align: center;
        padding: 1rem 0 2rem 0;
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .instrument-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #ff6b6b;
    }
    .info-box {
        padding: 0.5rem;
        background-color: #e9f7fe;
        border-left: 3px solid #0096ff;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply CSS and run app
local_css()

if __name__ == "__main__":
    main()