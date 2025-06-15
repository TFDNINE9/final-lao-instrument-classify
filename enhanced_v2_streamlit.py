import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import joblib
import onnxruntime as ort
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎵 Lao Traditional Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stAlert > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .instrument-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .accuracy-high { color: #28a745; font-weight: bold; }
    .accuracy-medium { color: #ffc107; font-weight: bold; }
    .accuracy-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Configuration class
class AppConfig:
    MODEL_PATH = "models/enhanced_v2_model_6sec"
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Instrument information in Lao and English
    INSTRUMENT_INFO = {
        'khean': {
            'lao_name': 'ແຄນ',
            'english_name': 'Khean',
            'description': 'Traditional Lao mouth organ made of bamboo pipes. UNESCO World Heritage.',
            'type': 'Wind Instrument',
            'emoji': '🎐'
        },
        'saw': {
            'lao_name': 'ຊໍອູ້',
            'english_name': 'Saw',
            'description': 'Two-stringed fiddle with coconut shell resonator.',
            'type': 'String Instrument',
            'emoji': '🎻'
        },
        'pin': {
            'lao_name': 'ພິນ',
            'english_name': 'Pin',
            'description': 'Plucked string instrument similar to a lute.',
            'type': 'String Instrument',
            'emoji': '🪕'
        },
        'ranad': {
            'lao_name': 'ລະນາດ',
            'english_name': 'Ranad',
            'description': 'Wooden xylophone with bamboo or wooden bars.',
            'type': 'Percussion Instrument',
            'emoji': '🎼'
        },
        'khong_vong': {
            'lao_name': 'ຄ້ອງວົງ',
            'english_name': 'Khong Vong',
            'description': 'Circle of tuned gongs used in traditional ensembles.',
            'type': 'Percussion Instrument',
            'emoji': '🥁'
        },
        'sing': {
            'lao_name': 'ຊິ່ງ',
            'english_name': 'Sing',
            'description': 'Small cymbals used for rhythm in traditional music.',
            'type': 'Percussion Instrument',
            'emoji': '🎵'
        }
    }

@st.cache_resource
def load_model_components():
    """Load all model components with caching - reading from metadata file"""
    try:
        model_path = AppConfig.MODEL_PATH
        
        # Load metadata first - this contains everything we need
        metadata_path = f"{model_path}/enhanced_model_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract class names and feature names from metadata
        class_names = metadata['class_names']
        feature_names = metadata['feature_names']
        
        # Load ONNX model
        onnx_model_path = f"{model_path}/enhanced_model.onnx"
        ort_session = ort.InferenceSession(onnx_model_path)
        
        # Load feature scaler
        scaler_path = f"{model_path}/enhanced_feature_scaler.pkl"
        scaler = joblib.load(scaler_path)
        
        return ort_session, scaler, metadata, class_names, feature_names
    
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None, None, None

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
    """Extract base features from audio"""
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

def preprocess_audio(audio_file):
    """Preprocess uploaded audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=AppConfig.SAMPLE_RATE)
        
        # Get best segment
        best_segment = process_audio_with_best_segment(audio, sr, AppConfig.SEGMENT_DURATION)
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=best_segment,
            sr=sr,
            n_fft=AppConfig.N_FFT,
            hop_length=AppConfig.HOP_LENGTH,
            n_mels=AppConfig.N_MELS,
            fmax=AppConfig.FMAX
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
        
        # Extract enhanced features
        enhanced_feat = extract_all_enhanced_features(best_segment, sr)
        
        # Convert to array (32 features total) - in the exact order from your training
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
        
        return mel_spec_with_channel, feat_array, best_segment, sr
        
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None, None, None, None

def predict_instrument(ort_session, scaler, mel_features, enhanced_features, class_names):
    """Make prediction using ONNX model"""
    try:
        # Prepare inputs
        mel_input = np.expand_dims(mel_features, axis=0).astype(np.float32)
        enhanced_input = scaler.transform(enhanced_features.reshape(1, -1)).astype(np.float32)
        
        # Run inference
        onnx_inputs = {
            'mel_input': mel_input,
            'enhanced_input': enhanced_input
        }
        
        predictions = ort_session.run(None, onnx_inputs)[0]
        probabilities = predictions[0]
        
        # Create results
        results = []
        for i, class_name in enumerate(class_names):
            results.append({
                'instrument': class_name,
                'probability': float(probabilities[i]),
                'confidence': f"{probabilities[i]*100:.1f}%"
            })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        return results
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def create_visualization_plots(audio, sr, mel_spec, predictions):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Waveform
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0, 0].plot(time, audio, color='blue', linewidth=0.8)
    axes[0, 0].set_title('Audio Waveform', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Mel-spectrogram
    librosa.display.specshow(mel_spec[:, :, 0], sr=sr, hop_length=AppConfig.HOP_LENGTH,
                            x_axis='time', y_axis='mel', ax=axes[0, 1], fmax=AppConfig.FMAX)
    axes[0, 1].set_title('Mel-Spectrogram', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mel Frequency')
    
    # 3. Prediction probabilities
    instruments = [p['instrument'] for p in predictions]
    probabilities = [p['probability'] for p in predictions]
    colors = plt.cm.viridis(np.linspace(0, 1, len(instruments)))
    
    bars = axes[1, 0].bar(instruments, probabilities, color=colors)
    axes[1, 0].set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Spectral features
    try:
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        time_frames = librosa.frames_to_time(np.arange(len(spectral_centroids)), sr=sr, hop_length=AppConfig.HOP_LENGTH)
        
        axes[1, 1].plot(time_frames, spectral_centroids, label='Spectral Centroid', color='red', linewidth=1.5)
        ax2 = axes[1, 1].twinx()
        ax2.plot(time_frames, spectral_bandwidth, label='Spectral Bandwidth', color='green', linewidth=1.5)
        
        axes[1, 1].set_title('Spectral Features', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Frequency (Hz)', color='red')
        ax2.set_ylabel('Bandwidth (Hz)', color='green')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
    except:
        axes[1, 1].text(0.5, 0.5, 'Spectral analysis\nnot available', ha='center', va='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
    
    plt.tight_layout()
    return fig

def display_instrument_info(instrument_name):
    """Display information about the predicted instrument"""
    if instrument_name in AppConfig.INSTRUMENT_INFO:
        info = AppConfig.INSTRUMENT_INFO[instrument_name]
        
        st.markdown(f"""
        <div class="instrument-card">
            <h3>{info['emoji']} {info['english_name']} ({info['lao_name']})</h3>
            <p><strong>Type:</strong> {info['type']}</p>
            <p><strong>Description:</strong> {info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🎵 ການຈໍາແນກເຄື່ອງດົນຕີລາວ")
    st.title("Lao Traditional Instrument Classifier")
    
    st.markdown("""
    This AI-powered application can identify six traditional Lao musical instruments from audio recordings.
    Upload an audio file to discover which instrument is being played!
    
    **Supported Instruments:** Khean (ແຄນ), Saw (ຊໍອູ້), Pin (ພິນ), Ranad (ລະນາດ), Khong Vong (ຄ້ອງວົງ), Sing (ຊິ່ງ)
    """)
    
    # Load model components
    ort_session, scaler, metadata, class_names, feature_names = load_model_components()
    
    if ort_session is None:
        st.error("Failed to load model components. Please check the model files.")
        return
    
    # Sidebar with model information
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Model Version:** {metadata.get('model_version', 'Unknown')}")
        st.write(f"**Test Accuracy:** {metadata.get('test_accuracy', 0)*100:.1f}%")
        st.write(f"**Enhanced Features:** {metadata.get('enhanced_feature_count', 0)}")
        
        st.header("🎯 Per-Class Accuracy")
        per_class_acc = metadata.get('per_class_accuracy', {})
        for instrument, accuracy in per_class_acc.items():
            if accuracy > 0.8:
                acc_class = "accuracy-high"
            elif accuracy > 0.5:
                acc_class = "accuracy-medium"
            else:
                acc_class = "accuracy-low"
            
            info = AppConfig.INSTRUMENT_INFO.get(instrument, {})
            emoji = info.get('emoji', '🎵')
            lao_name = info.get('lao_name', instrument)
            
            st.markdown(f"""
            <div style="margin: 0.2rem 0;">
                {emoji} {instrument} ({lao_name}): 
                <span class="{acc_class}">{accuracy*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("ℹ️ How to Use")
        st.markdown("""
        1. Upload an audio file (.wav, .mp3, .flac)
        2. Wait for processing (6-second segment)
        3. View prediction results
        4. Explore visualizations
        """)
    
    # File upload
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file...",
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a recording of a Lao traditional instrument"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
        st.info(f"File size: {file_size:.2f} MB")
        
        # Process audio
        with st.spinner("🔄 Processing audio... This may take a moment."):
            mel_features, enhanced_features, audio, sr = preprocess_audio(uploaded_file)
        
        if mel_features is not None and enhanced_features is not None:
            st.success("✅ Audio preprocessing complete!")
            
            # Make prediction
            with st.spinner("🎯 Making prediction..."):
                predictions = predict_instrument(ort_session, scaler, mel_features, enhanced_features, class_names)
            
            if predictions is not None:
                # Display main prediction
                top_prediction = predictions[0]
                confidence = top_prediction['probability']
                
                st.header("🎯 Prediction Results")
                
                # Main prediction card
                if confidence > 0.7:
                    confidence_color = "#28a745"  # Green
                    confidence_text = "High Confidence"
                elif confidence > 0.4:
                    confidence_color = "#ffc107"  # Yellow
                    confidence_text = "Medium Confidence"
                else:
                    confidence_color = "#dc3545"  # Red
                    confidence_text = "Low Confidence"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2 style="color: {confidence_color}; margin: 0;">
                        Predicted Instrument: {top_prediction['instrument'].title()}
                    </h2>
                    <h3 style="margin: 0.5rem 0;">
                        Confidence: {top_prediction['confidence']} ({confidence_text})
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display instrument information
                display_instrument_info(top_prediction['instrument'])
                
                # All predictions
                st.header("📊 All Predictions")
                
                # Create DataFrame for better display
                results_df = pd.DataFrame(predictions)
                results_df['Confidence'] = results_df['probability'].apply(lambda x: f"{x*100:.1f}%")
                results_df['Instrument'] = results_df['instrument'].apply(lambda x: x.title())
                
                # Add emoji and Lao names
                def add_instrument_info(instrument):
                    info = AppConfig.INSTRUMENT_INFO.get(instrument.lower(), {})
                    emoji = info.get('emoji', '🎵')
                    lao_name = info.get('lao_name', instrument)
                    return f"{emoji} {instrument} ({lao_name})"
                
                results_df['Full Name'] = results_df['instrument'].apply(add_instrument_info)
                
                # Display as colored bars
                for _, row in results_df.iterrows():
                    prob = row['probability']
                    bar_width = int(prob * 100)
                    
                    if prob > 0.7:
                        bar_color = "#28a745"
                    elif prob > 0.4:
                        bar_color = "#ffc107"
                    else:
                        bar_color = "#dc3545"
                    
                    st.markdown(f"""
                    <div style="margin: 0.5rem 0;">
                        <div style="font-weight: bold; margin-bottom: 0.2rem;">
                            {row['Full Name']} - {row['Confidence']}
                        </div>
                        <div style="background: #e9ecef; border-radius: 10px; height: 20px; position: relative;">
                            <div style="background: {bar_color}; width: {bar_width}%; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualizations
                st.header("📈 Audio Analysis Visualizations")
                
                try:
                    fig = create_visualization_plots(audio, sr, mel_features, predictions)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error creating visualizations: {e}")
                
                # Audio playback
                st.header("🔊 Audio Playback")
                try:
                    # Convert audio to bytes for playback
                    audio_bytes = BytesIO()
                    import soundfile as sf
                    sf.write(audio_bytes, audio, sr, format='WAV')
                    audio_bytes.seek(0)
                    
                    st.audio(audio_bytes.read(), format='audio/wav')
                except Exception as e:
                    st.warning("Audio playback not available - soundfile library may not be installed")
                
                # Feature analysis (expandable)
                with st.expander("🔬 Advanced Feature Analysis"):
                    st.subheader("Enhanced Audio Features")
                    
                    # Create feature DataFrame
                    feature_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Value': enhanced_features
                    })
                    
                    # Normalize for visualization
                    normalized_values = (enhanced_features - np.min(enhanced_features)) / (np.max(enhanced_features) - np.min(enhanced_features) + 1e-8)
                    feature_df['Normalized'] = normalized_values
                    
                    # Display top features
                    st.write("**Top 10 Most Significant Features:**")
                    top_features = feature_df.nlargest(10, 'Normalized')
                    
                    for _, row in top_features.iterrows():
                        st.write(f"• **{row['Feature']}**: {row['Value']:.4f}")
                    
                    # Feature visualization
                    fig, ax = plt.subplots(figsize=(12, 8))
                    bars = ax.barh(range(len(feature_names)), enhanced_features)
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in feature_names])
                    ax.set_xlabel('Feature Value')
                    ax.set_title('All Enhanced Audio Features')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Model performance insights
                with st.expander("📊 Model Performance Insights"):
                    st.subheader("Understanding the Results")
                    
                    # Show confusion matrix if available
                    if 'confusion_matrix' in metadata:
                        cm = np.array(metadata['confusion_matrix'])
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=class_names, yticklabels=class_names, ax=ax)
                        ax.set_title('Model Confusion Matrix', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('True')
                        st.pyplot(fig)
                        
                        st.write("**Interpretation:**")
                        st.write("- Diagonal values show correct predictions")
                        st.write("- Off-diagonal values show common confusions")
                        st.write("- Darker colors indicate higher counts")
                    
                    # Performance tips
                    st.subheader("Tips for Better Results")
                    st.markdown("""
                    **For best results:**
                    - Use clear, high-quality recordings
                    - Ensure the instrument is the dominant sound
                    - Avoid background noise or multiple instruments
                    - Recordings of 3-10 seconds work best
                    
                    **Model Strengths:**
                    - Excellent at identifying: Ranad (100%), Sing (100%), Saw (93.1%)
                    - Good at identifying: Khean (82.1%), Khong Vong (62.5%)
                    
                    **Model Challenges:**
                    - Pin instrument has lower accuracy (3.6%) - often confused with other string instruments
                    - Consider this when interpreting Pin predictions
                    """)
            
            else:
                st.error("❌ Prediction failed. Please try a different audio file.")
        
        else:
            st.error("❌ Audio preprocessing failed. Please check your audio file format and try again.")
    
    # Footer with additional information
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        ### 🎵 Lao Traditional Instrument Classifier
        
        This application uses deep learning to identify traditional Lao musical instruments from audio recordings.
        
        **Technology Stack:**
        - **Deep Learning Framework:** TensorFlow/Keras
        - **Model Architecture:** Multi-feature CNN with enhanced audio features
        - **Audio Processing:** Librosa library for feature extraction
        - **Deployment:** ONNX Runtime for optimized inference
        - **Interface:** Streamlit for web application
        
        **Model Details:**
        - **Training Data:** 1,392 training samples, 175 test samples
        - **Features:** 32 enhanced audio features + Mel-spectrograms
        - **Accuracy:** 73.7% overall test accuracy
        - **Input:** 6-second audio segments at 44.1kHz
        
        **Cultural Significance:**
        Traditional Lao instruments represent centuries of cultural heritage. This project aims to preserve
        and promote understanding of these beautiful instruments through modern AI technology.
        
        **Supported Instruments:**
        1. **Khean (ແຄນ)** - UNESCO World Heritage bamboo mouth organ
        2. **Saw (ຊໍອູ້)** - Two-stringed fiddle with coconut resonator
        3. **Pin (ພິນ)** - Plucked lute-like string instrument
        4. **Ranad (ລະນາດ)** - Wooden xylophone with bamboo bars
        5. **Khong Vong (ຄ້ອງວົງ)** - Circle of tuned bronze gongs
        6. **Sing (ຊິ່ງ)** - Small bronze cymbals for rhythm
        """)
    
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        🎵 <strong>Lao Traditional Instrument Classifier</strong><br>
        Built with deep learning for cultural preservation<br>
        <small>Supporting traditional Lao music and culture through AI</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()