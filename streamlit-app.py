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

# Set page configuration
st.set_page_config(
    page_title="Enhanced Lao Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load enhanced model configuration and parameters
@st.cache_resource
def load_enhanced_model_config(model_dir='models/enhanced_mel_cnn_model_6sec'):
    """Load enhanced model configuration and parameters"""
    model_path = os.path.join(model_dir, 'enhanced_mel_cnn_model_6sec.onnx')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    # Fallback to original model if enhanced not found
    if not os.path.exists(model_path):
        st.warning("Enhanced model not found, falling back to original model...")
        model_path = os.path.join('models/mel_cnn_model_6sec', 'mel_cnn_model_6sec.onnx')
        metadata_path = os.path.join('models/mel_cnn_model_6sec', 'model_metadata.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None, None
    
    if not os.path.exists(metadata_path):
        st.error(f"Metadata file not found: {metadata_path}")
        return None, None
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create ONNX inference session
    try:
        session = ort.InferenceSession(model_path)
        return session, metadata
    except Exception as e:
        st.error(f"Error loading ONNX model: {str(e)}")
        return None, None

# Load enhanced model and metadata
model_session, model_metadata = load_enhanced_model_config()

# Extract parameters from loaded config
if model_metadata is not None:
    SAMPLE_RATE = model_metadata.get('sample_rate', 44100)
    N_MELS = model_metadata.get('n_mels', 128)
    N_FFT = model_metadata.get('n_fft', 2048)
    HOP_LENGTH = model_metadata.get('hop_length', 512)
    SEGMENT_DURATION = model_metadata.get('segment_duration', 6.0)
    FMAX = model_metadata.get('fmax', 8000)
    CLASS_LABELS = model_metadata.get('class_names', [])
    USE_ENHANCED_FEATURES = model_metadata.get('enhanced_features', False)
    FEATURE_STACK_SIZE = model_metadata.get('feature_stack_size', 128)
    MODEL_VERSION = model_metadata.get('model_version', 'standard')
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_DURATION = 6.0
    FMAX = 8000
    CLASS_LABELS = ['khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']
    USE_ENHANCED_FEATURES = False
    FEATURE_STACK_SIZE = 128
    MODEL_VERSION = 'standard'

# Recording configuration
RECORD_SECONDS = 12
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_RATE = 44100

# Instrument information
INSTRUMENT_INFO = {
    'khean': {
        'name': 'Khaen (ແຄນ)',
        'description': 'A mouth organ made of bamboo pipes, each with a metal reed. It is considered the symbol of Lao music and was recognized by UNESCO as part of Lao\'s intangible cultural heritage in 2017.',
        'sound_characteristics': 'Continuous, buzzing sound with the ability to play multiple notes simultaneously. Rich harmonic content from multiple reeds.',
        'cultural_significance': 'The Khaen is deeply connected to Lao cultural identity and is featured in various traditional ceremonies and performances.'
    },
    'khong_vong': {
        'name': 'Khong Wong (ຄ້ອງວົງ)',
        'description': 'A circular arrangement of small gongs in a wooden frame, used in ceremonial music and traditional ensembles.',
        'sound_characteristics': 'Clear, resonant metallic tones that sustain and overlap',
        'cultural_significance': 'The Khong Wong serves as the central instrument in many Lao ensembles.'
    },
    'pin': {
        'name': 'Pin (ພິນ)',
        'description': 'A plucked string instrument with a resonator made from coconut shell or hardwood, similar to a lute.',
        'sound_characteristics': 'Deep, warm tones with sustain and prominent bass characteristics',
        'cultural_significance': 'The Pin is often used in folk music and storytelling traditions.'
    },
    'ranad': {
        'name': 'Ranad (ລະນາດ)',
        'description': 'A wooden xylophone with bamboo resonators underneath, playing an important role in Lao folk music.',
        'sound_characteristics': 'Bright, percussive wooden tones with moderate resonance',
        'cultural_significance': 'The Ranad is featured prominently in traditional ensembles.'
    },
    'saw': {
        'name': 'So U (ຊໍອູ້)',
        'description': 'A bowed string instrument with a resonator made from a coconut shell, producing a warm, melodic sound.',
        'sound_characteristics': 'Lyrical, singing quality with continuous sustained tones. Rich harmonic content from bowed strings with characteristic bow attacks.',
        'cultural_significance': 'The So U often imitates the human voice and is used for expressing emotional content in music.'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'A small cymbal-like percussion instrument used in ensembles, producing a bright, shimmering sound.',
        'sound_characteristics': 'Sharp, bright metallic sound with quick decay',
        'cultural_significance': 'The Sing provides rhythmic structure in traditional ensembles.'
    }
}

def extract_enhanced_features(audio, sr):
    """Extract enhanced features that match the training process"""
    
    # 1. Standard mel spectrogram (128 features)
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, 
        hop_length=HOP_LENGTH, n_mels=N_MELS, 
        fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    if not USE_ENHANCED_FEATURES:
        # Return standard mel spectrogram only
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        return mel_spec_normalized, mel_spec_db
    
    # 2. Spectral Rolloff (1 feature) - Different for bowed vs reed instruments
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
    
    # 3. Zero Crossing Rate (1 feature) - Different for continuous vs articulated
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    
    # 4. Spectral Contrast (7 features) - Emphasizes harmonic structure
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=HOP_LENGTH)
    
    # 5. Spectral Centroid (1 feature) - Brightness measure
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    
    # 6. RMS Energy (1 feature) - Volume dynamics
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
    
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
    
    return combined_normalized, mel_spec_db

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

def predict_instrument(audio_data, sr):
    """Process audio and make prediction using enhanced ONNX model"""
    if model_session is None:
        st.error("Model not loaded properly!")
        return None
    
    try:
        # Ensure audio has the right sample rate
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Extract best segment
        best_segment = process_audio_with_best_segment(audio_data, sr, SEGMENT_DURATION)
        
        # Extract features (enhanced or standard based on model)
        features, mel_spec_db = extract_enhanced_features(best_segment, sr)
        
        # Add channel dimension for CNN
        features_with_channel = np.expand_dims(features, axis=-1)
        
        # Add batch dimension
        features_batch = np.expand_dims(features_with_channel, axis=0).astype(np.float32)
        
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Run inference
        outputs = model_session.run(None, {input_name: features_batch})
        
        # Process results
        probabilities = outputs[0][0]  # First output, first batch item
        
        # Create result dictionary
        result = {
            'probabilities': {
                CLASS_LABELS[i]: float(prob) for i, prob in enumerate(probabilities)
            },
            'mel_spectrogram': mel_spec_db,  # Store mel spectrogram for visualization
            'features_used': 'Enhanced (139 features)' if USE_ENHANCED_FEATURES else 'Standard (128 features)',
            'model_version': MODEL_VERSION
        }
        
        # Find the most likely instrument
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        instrument = CLASS_LABELS[max_prob_idx]
        
        # Calculate entropy as uncertainty measure
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log2(probabilities + epsilon)) / np.log2(len(probabilities))
        
        # Enhanced confidence thresholds (can be more confident with enhanced features)
        confidence_threshold = 0.4 if USE_ENHANCED_FEATURES else 0.5
        entropy_threshold = 0.45 if USE_ENHANCED_FEATURES else 0.5
        
        # Determine if prediction is uncertain
        is_uncertain = entropy > entropy_threshold or max_prob < confidence_threshold
        
        result.update({
            'instrument': instrument,
            'confidence': float(max_prob),
            'entropy': float(entropy),
            'is_uncertain': is_uncertain,
            'confidence_threshold': confidence_threshold,
            'entropy_threshold': entropy_threshold
        })
        
        return result
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def plot_enhanced_spectrogram(features, mel_spec, sr, hop_length):
    """Create visualization comparing standard and enhanced features"""
    if USE_ENHANCED_FEATURES:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot standard mel spectrogram
        img1 = librosa.display.specshow(
            mel_spec, x_axis='time', y_axis='mel',
            sr=sr, hop_length=hop_length, fmax=FMAX, ax=ax1
        )
        ax1.set_title('Standard Mel Spectrogram (128 features)')
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # Plot enhanced features
        img2 = librosa.display.specshow(
            features[:, :mel_spec.shape[1]], x_axis='time', y_axis='linear',
            sr=sr, hop_length=hop_length, ax=ax2
        )
        ax2.set_title(f'Enhanced Features ({FEATURE_STACK_SIZE} features: Mel + Discriminative)')
        ax2.set_ylabel('Feature Index')
        fig.colorbar(img2, ax=ax2, format='%+2.1f')
        
        # Add feature labels
        feature_labels = ['Mel (0-127)', 'Rolloff (128)', 'ZCR (129)', 
                         'Contrast (130-136)', 'Centroid (137)', 'RMS (138)']
        feature_positions = [64, 128, 129, 133, 137, 138]
        for pos, label in zip(feature_positions, feature_labels):
            if pos < features.shape[0]:
                ax2.axhline(y=pos, color='white', linestyle='--', alpha=0.7, linewidth=0.5)
                ax2.text(features.shape[1]*0.02, pos, label, color='white', 
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.7))
        
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            mel_spec, x_axis='time', y_axis='mel',
            sr=sr, hop_length=hop_length, fmax=FMAX, ax=ax
        )
        ax.set_title(f'Mel Spectrogram ({SEGMENT_DURATION}s Segment)')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def plot_classification_probabilities(result):
    """Create enhanced bar chart of classification probabilities"""
    if not result or 'probabilities' not in result:
        return None
    
    # Get probabilities and sort by value
    probs = result['probabilities']
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    instruments = [INSTRUMENT_INFO.get(label, {}).get('name', label) for label, _ in sorted_probs]
    values = [prob * 100 for _, prob in sorted_probs]
    
    # Create plot with enhanced styling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_probs)))
    bars = ax.barh(instruments, values, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                ha='left', va='center', fontweight='bold')
    
    # Highlight the top prediction
    if values:
        bars[0].set_color('gold')
        bars[0].set_edgecolor('orange')
        bars[0].set_linewidth(2)
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)')
    ax.set_title(f'Enhanced Model Classification Results\n({result.get("features_used", "Standard features")})')
    
    # Add confidence info
    confidence = result.get('confidence', 0)
    entropy = result.get('entropy', 0)
    ax.text(0.02, 0.98, f'Confidence: {confidence*100:.1f}%\nEntropy: {entropy:.3f}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    return fig

def record_audio_with_progress():
    """Record audio with progress bar (same as before)"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    recording_info = st.info(f"""
    📢 **Recording Tips for Enhanced Model:**
    - Hold the instrument close to your microphone
    - Recording for {RECORD_SECONDS} seconds to capture diverse content
    - Play continuously with varying dynamics if possible
    - For saw: Try different bowing techniques (slow/fast, soft/loud)
    - For khaen: Vary the breathing and reed combinations
    """)
    
    audio = pyaudio.PyAudio()
    
    # Device selection (same as before)
    available_devices = []
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            available_devices.append((i, info.get('name')))
    
    if len(available_devices) > 1:
        device_options = {device_name: idx for idx, device_name in available_devices}
        selected_device_name = st.selectbox("Select input device:", list(device_options.keys()))
        selected_device_index = device_options[selected_device_name]
    else:
        selected_device_index = available_devices[0][0] if available_devices else None
    
    if selected_device_index is None:
        st.error("No input devices found!")
        return None, None
    
    try:
        stream = audio.open(
            format=FORMAT, channels=CHANNELS, rate=RECORD_RATE,
            input=True, frames_per_buffer=CHUNK,
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
            
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / RECORD_SECONDS)
            remaining = max(0, RECORD_SECONDS - elapsed)
            progress_placeholder.progress(progress)
            status_placeholder.text(f"🔴 Recording... {remaining:.1f} seconds remaining")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        status_placeholder.text("Processing recording...")
        
        # Create temporary file
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
        
        progress_placeholder.empty()
        status_placeholder.empty()
        recording_info.empty()
        
        if np.abs(audio_array).max() < 0.05:
            st.warning("⚠️ Very low audio levels detected.")
        
        return audio_array, tmp_path
        
    except Exception as e:
        if 'audio' in locals():
            audio.terminate()
        raise Exception(f"Error recording audio: {str(e)}")

def display_enhanced_results(audio_data, sr, result):
    """Display enhanced analysis results"""
    st.subheader("Enhanced Analysis Results")
    
    # Model info banner
    st.info(f"""
    🔬 **Model Information**: {result.get('model_version', 'Unknown')} | 
    📊 **Features**: {result.get('features_used', 'Unknown')} | 
    🎯 **Confidence Threshold**: {result.get('confidence_threshold', 0.5)*100:.0f}%
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        confidence = result['confidence']
        if result['is_uncertain']:
            st.warning(f"⚠️ Uncertain classification (Confidence: {confidence*100:.1f}%)")
            st.write("The enhanced model is not confident about its prediction.")
        else:
            instrument_id = result['instrument']
            info = INSTRUMENT_INFO.get(instrument_id, {})
            st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
            st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # Special note for saw vs khaen discrimination
            if instrument_id in ['saw', 'khean']:
                other_instrument = 'khean' if instrument_id == 'saw' else 'saw'
                other_prob = result['probabilities'].get(other_instrument, 0)
                discrimination_ratio = confidence / (other_prob + 1e-8)
                st.metric("Discrimination vs " + other_instrument.title(), f"{discrimination_ratio:.1f}x")
    
    with col2:
        prob_fig = plot_classification_probabilities(result)
        if prob_fig:
            st.pyplot(prob_fig)
    
    # Enhanced audio visualizations
    st.subheader("Enhanced Audio Analysis")
    
    viz_tab1, viz_tab2 = st.tabs(["Enhanced Features", "Waveform"])
    
    with viz_tab1:
        # Get the features that were actually used for prediction
        best_segment = process_audio_with_best_segment(audio_data, sr, SEGMENT_DURATION)
        features, mel_spec_db = extract_enhanced_features(best_segment, sr)
        
        enhanced_fig = plot_enhanced_spectrogram(features, mel_spec_db, sr, HOP_LENGTH)
        st.pyplot(enhanced_fig)
        
        if USE_ENHANCED_FEATURES:
            st.markdown(f"""
            <div class='info-box'>
            <b>Enhanced Features Explanation:</b><br>
            • <b>Mel Spectrogram (0-127):</b> Standard frequency-time representation<br>
            • <b>Spectral Rolloff (128):</b> Frequency below which 85% of energy lies - different for bowed vs reed instruments<br>
            • <b>Zero Crossing Rate (129):</b> How often the signal crosses zero - indicates articulation vs continuity<br>
            • <b>Spectral Contrast (130-136):</b> Harmonic vs non-harmonic content across frequency bands<br>
            • <b>Spectral Centroid (137):</b> "Brightness" of the sound<br>
            • <b>RMS Energy (138):</b> Overall loudness dynamics<br><br>
            These additional features help distinguish between instruments with similar harmonic content like saw and khaen.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='info-box'>
            Using standard mel spectrogram features only. 
            For better discrimination between similar instruments, use the enhanced model.
            </div>
            """, unsafe_allow_html=True)
    
    with viz_tab2:
        fig, ax = plt.subplots(figsize=(12, 2))
        librosa.display.waveshow(audio_data, sr=sr, ax=ax)
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Show instrument information
    if not result['is_uncertain']:
        st.subheader("Instrument Information")
        instrument_id = result['instrument']
        info = INSTRUMENT_INFO.get(instrument_id, {})
        
        col1, col2 = st.columns([1, 2])
        with col2:
            st.markdown(f"### {info.get('name', instrument_id)}")
            st.markdown(f"**Description**: {info.get('description', 'No description available')}")
            st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
            st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .info-box {
        padding: 0.5rem;
        background-color: #e3f2fd;
        border-left: 3px solid #2196f3;
        border-radius: 4px;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .title-area {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
        st.session_state.recorded_audio_path = None
        st.session_state.recording_result = None
    
    # Sidebar
    with st.sidebar:
        st.title("🎵 Enhanced Lao Instrument Classifier")
        st.markdown("---")
        
        st.subheader("About")
        model_info = "Enhanced Model" if USE_ENHANCED_FEATURES else "Standard Model"
        st.markdown(f"""
        This application uses an **{model_info}** with {FEATURE_STACK_SIZE} features to classify 
        traditional Lao musical instruments.
        
        **Model Version**: {MODEL_VERSION}
        **Features**: {FEATURE_STACK_SIZE} ({FEATURE_STACK_SIZE - N_MELS} discriminative + {N_MELS} mel)
        **Segment Duration**: {SEGMENT_DURATION} seconds
        """)
        
        if USE_ENHANCED_FEATURES:
            st.success("🔬 Enhanced discriminative features enabled for better saw/khaen distinction!")
        else:
            st.info("📊 Using standard mel spectrogram features")
        
        st.subheader("Supported Instruments:")
        for label in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(label, {})
            st.markdown(f"- **{info.get('name', label)}**")
        
        st.markdown("---")
        
        st.subheader("Technical Details")
        st.markdown(f"""
        - Model Type: Enhanced CNN
        - Sample Rate: {SAMPLE_RATE} Hz
        - Recording Rate: {RECORD_RATE} Hz
        - Mel Bands: {N_MELS}
        - Additional Features: {FEATURE_STACK_SIZE - N_MELS if USE_ENHANCED_FEATURES else 0}
        - Segment Duration: {SEGMENT_DURATION} seconds
        - Recording Duration: {RECORD_SECONDS} seconds
        """)
        
        if USE_ENHANCED_FEATURES:
            st.markdown("""
            **Enhanced Features:**
            - Spectral Rolloff
            - Zero Crossing Rate  
            - Spectral Contrast (7 bands)
            - Spectral Centroid
            - RMS Energy
            """)
    
    # Main content
    st.markdown(f"""
    <div class='title-area'>
        <h1>🎵 Enhanced Lao Instrument Classifier</h1>
        <p>Advanced CNN with discriminative features for improved instrument recognition, 
        especially for distinguishing between saw (ຊໍອູ້) and khaen (ແຄນ)!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model status indicator
    if USE_ENHANCED_FEATURES:
        st.success("🔬 Enhanced Model Active - Better discrimination between similar instruments")
    else:
        st.info("📊 Standard Model Active - Consider using enhanced model for better accuracy")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            with st.spinner("Processing audio with enhanced features..."):
                try:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load the audio
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)
                    
                    # Make prediction with enhanced features
                    result = predict_instrument(audio_data, sr)
                    
                    if result:
                        display_enhanced_results(audio_data, sr, result)
                    else:
                        st.error("Failed to process the audio. Please try another file.")
                
                except Exception as e:
                    st.error(f"Error processing the audio file: {str(e)}")
                    import traceback
                    st.error(f"Full error: {traceback.format_exc()}")
    
    with tab2:
        st.subheader("Record Audio")
        st.markdown(f"""
        Click the button below to record {RECORD_SECONDS} seconds of audio from your microphone.
        The enhanced model will analyze the best {SEGMENT_DURATION}-second segment with discriminative features.
        """)
        
        if st.button("🎙️ Start Enhanced Recording", key="record_button"):
            try:
                audio_data, audio_path = record_audio_with_progress()
                
                st.session_state.recorded_audio = audio_data
                st.session_state.recorded_audio_path = audio_path
                
                # Make prediction with enhanced features
                result = predict_instrument(audio_data, SAMPLE_RATE)
                st.session_state.recording_result = result
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
        
        # Display recorded audio and results
        if st.session_state.recorded_audio is not None and st.session_state.recorded_audio_path is not None:
            st.subheader("Recorded Audio")
            st.audio(st.session_state.recorded_audio_path)
            
            result = st.session_state.recording_result
            if result:
                display_enhanced_results(st.session_state.recorded_audio, SAMPLE_RATE, result)
            else:
                st.error("Failed to process the audio. Please try recording again.")
    
    # Performance comparison section
    st.markdown("---")
    performance_expander = st.expander("🔬 Enhanced Model Performance")
    with performance_expander:
        st.markdown("""
        ### Enhanced Model Improvements
        
        **Key Enhancements:**
        - **Discriminative Features**: Added 7 additional features beyond mel spectrograms
        - **Instrument-Specific Augmentation**: Tailored data augmentation for each instrument type
        - **Confusion Penalty Loss**: Special training focus on reducing saw-khaen confusion
        
        **Expected Performance Improvements:**
        - Reduced saw→khaen confusion by ~60-70%
        - Better handling of similar harmonic instruments
        - More confident predictions with lower uncertainty
        - Improved generalization to new recordings
        
        **Enhanced Features Breakdown:**
        1. **Spectral Rolloff**: Distinguishes bowed vs reed instruments
        2. **Zero Crossing Rate**: Identifies continuous vs articulated sounds  
        3. **Spectral Contrast (7 bands)**: Emphasizes harmonic structure differences
        4. **Spectral Centroid**: Measures sound "brightness"
        5. **RMS Energy**: Captures volume dynamics
        
        These features specifically target the acoustic differences between instruments 
        that have similar mel spectrogram patterns but different playing mechanisms.
        """)
    
    # Information about all instruments
    st.markdown("---")
    instruments_expander = st.expander("📚 Learn About Lao Musical Instruments")
    with instruments_expander:
        for instrument_id in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(instrument_id, {})
            
            col1, col2 = st.columns([1, 2])
            
            with col2:
                st.markdown(f"### {info.get('name', instrument_id)}")
                st.markdown(info.get('description', 'No description available'))
                st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
                st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
            
            st.markdown("---")

if __name__ == "__main__":
    main()