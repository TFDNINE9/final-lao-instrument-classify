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
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Lao Instrument Classifier - Memory Efficient",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model configuration and parameters
@st.cache_resource
def load_model_config(model_dir='models/onnx_model'):
    """Load ONNX model configuration and parameters"""
    model_path = os.path.join(model_dir, 'model.onnx')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    # Check if files exist
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.error("Please run the conversion script first to create ONNX model")
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
        st.success(f"✅ Model loaded successfully from {model_path}")
        return session, metadata
    except Exception as e:
        st.error(f"Error loading ONNX model: {str(e)}")
        return None, None

# Load model and metadata
model_session, model_metadata = load_model_config()

# Extract parameters from loaded config
if model_metadata is not None:
    SAMPLE_RATE = model_metadata.get('sample_rate', 44100)
    N_MELS = model_metadata.get('n_mels', 128)
    N_FFT = model_metadata.get('n_fft', 2048)
    HOP_LENGTH = model_metadata.get('hop_length', 512)
    SEGMENT_DURATION = model_metadata.get('segment_duration', 6.0)
    FMAX = model_metadata.get('fmax', 8000)
    CLASS_LABELS = model_metadata.get('class_names', [])
    USE_DELTA_FEATURES = model_metadata.get('use_delta_features', True)
    USE_HARMONIC_PERCUSSIVE = model_metadata.get('use_harmonic_percussive', True)
    EXPECTED_CHANNELS = model_metadata.get('expected_channels', 4)
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_DURATION = 6.0
    FMAX = 8000
    CLASS_LABELS = ['khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']
    USE_DELTA_FEATURES = True
    USE_HARMONIC_PERCUSSIVE = True
    EXPECTED_CHANNELS = 4

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
        'sound_characteristics': 'Continuous, buzzing sound with the ability to play multiple notes simultaneously',
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
        'sound_characteristics': 'Lyrical, singing quality with continuous sustained tones',
        'cultural_significance': 'The So U often imitates the human voice and is used for expressing emotional content in music.'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'A small cymbal-like percussion instrument used in ensembles, producing a bright, shimmering sound.',
        'sound_characteristics': 'Sharp, bright metallic sound with quick decay',
        'cultural_significance': 'The Sing provides rhythmic structure in traditional ensembles.'
    }
}

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio based on energy and spectral content"""
    segment_len = int(segment_duration * sr)
    
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # For inference, use energy-based selection (not random)
    hop_len = int(segment_len / 2)
    n_hops = max(1, int((len(audio) - segment_len) / hop_len) + 1)
    segments = []
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start >= segment_len * 0.8:
            segments.append(audio[start:end])
    
    if not segments:
        return audio[:segment_len]
    
    # Calculate energy for each segment
    energies = []
    for segment in segments:
        # Multiple criteria for best segment
        rms = np.sqrt(np.mean(segment**2))  # Energy
        
        # Spectral contrast
        try:
            contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        except:
            contrast = 0
        
        # Spectral flux
        try:
            stft = np.abs(librosa.stft(segment))
            if stft.shape[1] > 1:
                flux = np.mean(np.diff(stft, axis=1)**2)
            else:
                flux = 0
        except:
            flux = 0
        
        score = rms + 0.3 * contrast + 0.2 * flux
        energies.append(score)
    
    best_idx = np.argmax(energies)
    return segments[best_idx]

def extract_memory_efficient_features(audio, sr):
    """Extract the same features as training: mel + harmonic + percussive + delta"""
    
    # Process the best segment
    best_segment = process_audio_with_best_segment(audio, sr, segment_duration=SEGMENT_DURATION)
    
    features = []
    
    # Base mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=best_segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmax=FMAX
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.append(mel_spec_db)
    
    # Harmonic-percussive separation (if enabled)
    if USE_HARMONIC_PERCUSSIVE:
        try:
            harmonic, percussive = librosa.effects.hpss(best_segment)
            
            # Harmonic component
            mel_harmonic = librosa.feature.melspectrogram(
                y=harmonic, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmax=FMAX
            )
            features.append(librosa.power_to_db(mel_harmonic, ref=np.max))
            
            # Percussive component
            mel_percussive = librosa.feature.melspectrogram(
                y=percussive, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
                n_mels=N_MELS, fmax=FMAX
            )
            features.append(librosa.power_to_db(mel_percussive, ref=np.max))
        except Exception as e:
            st.warning(f"Harmonic-percussive separation failed: {e}")
    
    # Delta features (if enabled)
    if USE_DELTA_FEATURES:
        try:
            delta = librosa.feature.delta(mel_spec_db)
            features.append(delta)
        except Exception as e:
            st.warning(f"Delta features failed: {e}")
    
    # Stack features
    if len(features) > 1:
        combined_features = np.stack(features, axis=-1)
    else:
        combined_features = np.expand_dims(features[0], axis=-1)
    
    # Normalize each channel
    for i in range(combined_features.shape[-1]):
        channel = combined_features[:, :, i]
        mean_val = channel.mean()
        std_val = channel.std()
        if std_val > 1e-8:
            combined_features[:, :, i] = (channel - mean_val) / std_val
        else:
            combined_features[:, :, i] = channel - mean_val
    
    # Convert to float32
    combined_features = combined_features.astype(np.float32)
    
    return combined_features, mel_spec_db

def predict_instrument(audio_data, sr):
    """Process audio and make prediction using ONNX model"""
    if model_session is None:
        st.error("Model not loaded properly!")
        return None
    
    try:
        # Ensure audio has the right sample rate
        if sr != SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE
        
        # Extract features with same method as training
        features, mel_spec_db = extract_memory_efficient_features(audio_data, sr)
        
        # Add batch dimension
        features_batch = np.expand_dims(features, axis=0)
        
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
            'mel_spectrogram': mel_spec_db,
            'features_shape': features.shape
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
    fig, ax = plt.subplots(figsize=(12, 6))
    
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
    ax.set_title(f'Mel Spectrogram ({SEGMENT_DURATION}s Segment)')
    
    plt.tight_layout()
    return fig

def plot_waveform(audio, sr):
    """Create visualization of the audio waveform"""
    fig, ax = plt.subplots(figsize=(12, 3))
    
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title('Waveform')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    
    plt.tight_layout()
    return fig

def plot_classification_probabilities(result):
    """Create bar chart of classification probabilities"""
    if not result or 'probabilities' not in result:
        return None
    
    # Get probabilities and sort by value
    probs = result['probabilities']
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data for plotting
    instruments = [INSTRUMENT_INFO.get(label, {}).get('name', label) for label, _ in sorted_probs]
    values = [prob * 100 for _, prob in sorted_probs]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color bars based on confidence
    colors = ['#2E8B57' if val > 50 else '#4682B4' if val > 20 else '#D3D3D3' for val in values]
    bars = ax.barh(instruments, values, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)')
    ax.set_title('Instrument Classification Results')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def display_instrument_info(instrument_id, confidence):
    """Display information about a specific instrument"""
    if instrument_id not in INSTRUMENT_INFO:
        st.warning(f"Information not available for instrument: {instrument_id}")
        return
    
    info = INSTRUMENT_INFO[instrument_id]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"## {info['name']}")
        st.metric("Confidence", f"{confidence*100:.1f}%")
    
    with col2:
        st.markdown(f"**Description**: {info['description']}")
        st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
        st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")

def record_audio_with_progress():
    """Record audio with a progress bar"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    recording_info = st.info(f"""
    📢 **Recording Tips:**
    - Hold the instrument close to your microphone
    - Recording for {RECORD_SECONDS} seconds
    - Ensure the environment is relatively quiet
    - Play the instrument continuously during recording
    """)
    
    audio = pyaudio.PyAudio()
    
    try:
        # Use default input device
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECORD_RATE,
            input=True,
            frames_per_buffer=CHUNK
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
        
        # Combine frames
        audio_data = b''.join(frames)
        
        # Create temporary WAV file
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
            st.warning("⚠️ Very low audio levels detected. Recording may be too quiet.")
        
        return audio_array, tmp_path
        
    except Exception as e:
        if 'audio' in locals():
            audio.terminate()
        status_placeholder.empty()
        progress_placeholder.empty()
        recording_info.empty()
        raise Exception(f"Error recording audio: {str(e)}")

def display_results(audio_data, sr, result):
    """Display analysis results"""
    st.subheader("Analysis Results")
    
    # Main result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if result['is_uncertain']:
            st.warning(f"⚠️ Uncertain classification (Confidence: {result['confidence']*100:.1f}%)")
            st.write("The model is not confident about its prediction.")
        else:
            instrument_id = result['instrument']
            info = INSTRUMENT_INFO.get(instrument_id, {})
            st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
            
            # Show entropy
            st.info(f"Prediction Entropy: {result['entropy']:.3f}")
    
    with col2:
        # Plot probabilities
        prob_fig = plot_classification_probabilities(result)
        if prob_fig:
            st.pyplot(prob_fig)
    
    # Technical details
    with st.expander("Technical Details"):
        st.write(f"**Features Shape**: {result['features_shape']}")
        st.write(f"**Model Channels**: {EXPECTED_CHANNELS}")
        st.write(f"**Segment Duration**: {SEGMENT_DURATION}s")
        st.write(f"**Sample Rate**: {SAMPLE_RATE} Hz")
        st.write(f"**Features Used**: Mel-spectrogram")
        if USE_HARMONIC_PERCUSSIVE:
            st.write("+ Harmonic-Percussive Separation")
        if USE_DELTA_FEATURES:
            st.write("+ Delta Features")
    
    # Audio visualizations
    st.subheader("Audio Visualization")
    
    viz_tab1, viz_tab2 = st.tabs(["Mel Spectrogram", "Waveform"])
    
    with viz_tab1:
        mel_fig = plot_mel_spectrogram(result['mel_spectrogram'], sr, HOP_LENGTH)
        st.pyplot(mel_fig)
        st.info(f"This shows the frequency content of the best {SEGMENT_DURATION}-second segment from your audio.")
    
    with viz_tab2:
        waveform_fig = plot_waveform(audio_data, sr)
        st.pyplot(waveform_fig)
        st.info("This shows the amplitude (loudness) of the audio over time.")
    
    # Instrument information
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
        st.title("🎵 Lao Instrument Classifier")
        st.markdown("**Memory-Efficient CNN Model**")
        st.markdown("---")
        
        # Model info
        if model_metadata:
            st.subheader("Model Information")
            st.write(f"**Model Type**: {model_metadata.get('model_type', 'CNN')}")
            st.write(f"**Features**: {model_metadata.get('feature_type', 'Mel-spectrogram')}")
            st.write(f"**Accuracy**: {model_metadata.get('kfold_results', {}).get('mean_accuracy', 0)*100:.1f}%")
            st.write(f"**Channels**: {EXPECTED_CHANNELS}")
        
        st.markdown("---")
        
        st.subheader("Supported Instruments:")
        for label in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(label, {})
            st.markdown(f"- **{info.get('name', label)}**")
        
        st.markdown("---")
        
        st.subheader("Technical Details")
        st.markdown(f"""
        - Sample Rate: {SAMPLE_RATE} Hz
        - Segment Duration: {SEGMENT_DURATION}s
        - Recording Duration: {RECORD_SECONDS}s
        - Mel Bands: {N_MELS}
        - Features: Multi-channel
        """)
    
    # Main content
    st.markdown(f"""
    # 🎵 Lao Instrument Classifier
    ### Memory-Efficient CNN Model
    
    Upload an audio recording or record live audio of a Lao musical instrument, 
    and this app will identify which instrument is playing using a specialized CNN model 
    with multi-channel features (Mel-spectrogram + Harmonic-Percussive + Delta features).
    """)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "m4a", "flac"])
        
        if uploaded_file is not None:
            # Display and process the uploaded file
            st.audio(uploaded_file, format="audio/wav")
            
            # Load and process the audio
            with st.spinner("Processing audio..."):
                try:
                    # Save to a temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load the audio
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)  # Clean up
                    
                    # Show basic audio info
                    st.info(f"**Audio Info**: Duration: {len(audio_data)/sr:.2f}s, Sample Rate: {sr} Hz")
                    
                    # Make prediction
                    result = predict_instrument(audio_data, sr)
                    
                    if result:
                        display_results(audio_data, sr, result)
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
        The system will analyze the best {SEGMENT_DURATION}-second segment from your recording.
        """)
        
        # Record button
        if st.button("🎙️ Start Recording", key="record_button"):
            try:
                # Record audio with progress bar
                audio_data, audio_path = record_audio_with_progress()
                
                # Store recorded audio in session state
                st.session_state.recorded_audio = audio_data
                st.session_state.recorded_audio_path = audio_path
                
                # Show basic audio info
                st.info(f"**Recorded Audio**: Duration: {len(audio_data)/SAMPLE_RATE:.2f}s, Sample Rate: {SAMPLE_RATE} Hz")
                
                # Make prediction
                result = predict_instrument(audio_data, SAMPLE_RATE)
                st.session_state.recording_result = result
                
                # Force rerun to display results
                st.rerun()
                
            except Exception as e:
                st.error(f"Error recording audio: {str(e)}")
                import traceback
                st.error(f"Full error: {traceback.format_exc()}")
        
        # Display recorded audio and results if available
        if st.session_state.recorded_audio is not None and st.session_state.recorded_audio_path is not None:
            st.subheader("Recorded Audio")
            st.audio(st.session_state.recorded_audio_path)
            
            result = st.session_state.recording_result
            if result:
                display_results(st.session_state.recorded_audio, SAMPLE_RATE, result)
            else:
                st.error("Failed to process the audio. Please try recording again.")
    
    # Information about all instruments
    st.markdown("---")
    with st.expander("📚 Learn About Lao Musical Instruments"):
        for instrument_id in CLASS_LABELS:
            info = INSTRUMENT_INFO.get(instrument_id, {})
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown(f"### {info.get('name', instrument_id)}")
            
            with col2:
                st.markdown(f"**Description**: {info.get('description', 'No description available')}")
                st.markdown(f"**Sound**: {info.get('sound_characteristics', 'Not available')}")
                st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
            
            st.markdown("---")
    
    # Model performance info
    if model_metadata and 'kfold_results' in model_metadata:
        with st.expander("📊 Model Performance Details"):
            kfold_results = model_metadata['kfold_results']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Accuracy", f"{kfold_results['mean_accuracy']*100:.1f}%")
            
            with col2:
                st.metric("Standard Deviation", f"{kfold_results['std_accuracy']*100:.1f}%")
            
            with col3:
                st.metric("Best Fold", f"{max(kfold_results['fold_accuracies'])*100:.1f}%")
            
            st.markdown("**Fold Accuracies:**")
            for i, acc in enumerate(kfold_results['fold_accuracies']):
                st.write(f"Fold {i+1}: {acc*100:.1f}%")
            
            st.markdown("""
            **About this model:**
            - Trained with 5-fold cross-validation for robust evaluation
            - Uses multi-channel features: Mel-spectrogram + Harmonic-Percussive separation + Delta features
            - Memory-efficient architecture designed for better generalization
            - Realistic accuracy range indicates good learning without overfitting
            """)

# Add custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 0.5rem;
        background-color: #e9f7fe;
        border-left: 3px solid #0096ff;
        border-radius: 4px;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .instrument-info {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
local_css()

if __name__ == "__main__":
    # Check if model is loaded
    if model_session is None:
        st.error("❌ Model not loaded! Please run the conversion script first.")
        st.markdown("""
        ### Setup Instructions:
        
        1. **Convert your model to ONNX**:
           ```bash
           python convert_to_onnx.py
           ```
        
        2. **Install required packages**:
           ```bash
           pip install streamlit onnxruntime librosa matplotlib seaborn pyaudio soundfile
           ```
        
        3. **Run this Streamlit app**:
           ```bash
           streamlit run streamlit_app.py
           ```
        
        Make sure your `model.h5` file is in the correct location before conversion.
        """)
    else:
        main()