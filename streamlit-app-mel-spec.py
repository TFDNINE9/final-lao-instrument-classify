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
    page_title="Lao Instrument Classifier (6-Second)",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model configuration and parameters
@st.cache_resource
def load_model_config(model_dir='models/mel_cnn_model_6sec'):  # Updated path
    """Load model configuration and parameters"""
    model_path = os.path.join(model_dir, 'mel_cnn_model_6sec.onnx')  # Updated filename
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
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

# Load model and metadata
model_session, model_metadata = load_model_config()

# Extract parameters from loaded config
if model_metadata is not None:
    SAMPLE_RATE = model_metadata.get('sample_rate', 44100)
    N_MELS = model_metadata.get('n_mels', 128)
    N_FFT = model_metadata.get('n_fft', 2048)
    HOP_LENGTH = model_metadata.get('hop_length', 512)
    SEGMENT_DURATION = model_metadata.get('segment_duration', 6.0)  # Now 6.0
    FMAX = model_metadata.get('fmax', 8000)
    CLASS_LABELS = model_metadata.get('class_names', [])
else:
    # Default parameters if loading failed
    SAMPLE_RATE = 44100
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    SEGMENT_DURATION = 6.0  # Now 6.0
    FMAX = 8000
    CLASS_LABELS = ['background', 'khean', 'khong_vong', 'pin', 'ranad', 'saw', 'sing']

# Recording configuration - UPDATED FOR LONGER DURATION
RECORD_SECONDS = 12  # Increased from 10 to 12 to ensure we capture enough audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_RATE = 44100  # Fixed recording rate for consistency across devices

# Instrument information and descriptions
INSTRUMENT_INFO = {
    'khean': {
        'name': 'Khaen (ແຄນ)',
        'description': 'A mouth organ made of bamboo pipes, each with a metal reed. It is considered the symbol of Lao music and was recognized by UNESCO as part of Lao\'s intangible cultural heritage in 2017.',
        'image': 'assets/khean.jpg',
        'sound_characteristics': 'Continuous, buzzing sound with the ability to play multiple notes simultaneously',
        'cultural_significance': 'The Khaen is deeply connected to Lao cultural identity and is featured in various traditional ceremonies and performances.'
    },
    'khong_vong': {
        'name': 'Khong Wong (ຄ້ອງວົງ)',
        'description': 'A circular arrangement of small gongs in a wooden frame, used in ceremonial music and traditional ensembles.',
        'image': 'assets/khong_vong.jpg',
        'sound_characteristics': 'Clear, resonant metallic tones that sustain and overlap',
        'cultural_significance': 'The Khong Wong serves as the central instrument in many Lao ensembles.'
    },
    'pin': {
        'name': 'Pin (ພິນ)',
        'description': 'A plucked string instrument with a resonator made from coconut shell or hardwood, similar to a lute.',
        'image': 'assets/pin.jpg',
        'sound_characteristics': 'Deep, warm tones with sustain and prominent bass characteristics',
        'cultural_significance': 'The Pin is often used in folk music and storytelling traditions.'
    },
    'ranad': {
        'name': 'Ranad (ລະນາດ)',
        'description': 'A wooden xylophone with bamboo resonators underneath, playing an important role in Lao folk music.',
        'image': 'assets/ranad.jpg',
        'sound_characteristics': 'Bright, percussive wooden tones with moderate resonance',
        'cultural_significance': 'The Ranad is featured prominently in traditional ensembles.'
    },
    'saw': {
        'name': 'So U (ຊໍອູ້)',
        'description': 'A bowed string instrument with a resonator made from a coconut shell, producing a warm, melodic sound.',
        'image': 'assets/saw.jpg',
        'sound_characteristics': 'Lyrical, singing quality with continuous sustained tones',
        'cultural_significance': 'The So U often imitates the human voice and is used for expressing emotional content in music.'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'A small cymbal-like percussion instrument used in ensembles, producing a bright, shimmering sound.',
        'image': 'assets/sing.jpg',
        'sound_characteristics': 'Sharp, bright metallic sound with quick decay',
        'cultural_significance': 'The Sing provides rhythmic structure in traditional ensembles.'
    },
    'background': {
        'name': 'Background Noise',
        'description': 'Ambient sounds or noise that does not contain a Lao musical instrument.',
        'image': 'assets/background.jpg',
        'sound_characteristics': 'Varied and non-musical',
        'cultural_significance': 'Not applicable'
    }
}

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio based on energy and spectral content"""
    # Calculate segment length in samples
    segment_len = int(segment_duration * sr)
    
    # If audio is shorter than segment duration, just pad
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Create segments with 50% overlap
    hop_len = int(segment_len / 2)
    n_hops = max(1, int((len(audio) - segment_len) / hop_len) + 1)
    segments = []
    segment_starts = []
    
    for i in range(n_hops):
        start = i * hop_len
        end = min(start + segment_len, len(audio))
        if end - start < segment_len * 0.8:  # Skip too short segments
            continue
        segments.append(audio[start:end])
        segment_starts.append(start)
    
    if not segments:  # Just in case no valid segments found
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant')
    
    # Calculate metrics for each segment
    metrics = []
    for segment in segments:
        # Energy (RMS)
        rms = np.sqrt(np.mean(segment**2))
        
        # Spectral contrast (harmonic-to-noise ratio)
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Spectral flux (tonal variation)
        stft = np.abs(librosa.stft(segment))
        if stft.shape[1] > 1:  # Make sure we have at least 2 frames
            flux = np.mean(np.diff(stft, axis=1)**2)
        else:
            flux = 0
        
        # Combined score (weighted sum of metrics)
        score = rms + 0.3 * contrast + 0.2 * flux
        metrics.append(score)
    
    # Find the best segment
    best_idx = np.argmax(metrics)
    return segments[best_idx]

def extract_mel_spectrogram(audio, sr):
    """Extract mel spectrogram features with intelligent segment selection"""
    # Find the best segment
    best_segment = process_audio_with_best_segment(audio, sr, segment_duration=SEGMENT_DURATION)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=best_segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=FMAX
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
    
    return mel_spec_normalized, mel_spec_db

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
        
        # Extract mel spectrogram features with intelligent segment selection
        mel_spec_normalized, mel_spec_db = extract_mel_spectrogram(audio_data, sr)
        
        # Add channel dimension for CNN
        mel_spec_with_channel = np.expand_dims(mel_spec_normalized, axis=-1)
        
        # Add batch dimension
        features_batch = np.expand_dims(mel_spec_with_channel, axis=0).astype(np.float32)
        
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
            'mel_spectrogram': mel_spec_db  # Store for visualization
        }
        
        # Find the most likely instrument
        max_prob_idx = np.argmax(probabilities)
        max_prob = probabilities[max_prob_idx]
        instrument = CLASS_LABELS[max_prob_idx]
        
        # Calculate entropy as uncertainty measure
        epsilon = 1e-10
        entropy = -np.sum(probabilities * np.log2(probabilities + epsilon)) / np.log2(len(probabilities))
        
        # Determine if prediction is uncertain (high entropy or low confidence)
        is_uncertain = entropy > 0.5 or max_prob < 0.5
        is_background = instrument == 'background' and max_prob > 0.5
        
        result.update({
            'instrument': instrument,
            'confidence': float(max_prob),
            'entropy': float(entropy),
            'is_uncertain': is_uncertain,
            'is_background': is_background
        })
        
        return result
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None

def plot_mel_spectrogram(mel_spec, sr, hop_length):
    """Create visualization of the mel spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))  # Wider figure for 6-second spectrograms
    
    # Plot mel spectrogram
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
    fig, ax = plt.subplots(figsize=(12, 2))
    
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
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create bars with better color gradient based on probability
    colors = sns.color_palette("YlOrRd", n_colors=len(sorted_probs))
    bars = ax.barh(instruments, values, color=colors)
    
    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{values[i]:.1f}%',
                ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 105)  # Give a little extra space for the text
    ax.set_xlabel('Probability (%)')
    ax.set_title('Instrument Classification Results')
    
    plt.tight_layout()
    return fig

def display_instrument_info(instrument_id, confidence):
    """Display information about a specific instrument"""
    if instrument_id not in INSTRUMENT_INFO:
        st.warning(f"Information not available for instrument: {instrument_id}")
        return
    
    info = INSTRUMENT_INFO[instrument_id]
    
    st.markdown(f"<div class='instrument-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Try to load an image if available
        try:
            if os.path.exists(info['image']):
                st.image(info['image'], caption=info['name'], width=300)
            else:
                st.write(f"## {info['name']}")
        except:
            st.write(f"## {info['name']}")
    
    with col2:
        st.markdown(f"## {info['name']}")
        st.markdown(f"**Confidence**: {confidence*100:.1f}%")
        st.markdown(f"**Description**: {info['description']}")
        st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
        st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def list_audio_devices():
    """List available audio input devices"""
    try:
        audio = pyaudio.PyAudio()
        device_info = []
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:  # This is an input device
                device_info.append(f"Device {i}: {info.get('name')}")
                device_info.append(f"  Sample Rate: {int(info.get('defaultSampleRate'))} Hz")
                device_info.append(f"  Channels: {info.get('maxInputChannels')}")
                device_info.append("")
        
        audio.terminate()
        return device_info
    except Exception as e:
        return [f"Error getting audio devices: {e}"]

def record_audio_with_progress():
    """Record audio with a progress bar"""
    # Create placeholders for the recording UI
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Show recording instructions
    recording_info = st.info(f"""
    📢 **Recording Tips:**
    - Hold the instrument close to your microphone
    - Recording for {RECORD_SECONDS} seconds to ensure we capture enough audio
    - Ensure the environment is relatively quiet
    - Play the instrument continuously during recording
    - Avoid noise cancellation or "voice enhancement" features on your device if possible
    """)
    
    # Create PyAudio instance
    audio = pyaudio.PyAudio()
    
    # List available input devices
    available_devices = []
    default_device_index = None
    
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:  # This is an input device
            available_devices.append((i, info.get('name')))
            if default_device_index is None:
                default_device_index = i
    
    # Display available devices and allow selection
    if len(available_devices) > 1:
        device_options = {device_name: idx for idx, device_name in available_devices}
        selected_device_name = st.selectbox("Select input device:", list(device_options.keys()))
        selected_device_index = device_options[selected_device_name]
    elif len(available_devices) == 1:
        selected_device_index = available_devices[0][0]
        st.write(f"Using audio input device: {available_devices[0][1]}")
    else:
        st.error("No input devices found! Please check your microphone connection.")
        return None, None
    
    try:
        # Open stream with specific device and fixed sample rate
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RECORD_RATE,  # Use fixed recording rate
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=selected_device_index
        )
        
        # Initialize variables
        frames = []
        start_time = time.time()
        
        # Record audio for specified duration with progress bar
        status_placeholder.text("🔴 Recording...")
        
        for i in range(0, int(RECORD_RATE / CHUNK * RECORD_SECONDS)):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                st.error(f"Error reading from microphone: {e}")
                break
            
            # Update progress bar
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / RECORD_SECONDS)
            remaining = max(0, RECORD_SECONDS - elapsed)
            progress_placeholder.progress(progress)
            status_placeholder.text(f"🔴 Recording... {remaining:.1f} seconds remaining")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Update status
        status_placeholder.text("Processing recording...")
        
        # Combine all frames into one buffer
        audio_data = b''.join(frames)
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            # Write WAV file with our fixed recording rate
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RECORD_RATE)
                wf.writeframes(audio_data)
            
        # Re-read the file using librosa, ensuring proper resampling if needed
        audio_array, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
        
        # Clear placeholders
        progress_placeholder.empty()
        status_placeholder.empty()
        recording_info.empty()
        
        # Let the user know if we're detecting very low audio levels
        if np.abs(audio_array).max() < 0.05:
            st.warning("⚠️ Very low audio levels detected. The recording may be too quiet or the microphone might not be capturing audio properly.")
        
        return audio_array, tmp_path
        
    except Exception as e:
        if 'audio' in locals():
            audio.terminate()
        status_placeholder.empty()
        progress_placeholder.empty()
        recording_info.empty()
        raise Exception(f"Error recording audio: {str(e)}")

def display_results(audio_data, sr, result):
    """Display analysis results in a structured format"""
    # Show result visualization
    st.subheader("Analysis Results")
    
    # Display prediction result
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if result['is_uncertain']:
            st.warning(f"⚠️ Uncertain classification (Confidence: {result['confidence']*100:.1f}%)")
            st.write("The model is not confident about its prediction. This might not be one of the supported Lao instruments.")
        elif result['is_background']:
            st.info("🔊 Background noise detected")
            st.write("No clear Lao instrument detected in this recording.")
        else:
            instrument_id = result['instrument']
            info = INSTRUMENT_INFO.get(instrument_id, {})
            st.success(f"✅ Detected: **{info.get('name', instrument_id)}**")
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
    
    with col2:
        # Plot probabilities
        prob_fig = plot_classification_probabilities(result)
        if prob_fig:
            st.pyplot(prob_fig)
    
    # Audio visualizations
    st.subheader("Audio Visualization")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2 = st.tabs(["Mel Spectrogram", "Waveform"])
    
    with viz_tab1:
        mel_fig = plot_mel_spectrogram(result['mel_spectrogram'], sr, HOP_LENGTH)
        st.pyplot(mel_fig)
        st.markdown(f"""
        <div class='info-box'>
        The Mel Spectrogram shows the frequency content of the audio over time. 
        This visualization represents the best {SEGMENT_DURATION}-second segment from your recording.
        Different instruments have characteristic patterns in their spectrograms.
        </div>
        """, unsafe_allow_html=True)
    
    with viz_tab2:
        waveform_fig = plot_waveform(audio_data, sr)
        st.pyplot(waveform_fig)
        st.markdown("""
        <div class='info-box'>
        The waveform shows the amplitude (loudness) of the audio over time.
        </div>
        """, unsafe_allow_html=True)
    
    # Show instrument information if a specific instrument was detected with good confidence
    if not result['is_uncertain'] and not result['is_background']:
        st.subheader("Instrument Information")
        display_instrument_info(result['instrument'], result['confidence'])

def main():
    # Initialize session state for recording
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
        st.session_state.recorded_audio_path = None
        st.session_state.recording_result = None
    
    # Sidebar
    with st.sidebar:
        st.title("Lao Instrument Classifier")
        st.markdown("---")
        
        st.subheader("About")
        st.markdown(f"""
        This application classifies traditional Lao musical instruments based on audio recordings
        using a Convolutional Neural Network (CNN) with Mel Spectrogram features.
        
        Upload a WAV file or record audio to identify the instrument being played.
        
        The model analyzes **{SEGMENT_DURATION}-second** segments of audio, using intelligent
        segment selection to find the most musically relevant portion.
        """)
        
        # List instruments with nicer formatting
        st.subheader("Supported Instruments:")
        for label in [l for l in CLASS_LABELS if l != 'background']:
            info = INSTRUMENT_INFO.get(label, {})
            st.markdown(f"- **{info.get('name', label)}**")
        
        st.markdown("---")
        
        # Technical details without nested expanders
        st.subheader("Technical Details")
        st.markdown(f"""
        - Model Type: CNN with Mel Spectrogram
        - Sample Rate: {SAMPLE_RATE} Hz
        - Recording Rate: {RECORD_RATE} Hz
        - Mel Bands: {N_MELS}
        - Segment Duration: {SEGMENT_DURATION} seconds
        - Recording Duration: {RECORD_SECONDS} seconds
        """)
        
        # Audio devices as a separate section, not an expander
        show_devices = st.checkbox("Show Audio Devices")
        if show_devices:
            st.subheader("Audio Devices")
            device_info = list_audio_devices()
            st.text("\n".join(device_info))
    
    # Main content
    st.markdown(f"""
    <div class='title-area'>
        <h1>🎵 Lao Instrument Classifier ({SEGMENT_DURATION}s)</h1>
        <p>Upload an audio recording or record live audio of a Lao musical instrument, 
        and this app will identify which instrument is playing using a Convolutional Neural Network!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg"])
        
        if uploaded_file is not None:
            # Display and process the uploaded file
            st.audio(uploaded_file, format="audio/wav")
            
            # Load and process the audio
            with st.spinner("Processing audio..."):
                try:
                    # Save to a temporary file (workaround for librosa loading)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load the audio
                    audio_data, sr = librosa.load(tmp_path, sr=None)
                    os.remove(tmp_path)  # Clean up temp file
                    
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
        Make sure your microphone is enabled and working properly.
        """)
        
        # Record button
        if st.button("🎙️ Start Recording", key="record_button"):
            try:
                # Record audio with progress bar
                audio_data, audio_path = record_audio_with_progress()
                
                # Store recorded audio in session state
                st.session_state.recorded_audio = audio_data
                st.session_state.recorded_audio_path = audio_path
                
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
    instruments_expander = st.expander("Learn About Lao Musical Instruments")
    with instruments_expander:
        for instrument_id in [label for label in CLASS_LABELS if label != 'background']:
            info = INSTRUMENT_INFO.get(instrument_id, {})
            
            st.markdown(f"<div class='instrument-card'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                try:
                    if os.path.exists(info.get('image', '')):
                        st.image(info['image'], width=250)
                except:
                    pass
            
            with col2:
                st.markdown(f"### {info.get('name', instrument_id)}")
                st.markdown(info.get('description', 'No description available'))
                st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
                st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")

# Add some custom CSS for better styling
def local_css():
    st.markdown("""
    <style>
    .title-area {
        text-align: center;
        padding: 1rem 0 2rem 0;
    }
    .instrument-card {
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
    }
    </style>
    """, unsafe_allow_html=True)

# Apply custom CSS
local_css()

if __name__ == "__main__":
    main()