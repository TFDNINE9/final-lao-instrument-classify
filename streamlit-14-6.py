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
import io
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Enhanced Lao Instrument Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration class matching the training
class Config:
    # Audio parameters (must match training)
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Recording configuration
    RECORD_SECONDS = 8  # Slightly longer for better segment selection
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RECORD_RATE = 44100

# Enhanced instrument information
INSTRUMENT_INFO = {
    'khean': {
        'name': 'Khaen (ແຄນ)',
        'description': 'A traditional Lao mouth organ made of bamboo pipes with metal reeds. UNESCO recognized it as part of Lao\'s intangible cultural heritage in 2017.',
        'image': 'assets/khean.jpg',
        'sound_characteristics': 'Continuous, droning sound with harmonic overtones. Can play multiple notes simultaneously.',
        'cultural_significance': 'Central to Lao cultural identity, used in ceremonies, celebrations, and traditional performances.',
        'playing_technique': 'Blown through a wind chamber, with fingering techniques controlling pitch and harmony.',
        'typical_pitch_range': '80-400 Hz (fundamental)',
        'key_features': 'High harmonic content, sustained tones, low zero-crossing rate'
    },
    'khong_vong': {
        'name': 'Khong Wong (ຄ້ອງວົງ)',
        'description': 'A circular arrangement of small bronze gongs in a wooden frame, used in traditional Lao ensembles.',
        'image': 'assets/khong_vong.jpg',
        'sound_characteristics': 'Clear, bright metallic tones with sustain and natural decay patterns.',
        'cultural_significance': 'Central instrument in traditional Lao orchestras, often leading melodic lines.',
        'playing_technique': 'Struck with mallets, creating resonant metallic tones.',
        'typical_pitch_range': '200-2000 Hz',
        'key_features': 'Sharp attack transients, metallic timbre, moderate sustain'
    },
    'pin': {
        'name': 'Pin (ພິນ)',
        'description': 'A plucked string instrument with a coconut shell or wooden resonator, similar to a lute.',
        'image': 'assets/pin.jpg',
        'sound_characteristics': 'Warm, woody tones with distinct attack-decay patterns from plucking.',
        'cultural_significance': 'Used in folk music, storytelling, and solo performances.',
        'playing_technique': 'Strings are plucked with fingers or plectrum.',
        'typical_pitch_range': '80-800 Hz',
        'key_features': 'Sharp attack followed by exponential decay, woody resonance'
    },
    'ranad': {
        'name': 'Ranad (ລະນາດ)',
        'description': 'A wooden xylophone with bamboo resonators, producing bright percussive tones.',
        'image': 'assets/ranad.jpg',
        'sound_characteristics': 'Bright, percussive wooden tones with short sustain.',
        'cultural_significance': 'Featured in traditional ensembles and ceremonial music.',
        'playing_technique': 'Struck with mallets on wooden keys.',
        'typical_pitch_range': '200-2000 Hz',
        'key_features': 'Sharp attack, wooden timbre, short decay'
    },
    'saw': {
        'name': 'So U (ຊໍອູ້)',
        'description': 'A two-stringed bowed instrument with a coconut shell resonator.',
        'image': 'assets/saw.jpg',
        'sound_characteristics': 'Smooth, lyrical tones that can imitate the human voice.',
        'cultural_significance': 'Often used for expressive, emotional musical passages.',
        'playing_technique': 'Bowed like a violin, with sliding pitch techniques.',
        'typical_pitch_range': '150-1000 Hz',
        'key_features': 'Smooth sustain, expressive pitch bending, bow noise in high frequencies'
    },
    'sing': {
        'name': 'Sing (ຊິ່ງ)',
        'description': 'Small cymbals used for rhythmic accompaniment in traditional ensembles.',
        'image': 'assets/sing.jpg',
        'sound_characteristics': 'Bright, sharp metallic sounds with quick decay.',
        'cultural_significance': 'Provides rhythmic structure in ensemble performances.',
        'playing_technique': 'Struck together or with small mallets.',
        'typical_pitch_range': '1000-8000 Hz',
        'key_features': 'Very sharp attack, bright metallic timbre, fast decay'
    },
    'unknown': {
        'name': 'Unknown Sound',
        'description': 'Audio that does not match traditional Lao musical instruments, including speech, environmental sounds, or other non-instrumental audio.',
        'image': 'assets/unknown.jpg',
        'sound_characteristics': 'Varied characteristics that don\'t match instrumental patterns.',
        'cultural_significance': 'Not applicable - represents non-instrumental audio.',
        'playing_technique': 'Not applicable',
        'typical_pitch_range': 'Variable',
        'key_features': 'Irregular patterns, speech-like or environmental characteristics'
    }
}

def get_default_instrument_image():
    """Create a default placeholder image for missing instrument images"""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, 'No Image\nAvailable', 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16,
            color='gray')
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

class EnhancedInstrumentClassifier:
    """Enhanced classifier with ensemble prediction and confidence estimation"""
    
    def __init__(self, model_path='model/model.onnx', label_mapping_path='model/label_mapping.json'):
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.session = None
        self.idx_to_label = None
        self.model_loaded = False
        
    def load_model(self):
        """Load ONNX model and label mapping with error handling"""
        try:
            if not os.path.exists(self.model_path):
                return False, f"Model file not found: {self.model_path}"
                
            if not os.path.exists(self.label_mapping_path):
                return False, f"Label mapping file not found: {self.label_mapping_path}"
            
            # Load label mapping
            with open(self.label_mapping_path, 'r') as f:
                label_mapping = json.load(f)
            self.idx_to_label = {int(idx): label for label, idx in label_mapping.items()}
            
            # Create ONNX session
            self.session = ort.InferenceSession(self.model_path)
            self.model_loaded = True
            
            return True, "Model loaded successfully"
            
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    
    def fast_segment_selection(self, audio, sr, n_segments=3):
        """Fast segment selection for ensemble prediction"""
        segment_len = int(Config.SEGMENT_DURATION * sr)
        
        if len(audio) <= segment_len:
            return [np.pad(audio, (0, segment_len - len(audio)), mode='constant')]
        
        # Create overlapping segments
        hop_len = segment_len // 3  # 67% overlap
        segments = []
        scores = []
        
        for start in range(0, len(audio) - segment_len + 1, hop_len):
            segment = audio[start:start + segment_len]
            
            # Fast scoring based on energy and spectral characteristics
            try:
                rms = np.sqrt(np.mean(segment**2))
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr))
                score = rms * 0.7 + (spectral_centroid / 4000) * 0.3
            except:
                score = np.sqrt(np.mean(segment**2))  # Fallback to RMS
            
            segments.append(segment)
            scores.append(score)
        
        # Return top N segments
        if len(segments) <= n_segments:
            return segments
        
        top_indices = np.argsort(scores)[-n_segments:]
        return [segments[i] for i in top_indices]
    
    def extract_features(self, audio, sr):
        """Extract features matching training process"""
        # Fast segment selection
        best_segment = self.fast_segment_selection(audio, sr, n_segments=1)[0]
        
        # Optional: Light harmonic separation (matching training)
        # For maximum speed, this can be disabled by setting use_hpss=False
        use_hpss = True
        if use_hpss:
            try:
                harmonic, percussive = librosa.effects.hpss(best_segment, margin=(1.0, 2.0))
                enhanced_audio = harmonic * 0.9 + percussive * 0.1
            except:
                enhanced_audio = best_segment
        else:
            enhanced_audio = best_segment
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=enhanced_audio,
            sr=sr,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH,
            n_mels=Config.N_MELS,
            fmax=Config.FMAX
        )
        
        # Convert to dB and normalize
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        return mel_spec_normalized, mel_spec_db, enhanced_audio
    
    def ensemble_predict(self, audio, sr, confidence_threshold=0.4):
        """Make ensemble prediction with confidence estimation"""
        if not self.model_loaded:
            return None
        
        try:
            # Resample if needed
            if sr != Config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
                sr = Config.SAMPLE_RATE
            
            # Get multiple segments for ensemble
            segments = self.fast_segment_selection(audio, sr, n_segments=3)
            
            predictions = []
            confidences = []
            all_features = []
            
            # Predict on each segment
            for segment in segments:
                mel_spec, mel_spec_db, processed_audio = self.extract_features(segment, sr)
                
                # Prepare input for model
                features_batch = np.expand_dims(np.expand_dims(mel_spec, axis=-1), axis=0).astype(np.float32)
                
                # Run inference
                input_name = self.session.get_inputs()[0].name
                outputs = self.session.run(None, {input_name: features_batch})
                probabilities = outputs[0][0]
                
                predictions.append(probabilities)
                confidences.append(np.max(probabilities))
                all_features.append((mel_spec_db, processed_audio))
            
            # Ensemble prediction (confidence-weighted average)
            if len(predictions) > 1:
                weights = np.array(confidences) / (np.sum(confidences) + 1e-8)
                ensemble_probs = np.average(predictions, axis=0, weights=weights)
            else:
                ensemble_probs = predictions[0]
            
            # Final prediction
            max_prob_idx = np.argmax(ensemble_probs)
            max_prob = ensemble_probs[max_prob_idx]
            instrument = self.idx_to_label[max_prob_idx]
            
            # Calculate uncertainty metrics
            entropy = -np.sum(ensemble_probs * np.log2(ensemble_probs + 1e-10)) / np.log2(len(ensemble_probs))
            prediction_std = np.std([np.max(p) for p in predictions])
            is_uncertain = entropy > 0.6 or max_prob < confidence_threshold or prediction_std > 0.15
            
            # Use the best segment's features for visualization
            best_segment_idx = np.argmax(confidences)
            best_mel_spec, best_audio = all_features[best_segment_idx]
            
            return {
                'instrument': instrument,
                'confidence': float(max_prob),
                'entropy': float(entropy),
                'prediction_std': float(prediction_std),
                'is_uncertain': is_uncertain,
                'segments_used': len(predictions),
                'individual_confidences': confidences,
                'probabilities': {self.idx_to_label[i]: float(prob) for i, prob in enumerate(ensemble_probs)},
                'mel_spectrogram': best_mel_spec,
                'processed_audio': best_audio,
                'confidence_category': self._get_confidence_category(max_prob, entropy, prediction_std)
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None
    
    def _get_confidence_category(self, confidence, entropy, std):
        """Categorize prediction confidence"""
        if confidence > 0.8 and entropy < 0.4 and std < 0.1:
            return "Very High"
        elif confidence > 0.6 and entropy < 0.6 and std < 0.15:
            return "High"
        elif confidence > 0.4 and entropy < 0.8:
            return "Medium"
        else:
            return "Low"

# Initialize the classifier
@st.cache_resource
def load_classifier():
    classifier = EnhancedInstrumentClassifier()
    success, message = classifier.load_model()
    return classifier, success, message

def plot_mel_spectrogram(mel_spec, sr=Config.SAMPLE_RATE, hop_length=Config.HOP_LENGTH):
    """Create an enhanced mel spectrogram visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    img = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        hop_length=hop_length,
        fmax=Config.FMAX,
        ax=ax,
        cmap='viridis'
    )
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(f'Mel Spectrogram ({Config.SEGMENT_DURATION}s segment)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
    
    plt.tight_layout()
    return fig

def plot_waveform(audio, sr=Config.SAMPLE_RATE):
    """Create waveform visualization"""
    fig, ax = plt.subplots(figsize=(12, 3))
    
    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time_axis, audio, linewidth=0.8, color='steelblue')
    ax.set_title('Audio Waveform', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_confidence_visualization(result):
    """Create enhanced confidence visualization using Plotly"""
    # Prepare data
    instruments = list(result['probabilities'].keys())
    probabilities = [result['probabilities'][inst] * 100 for inst in instruments]
    
    # Create color scheme based on confidence
    colors = ['#FF6B6B' if inst == result['instrument'] else '#E8E8E8' for inst in instruments]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=instruments,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.1f}%' for p in probabilities],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': f'Prediction Results - {result["instrument"].title()} ({result["confidence"]:.1%})',
            'x': 0.5,
            'font': {'size': 16, 'color': '#2E4057'}
        },
        xaxis_title='Instrument',
        yaxis_title='Probability (%)',
        yaxis=dict(range=[0, 105]),
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig

def create_confidence_gauge(result):
    """Create a confidence gauge chart"""
    confidence_pct = result['confidence'] * 100
    category = result['confidence_category']
    
    # Color scheme based on confidence
    color_map = {
        'Very High': '#00CC66',
        'High': '#66CC00', 
        'Medium': '#FFCC00',
        'Low': '#FF6B6B'
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence_pct,
        title = {'text': f"Confidence Level: {category}"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color_map.get(category, '#888888')},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "lightgreen"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, template='plotly_white')
    return fig

def display_instrument_info(instrument_id, confidence):
    """Display comprehensive instrument information"""
    if instrument_id not in INSTRUMENT_INFO:
        st.warning(f"Information not available for: {instrument_id}")
        return
    
    info = INSTRUMENT_INFO[instrument_id]
    
    # Create columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Try to load instrument image
        try:
            if os.path.exists(info.get('image', '')):
                st.image(info['image'], caption=info['name'], width=300)
            else:
                st.image(get_default_instrument_image(), caption=info['name'], width=300)
        except:
            st.image(get_default_instrument_image(), caption=info['name'], width=300)
    
    with col2:
        st.markdown(f"## {info['name']}")
        
        # Confidence badge
        confidence_pct = confidence * 100
        if confidence_pct > 80:
            st.success(f"🎯 **High Confidence**: {confidence_pct:.1f}%")
        elif confidence_pct > 60:
            st.info(f"✅ **Good Confidence**: {confidence_pct:.1f}%")
        elif confidence_pct > 40:
            st.warning(f"⚠️ **Medium Confidence**: {confidence_pct:.1f}%")
        else:
            st.error(f"❌ **Low Confidence**: {confidence_pct:.1f}%")
        
        # Instrument details
        st.markdown(f"**Description**: {info['description']}")
        
        with st.expander("🎵 Sound Characteristics"):
            st.write(f"**Sound**: {info.get('sound_characteristics', 'Not available')}")
            st.write(f"**Pitch Range**: {info.get('typical_pitch_range', 'Not available')}")
            st.write(f"**Key Features**: {info.get('key_features', 'Not available')}")
        
        with st.expander("🎭 Cultural Information"):
            st.write(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
            st.write(f"**Playing Technique**: {info.get('playing_technique', 'Not available')}")

def record_audio_with_progress():
    """Record audio with enhanced progress tracking"""
    # Create placeholders
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Show recording info
    recording_info = st.info(f"""
    🎙️ **Recording Instructions:**
    - Recording for {Config.RECORD_SECONDS} seconds
    - Play the instrument clearly and continuously
    - Keep background noise to a minimum
    - The system will analyze the best segments automatically
    """)
    
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()
        
        # Find input devices
        input_devices = []
        default_device = None
        
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                input_devices.append((i, info.get('name')))
                if default_device is None:
                    default_device = i
        
        if not input_devices:
            st.error("No microphone found! Please check your audio setup.")
            return None, None
        
        # Device selection
        if len(input_devices) > 1:
            device_names = {name: idx for idx, name in input_devices}
            selected_name = st.selectbox("Select microphone:", list(device_names.keys()))
            selected_device = device_names[selected_name]
        else:
            selected_device = input_devices[0][0]
            st.write(f"Using microphone: {input_devices[0][1]}")
        
        # Open audio stream
        stream = audio.open(
            format=Config.FORMAT,
            channels=Config.CHANNELS,
            rate=Config.RECORD_RATE,
            input=True,
            frames_per_buffer=Config.CHUNK,
            input_device_index=selected_device
        )
        
        # Record audio
        frames = []
        start_time = time.time()
        
        status_placeholder.text("🔴 Recording...")
        
        for i in range(0, int(Config.RECORD_RATE / Config.CHUNK * Config.RECORD_SECONDS)):
            try:
                data = stream.read(Config.CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception as e:
                st.error(f"Recording error: {e}")
                break
            
            # Update progress
            elapsed = time.time() - start_time
            progress = min(1.0, elapsed / Config.RECORD_SECONDS)
            remaining = max(0, Config.RECORD_SECONDS - elapsed)
            
            progress_placeholder.progress(progress)
            status_placeholder.text(f"🔴 Recording... {remaining:.1f}s remaining")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        status_placeholder.text("✅ Processing recording...")
        
        # Process recorded audio
        audio_data = b''.join(frames)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(Config.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(Config.FORMAT))
                wf.setframerate(Config.RECORD_RATE)
                wf.writeframes(audio_data)
        
        # Load with librosa
        audio_array, sr = librosa.load(tmp_path, sr=Config.SAMPLE_RATE)
        
        # Clear UI elements
        progress_placeholder.empty()
        status_placeholder.empty()
        recording_info.empty()
        
        # Check audio level
        if np.abs(audio_array).max() < 0.02:
            st.warning("⚠️ Very low audio levels detected. Try recording closer to the microphone.")
        
        return audio_array, tmp_path
        
    except Exception as e:
        # Clean up
        if 'audio' in locals():
            audio.terminate()
        progress_placeholder.empty()
        status_placeholder.empty()
        recording_info.empty()
        
        st.error(f"Recording failed: {str(e)}")
        return None, None

def display_analysis_results(audio_data, sr, result):
    """Display comprehensive analysis results"""
    st.subheader("🎵 Analysis Results")
    
    # Main result display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Confidence visualization
        confidence_fig = create_confidence_visualization(result)
        st.plotly_chart(confidence_fig, use_container_width=True)
    
    with col2:
        # Confidence gauge
        gauge_fig = create_confidence_gauge(result)
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    # Uncertainty warning
    if result['is_uncertain']:
        st.warning(f"""
        ⚠️ **Uncertain Prediction** 
        - Confidence: {result['confidence']:.1%}
        - Entropy: {result['entropy']:.2f}
        - Prediction Std: {result['prediction_std']:.3f}
        
        The model is not confident about this prediction. Consider:
        - Recording in a quieter environment
        - Playing the instrument more clearly
        - Ensuring the instrument is the dominant sound
        """)
    else:
        st.success(f"✅ **Confident Prediction**: {result['instrument'].title()} ({result['confidence']:.1%})")
    
    # Detailed metrics
    with st.expander("📊 Detailed Analysis Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence", f"{result['confidence']:.1%}")
            st.metric("Segments Used", result['segments_used'])
        
        with col2:
            st.metric("Entropy", f"{result['entropy']:.3f}")
            st.metric("Prediction Std", f"{result['prediction_std']:.3f}")
        
        with col3:
            st.metric("Confidence Category", result['confidence_category'])
            individual_confs = result['individual_confidences']
            st.metric("Segment Consistency", f"{np.std(individual_confs):.3f}")
    
    # Audio visualizations
    st.subheader("🎼 Audio Analysis")
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Mel Spectrogram", "Waveform", "All Probabilities"])
    
    with viz_tab1:
        mel_fig = plot_mel_spectrogram(result['mel_spectrogram'])
        st.pyplot(mel_fig)
        st.info("""
        **Mel Spectrogram**: Shows frequency content over time. Different instruments have 
        characteristic patterns - look for consistent frequency bands, attack patterns, and harmonic structure.
        """)
    
    with viz_tab2:
        waveform_fig = plot_waveform(result['processed_audio'])
        st.pyplot(waveform_fig)
        st.info("""
        **Waveform**: Shows amplitude over time. Notice attack patterns, sustain characteristics, 
        and decay patterns that are unique to each instrument type.
        """)
    
    with viz_tab3:
        # Detailed probability table
        prob_df = pd.DataFrame([
            {"Instrument": INSTRUMENT_INFO[inst]['name'], 
             "Probability": f"{prob:.1%}", 
             "Confidence": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"}
            for inst, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(prob_df, use_container_width=True)
    
    # Instrument information
    st.subheader("🎎 Instrument Information")
    display_instrument_info(result['instrument'], result['confidence'])

def main():
    # Load the classifier
    classifier, model_loaded, load_message = load_classifier()
    
    # App header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1>🎵 Enhanced Lao Instrument Classifier</h1>
        <p style='font-size: 1.2em; color: #666;'>
            AI-powered recognition of traditional Lao musical instruments with ensemble prediction
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if model is loaded
    if not model_loaded:
        st.error(f"❌ Model Loading Error: {load_message}")
        st.info("""
        **Troubleshooting:**
        1. Make sure `model.onnx` and `label_mapping.json` are in the `model/` folder
        2. Check file permissions
        3. Verify the model was generated successfully during training
        """)
        return
    
    st.success(f"✅ {load_message}")
    
    # Sidebar
    with st.sidebar:
        st.title("🎼 About")
        st.markdown("""
        This enhanced classifier uses:
        - **Ensemble prediction** from multiple audio segments
        - **Confidence estimation** with uncertainty metrics
        - **Fast segment selection** for optimal performance
        - **Background music robustness**
        
        **Supported Instruments:**
        """)
        
        # List instruments in sidebar
        for label, info in INSTRUMENT_INFO.items():
            if label != 'unknown':
                st.markdown(f"- **{info['name']}**")
        
        st.markdown("---")
        
        st.subheader("📈 Model Performance")
        st.markdown("""
        **Overall Accuracy**: 82.5%
        
        **Top Performers**:
        - Sing (Cymbals): 97.4% F1
        - Khong Wong (Gongs): 85.3% F1
        - Ranad (Xylophone): 82.0% F1
        - Pin (Lute): 78.4% F1
        
        **Key Improvements**:
        - Better handling of background music
        - Reduced khean/pin confusion
        - Ensemble prediction for reliability
        """)
        
        st.markdown("---")
        
        st.subheader("⚙️ Technical Details")
        st.markdown(f"""
        - **Sample Rate**: {Config.SAMPLE_RATE} Hz
        - **Segment Duration**: {Config.SEGMENT_DURATION}s
        - **Mel Frequency Bands**: {Config.N_MELS}
        - **Ensemble Segments**: 3 per prediction
        - **Model Type**: CNN with Multi-scale Features
        """)
        
        # Performance tips
        st.subheader("💡 Tips for Best Results")
        st.markdown("""
        **For Recording**:
        - Play instrument clearly and continuously
        - Minimize background noise when possible
        - Record for the full duration
        - Hold instrument close to microphone
        
        **For File Upload**:
        - Use WAV format for best quality
        - Minimum 4 seconds duration
        - Clear, audible instrument sound
        
        **Understanding Results**:
        - High confidence (>80%): Very reliable
        - Medium confidence (40-80%): Generally good
        - Low confidence (<40%): Review audio quality
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎵 Classify Audio", "📁 Upload File", "📚 Learn About Instruments"])
    
    with tab1:
        st.subheader("🎙️ Record Audio")
        st.markdown("""
        Click the button below to record audio from your microphone. The system will automatically:
        1. Record for 8 seconds to capture enough audio
        2. Select the best segments where the instrument is most prominent  
        3. Make ensemble predictions for improved accuracy
        4. Provide confidence estimates and uncertainty analysis
        """)
        
        # Initialize session state for recording
        if 'recorded_audio' not in st.session_state:
            st.session_state.recorded_audio = None
            st.session_state.recorded_audio_path = None
            st.session_state.recording_result = None
        
        # Recording button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎙️ Start Recording", type="primary", use_container_width=True):
                with st.spinner("Initializing recording..."):
                    audio_data, audio_path = record_audio_with_progress()
                    
                    if audio_data is not None:
                        # Store in session state
                        st.session_state.recorded_audio = audio_data
                        st.session_state.recorded_audio_path = audio_path
                        
                        # Make prediction
                        with st.spinner("🤖 Analyzing audio with ensemble prediction..."):
                            result = classifier.ensemble_predict(audio_data, Config.SAMPLE_RATE)
                            st.session_state.recording_result = result
                        
                        st.rerun()
        
        # Display results if available
        if st.session_state.recorded_audio is not None and st.session_state.recording_result is not None:
            st.subheader("🎵 Recorded Audio")
            st.audio(st.session_state.recorded_audio_path)
            
            # Display analysis results
            display_analysis_results(
                st.session_state.recorded_audio, 
                Config.SAMPLE_RATE, 
                st.session_state.recording_result
            )
    
    with tab2:
        st.subheader("📁 Upload Audio File")
        st.markdown("""
        Upload an audio file containing a Lao musical instrument. Supported formats: WAV, MP3, OGG, M4A, FLAC
        
        **Best Results**: 
        - Clear recording of a single instrument
        - At least 4 seconds duration
        - Minimal background noise (though the model handles some background music)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=["wav", "mp3", "ogg", "m4a", "flac"],
            help="Upload a clear recording of a Lao musical instrument"
        )
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB",
                "File type": uploaded_file.type
            }
            
            col1, col2 = st.columns(2)
            with col1:
                for key, value in file_details.items():
                    st.metric(key, value)
            
            # Play uploaded audio
            st.audio(uploaded_file, format="audio/wav")
            
            # Process the file
            with st.spinner("🤖 Processing uploaded file..."):
                try:
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Load audio
                    audio_data, sr = librosa.load(tmp_path, sr=Config.SAMPLE_RATE)
                    
                    # Check duration
                    duration = len(audio_data) / sr
                    if duration < 2.0:
                        st.warning(f"⚠️ Audio is quite short ({duration:.1f}s). Longer recordings typically give better results.")
                    
                    # Make prediction
                    result = classifier.ensemble_predict(audio_data, sr)
                    
                    if result:
                        # Display results
                        display_analysis_results(audio_data, sr, result)
                    else:
                        st.error("❌ Failed to analyze the audio. Please try a different file.")
                    
                    # Clean up
                    os.remove(tmp_path)
                    
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.info("Try converting your file to WAV format or check if the file is corrupted.")
    
    with tab3:
        st.subheader("📚 Traditional Lao Musical Instruments")
        st.markdown("""
        Learn about the traditional musical instruments of Laos that this classifier can recognize.
        Each instrument has unique characteristics in terms of sound, playing technique, and cultural significance.
        """)
        
        # Create instrument cards
        for instrument_id, info in INSTRUMENT_INFO.items():
            if instrument_id == 'unknown':
                continue  # Skip unknown in the learning section
            
            with st.expander(f"🎵 {info['name']}", expanded=False):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    try:
                        if os.path.exists(info.get('image', '')):
                            st.image(info['image'], width=250)
                        else:
                            st.image(get_default_instrument_image(), width=250)
                    except:
                        st.image(get_default_instrument_image(), width=250)
                
                with col2:
                    st.markdown(f"**Description**: {info['description']}")
                    st.markdown(f"**Sound Characteristics**: {info.get('sound_characteristics', 'Not available')}")
                    st.markdown(f"**Playing Technique**: {info.get('playing_technique', 'Not available')}")
                    st.markdown(f"**Typical Pitch Range**: {info.get('typical_pitch_range', 'Not available')}")
                    st.markdown(f"**Cultural Significance**: {info.get('cultural_significance', 'Not available')}")
                    
                    # AI Recognition Features
                    st.markdown("**🤖 AI Recognition Features**:")
                    st.markdown(f"- {info.get('key_features', 'Not available')}")
        
        # Add cultural context
        st.markdown("---")
        st.subheader("🏛️ Cultural Context")
        st.markdown("""
        Traditional Lao music plays a crucial role in the cultural identity of Laos. These instruments are used in:
        
        - **Religious Ceremonies**: Buddhist temple rituals and festivals
        - **Cultural Celebrations**: New Year (Pi Mai), harvest festivals, and community gatherings  
        - **Folk Traditions**: Storytelling, courting songs, and social events
        - **Classical Performances**: Formal concerts and cultural presentations
        
        The **Khaen** holds special significance as it was recognized by UNESCO as part of Lao's 
        Intangible Cultural Heritage in 2017, highlighting its importance to Lao cultural identity.
        """)
        
        # Model performance by instrument
        st.subheader("🎯 AI Model Performance by Instrument")
        
        # Create performance visualization
        performance_data = {
            'Instrument': ['Sing', 'Khong Wong', 'Ranad', 'Pin', 'Saw', 'Khaen', 'Unknown'],
            'F1 Score': [0.974, 0.853, 0.820, 0.784, 0.786, 0.717, 0.817],
            'Precision': [1.000, 0.763, 0.722, 0.950, 0.655, 1.000, 0.979],
            'Recall': [0.949, 0.968, 0.950, 0.667, 0.983, 0.559, 0.701]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Create performance chart
        fig = px.bar(
            perf_df, 
            x='Instrument', 
            y='F1 Score',
            title='Model Performance by Instrument (F1 Score)',
            color='F1 Score',
            color_continuous_scale='viridis',
            text='F1 Score'
        )
        
        fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig.update_layout(
            xaxis_title='Instrument',
            yaxis_title='F1 Score',
            yaxis=dict(range=[0, 1.1]),
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Performance Notes**:
        - **Sing (Cymbals)** and **Khaen** show perfect precision (no false positives)
        - **Khong Wong** and **Saw** have excellent recall (few missed detections)  
        - **Pin** and **Khaen** are sometimes confused due to similar frequency characteristics
        - The model performs well even with background music present
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎵 Enhanced Lao Instrument Classifier | Built with Ensemble Deep Learning</p>
        <p>Preserving and promoting traditional Lao musical heritage through AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()