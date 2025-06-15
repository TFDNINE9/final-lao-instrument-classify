import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import librosa
import librosa.display
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import warnings
import soundfile as sf
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Create output directory
os.makedirs('data_analysis_outputs', exist_ok=True)

# Configuration
class Config:
    # Audio parameters
    SAMPLE_RATE = 44100
    SEGMENT_DURATION = 6.0
    
    # Feature extraction parameters
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    FMAX = 8000
    
    # Data paths
    DATA_PATH = "dataset"  # Change this to your dataset path
    
    # Instrument mapping (transliteration standardization)
    INSTRUMENT_MAPPING = {
        'khean': ['khean', 'khaen', 'แคน', 'ແຄນ'],
        'khong_vong': ['khong', 'kong', 'ຄ້ອງວົງ', 'khong_vong'],
        'pin': ['pin', 'ພິນ'],
        'ranad': ['ranad', 'nad', 'ລະນາດ'],
        'saw': ['saw', 'so', 'ຊໍ', 'ຊໍອູ້'],
        'sing': ['sing', 'ຊິ່ງ'],
        'unknown': ['unknown', 'other', 'misc']
    }

def map_instrument_folder(folder_name):
    """Map a folder name to the corresponding instrument class name"""
    folder_lower = folder_name.lower()
    
    # Special case for unknown folders
    if folder_lower.startswith('unknown-'):
        return 'unknown'
    
    for standard_name, variants in Config.INSTRUMENT_MAPPING.items():
        for variant in variants:
            if variant.lower() in folder_lower:
                return standard_name
    
    # Skip noise/background folders by returning None
    if 'noise' in folder_lower or 'background' in folder_lower:
        return None
    
    return folder_lower  # Return as is if no match

def process_audio_with_best_segment(audio, sr, segment_duration=6.0):
    """Extract the best segment from audio based on energy and spectral content"""
    # Calculate segment length in samples
    segment_len = int(segment_duration * sr)
    
    # If audio is shorter than segment duration, just pad
    if len(audio) <= segment_len:
        return np.pad(audio, (0, segment_len - len(audio)), mode='constant'), 0
    
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
        return audio[:segment_len] if len(audio) >= segment_len else np.pad(audio, (0, segment_len - len(audio)), mode='constant'), 0
    
    # Calculate metrics for each segment
    metrics = []
    for segment in segments:
        # Energy (RMS)
        rms = np.sqrt(np.mean(segment**2))
        
        # Spectral contrast
        contrast = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr))
        
        # Spectral flux
        stft = np.abs(librosa.stft(segment))
        if stft.shape[1] > 1:  # Make sure we have at least 2 frames
            flux = np.mean(np.diff(stft, axis=1)**2)
        else:
            flux = 0
        
        score = rms + 0.3 * contrast + 0.2 * flux
        metrics.append(score)
    
    best_idx = np.argmax(metrics)
    return segments[best_idx], segment_starts[best_idx]

def extract_features(audio_path, feature_types=None):
    """Extract various audio features from a file"""
    if feature_types is None:
        feature_types = ['mfcc', 'spectral_contrast', 'chroma', 'mel']
    
    features = {}
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        # Process best segment
        best_segment, segment_start = process_audio_with_best_segment(audio, sr, Config.SEGMENT_DURATION)
        
        # Store basic properties
        features['duration'] = len(audio) / sr
        features['segment_start'] = segment_start / sr
        features['segment_duration'] = len(best_segment) / sr
        features['rms'] = float(np.sqrt(np.mean(best_segment**2)))
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(best_segment)))
        
        # Extract mel spectrogram
        if 'mel' in feature_types:
            mel_spec = librosa.feature.melspectrogram(
                y=best_segment,
                sr=sr,
                n_fft=Config.N_FFT,
                hop_length=Config.HOP_LENGTH,
                n_mels=Config.N_MELS,
                fmax=Config.FMAX
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            features['mel_spectrogram'] = mel_spec_db
            
            # Compute summary statistics on mel spectrogram
            features['mel_mean'] = float(np.mean(mel_spec_db))
            features['mel_std'] = float(np.std(mel_spec_db))
            features['mel_min'] = float(np.min(mel_spec_db))
            features['mel_max'] = float(np.max(mel_spec_db))
        
        # Extract MFCCs
        if 'mfcc' in feature_types:
            mfccs = librosa.feature.mfcc(y=best_segment, sr=sr, n_mfcc=13)
            features['mfcc'] = mfccs
            features['mfcc_mean'] = mfccs.mean(axis=1).tolist()
            features['mfcc_std'] = mfccs.std(axis=1).tolist()
        
        # Extract spectral contrast
        if 'spectral_contrast' in feature_types:
            contrast = librosa.feature.spectral_contrast(y=best_segment, sr=sr)
            features['spectral_contrast'] = contrast
            features['contrast_mean'] = contrast.mean(axis=1).tolist()
            features['contrast_std'] = contrast.std(axis=1).tolist()
        
        # Extract chroma features
        if 'chroma' in feature_types:
            chroma = librosa.feature.chroma_stft(y=best_segment, sr=sr)
            features['chroma'] = chroma
            features['chroma_mean'] = chroma.mean(axis=1).tolist()
            features['chroma_std'] = chroma.std(axis=1).tolist()
        
        # Audio waveform
        features['waveform'] = best_segment
        
        return features, True
    
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {str(e)}")
        return None, False

def analyze_dataset():
    """Analyze the complete dataset including unknown class"""
    print("🔍 Analyzing dataset...")
    
    # Get all folders in the dataset
    all_folders = [d for d in os.listdir(Config.DATA_PATH) 
                  if os.path.isdir(os.path.join(Config.DATA_PATH, d))]
    
    # Map folders to instrument classes
    folder_mapping = {}
    for folder in all_folders:
        instrument = map_instrument_folder(folder)
        if instrument is not None:
            folder_mapping[folder] = instrument
    
    # Collect class names
    class_names = sorted(set(folder_mapping.values()))
    print(f"Detected {len(class_names)} classes: {class_names}")
    
    # Group folders by instrument class
    class_folders = defaultdict(list)
    for folder, instrument in folder_mapping.items():
        class_folders[instrument].append(folder)
    
    # Print folder mapping
    print("\nFolder to Instrument Mapping:")
    for instrument, folders in class_folders.items():
        print(f"  {instrument}: {', '.join(folders)}")
    
    # Collect file paths by class
    file_paths = defaultdict(list)
    
    for folder in tqdm(all_folders, desc="Finding audio files"):
        if folder not in folder_mapping:
            continue
            
        instrument = folder_mapping[folder]
        folder_path = os.path.join(Config.DATA_PATH, folder)
        
        # Get all audio files
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            file_paths[instrument].append(file_path)
    
    # Print file counts by class
    print("\nFile counts by class:")
    total_files = 0
    for instrument, files in file_paths.items():
        count = len(files)
        total_files += count
        print(f"  {instrument}: {count} files")
    print(f"Total files: {total_files}")
    
    # Analyze a sample of files from each class
    max_files_per_class = 50  # Limit for detailed analysis
    sampled_files = {}
    
    for instrument, files in file_paths.items():
        if len(files) > max_files_per_class:
            sampled_files[instrument] = random.sample(files, max_files_per_class)
        else:
            sampled_files[instrument] = files
    
    # Extract features
    features_by_class = defaultdict(list)
    file_info = []
    
    for instrument, files in tqdm(sampled_files.items(), desc="Extracting features"):
        for file_path in tqdm(files, desc=f"Processing {instrument}", leave=False):
            features, success = extract_features(file_path)
            if success:
                features_by_class[instrument].append(features)
                
                # Add to file info list
                file_info.append({
                    'path': file_path,
                    'instrument': instrument,
                    'duration': features['duration'],
                    'rms': features['rms'],
                    'zero_crossing_rate': features['zero_crossing_rate'],
                    'mel_mean': features['mel_mean'],
                    'mel_std': features['mel_std']
                })
    
    # Convert to DataFrame for easier analysis
    file_df = pd.DataFrame(file_info)
    
    return {
        'class_names': class_names,
        'folder_mapping': folder_mapping,
        'class_folders': class_folders,
        'file_paths': file_paths,
        'features_by_class': features_by_class,
        'file_df': file_df
    }

def plot_class_distribution(data):
    """Plot distribution of files across classes"""
    file_paths = data['file_paths']
    counts = [len(files) for instrument, files in file_paths.items()]
    instruments = list(file_paths.keys())
    
    # Sort by count
    sorted_data = sorted(zip(instruments, counts), key=lambda x: x[1], reverse=True)
    instruments = [x[0] for x in sorted_data]
    counts = [x[1] for x in sorted_data]
    
    # Create figure
    plt.figure(figsize=(12, 7))
    bars = plt.bar(instruments, counts, color=sns.color_palette("husl", len(instruments)))
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribution of Audio Files Across Classes', fontsize=16, fontweight='bold')
    plt.xlabel('Instrument Class', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_analysis_outputs/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_duration_distribution(data):
    """Plot distribution of file durations"""
    file_df = data['file_df']
    
    plt.figure(figsize=(15, 8))
    
    # Overall duration distribution
    plt.subplot(1, 2, 1)
    sns.histplot(file_df['duration'], bins=30, kde=True)
    plt.axvline(x=Config.SEGMENT_DURATION, color='red', linestyle='--', 
               label=f'Segment Duration ({Config.SEGMENT_DURATION}s)')
    plt.title('Overall Duration Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Duration (seconds)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    
    # Duration by class
    plt.subplot(1, 2, 2)
    sns.boxplot(x='instrument', y='duration', data=file_df)
    plt.axhline(y=Config.SEGMENT_DURATION, color='red', linestyle='--', 
               label=f'Segment Duration ({Config.SEGMENT_DURATION}s)')
    plt.title('Duration Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Instrument Class', fontsize=12)
    plt.ylabel('Duration (seconds)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_analysis_outputs/duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive Plotly version
    fig = make_subplots(rows=1, cols=2, 
                       subplot_titles=["Overall Duration Distribution", 
                                      "Duration Distribution by Class"])
    
    # Overall histogram
    fig.add_trace(
        go.Histogram(x=file_df['duration'], nbinsx=30, name="Duration",
                    marker_color='rgba(73, 133, 208, 0.7)'),
        row=1, col=1
    )
    
    # Add segment duration line
    fig.add_shape(type="line", x0=Config.SEGMENT_DURATION, x1=Config.SEGMENT_DURATION, 
                 y0=0, y1=1, yref="paper", xref="x",
                 line=dict(color="red", width=2, dash="dash"),
                 row=1, col=1)
    
    # Box plots by instrument
    for i, instrument in enumerate(file_df['instrument'].unique()):
        subset = file_df[file_df['instrument'] == instrument]
        fig.add_trace(
            go.Box(y=subset['duration'], name=instrument),
            row=1, col=2
        )
    
    # Add segment duration line to box plots
    fig.add_shape(type="line", x0=0, x1=1, xref="paper", 
                 y0=Config.SEGMENT_DURATION, y1=Config.SEGMENT_DURATION,
                 line=dict(color="red", width=2, dash="dash"),
                 row=1, col=2)
    
    fig.update_layout(
        title_text="Audio Duration Analysis",
        height=600,
        width=1200
    )
    
    fig.write_html('data_analysis_outputs/duration_distribution_interactive.html')

def plot_audio_properties(data):
    """Plot various audio properties by class"""
    file_df = data['file_df']
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot RMS by class
    sns.boxplot(x='instrument', y='rms', data=file_df, ax=axes[0])
    axes[0].set_title('RMS (Volume) by Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Instrument Class', fontsize=12)
    axes[0].set_ylabel('RMS', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot zero crossing rate by class
    sns.boxplot(x='instrument', y='zero_crossing_rate', data=file_df, ax=axes[1])
    axes[1].set_title('Zero Crossing Rate by Class', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Instrument Class', fontsize=12)
    axes[1].set_ylabel('Zero Crossing Rate', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Plot mel spectrogram mean by class
    sns.boxplot(x='instrument', y='mel_mean', data=file_df, ax=axes[2])
    axes[2].set_title('Mel Spectrogram Mean by Class', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Instrument Class', fontsize=12)
    axes[2].set_ylabel('Mean Value', fontsize=12)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Plot mel spectrogram std by class
    sns.boxplot(x='instrument', y='mel_std', data=file_df, ax=axes[3])
    axes[3].set_title('Mel Spectrogram Standard Deviation by Class', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Instrument Class', fontsize=12)
    axes[3].set_ylabel('Standard Deviation', fontsize=12)
    axes[3].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_analysis_outputs/audio_properties.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_mel_spectrograms_by_class(data):
    """Plot mel spectrograms for each class"""
    features_by_class = data['features_by_class']
    class_names = data['class_names']
    
    # Create a grid of subplots, 2 rows if <= 6 classes, otherwise more
    if len(class_names) <= 6:
        rows, cols = 2, 3
    else:
        rows = (len(class_names) + 2) // 3  # Ceiling division
        cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))
    axes = axes.flatten()
    
    for i, instrument in enumerate(class_names):
        if i >= len(axes):  # Safety check
            break
            
        if instrument in features_by_class and features_by_class[instrument]:
            # Get a random example
            example = random.choice(features_by_class[instrument])
            
            if 'mel_spectrogram' in example:
                mel_spec = example['mel_spectrogram']
                
                img = librosa.display.specshow(mel_spec, sr=Config.SAMPLE_RATE, 
                                             hop_length=Config.HOP_LENGTH,
                                             x_axis='time', y_axis='mel',
                                             ax=axes[i], fmax=Config.FMAX)
                axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
                fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
            else:
                axes[i].text(0.5, 0.5, f'No mel spectrogram\nfor {instrument}', 
                           ha='center', va='center', fontsize=12)
                axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'No data for\n{instrument}', 
                       ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
    
    # Hide any unused subplots
    for i in range(len(class_names), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Mel Spectrograms by Instrument Class', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('data_analysis_outputs/mel_spectrograms_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_waveforms_by_class(data):
    """Plot waveforms for each class"""
    features_by_class = data['features_by_class']
    class_names = data['class_names']
    
    # Create a grid of subplots
    if len(class_names) <= 6:
        rows, cols = 2, 3
    else:
        rows = (len(class_names) + 2) // 3  # Ceiling division
        cols = 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows))
    axes = axes.flatten()
    
    for i, instrument in enumerate(class_names):
        if i >= len(axes):  # Safety check
            break
            
        if instrument in features_by_class and features_by_class[instrument]:
            # Get a random example
            example = random.choice(features_by_class[instrument])
            
            if 'waveform' in example:
                waveform = example['waveform']
                time = np.linspace(0, len(waveform)/Config.SAMPLE_RATE, len(waveform))
                
                axes[i].plot(time, waveform, linewidth=0.5)
                axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Time (s)')
                axes[i].set_ylabel('Amplitude')
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f'No waveform\nfor {instrument}', 
                           ha='center', va='center', fontsize=12)
                axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
        else:
            axes[i].text(0.5, 0.5, f'No data for\n{instrument}', 
                       ha='center', va='center', fontsize=12)
            axes[i].set_title(f'{instrument}', fontsize=12, fontweight='bold')
    
    # Hide any unused subplots
    for i in range(len(class_names), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Waveforms by Instrument Class', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('data_analysis_outputs/waveforms_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_unknown_class(data):
    """Detailed analysis of the unknown class subfolders"""
    file_paths = data['file_paths']
    folder_mapping = data['folder_mapping']
    
    # Check if unknown class exists
    if 'unknown' not in file_paths:
        print("No 'unknown' class found in the dataset")
        return
    
    unknown_files = file_paths['unknown']
    
    # Group unknown files by subfolder
    unknown_subfolders = defaultdict(list)
    
    for file_path in unknown_files:
        # Extract subfolder name
        rel_path = os.path.relpath(file_path, Config.DATA_PATH)
        subfolder = rel_path.split(os.sep)[0]  # First part of relative path
        unknown_subfolders[subfolder].append(file_path)
    
    # Print subfolder information
    print("\nUnknown Class Subfolders:")
    total_unknown = 0
    subfolder_counts = {}
    
    for subfolder, files in unknown_subfolders.items():
        count = len(files)
        total_unknown += count
        subfolder_counts[subfolder] = count
        print(f"  {subfolder}: {count} files")
    
    print(f"Total unknown files: {total_unknown}")
    
    # Plot subfolder distribution
    plt.figure(figsize=(12, 7))
    
    subfolders = list(subfolder_counts.keys())
    counts = [subfolder_counts[sf] for sf in subfolders]
    
    # Sort by count
    sorted_data = sorted(zip(subfolders, counts), key=lambda x: x[1], reverse=True)
    subfolders = [x[0] for x in sorted_data]
    counts = [x[1] for x in sorted_data]
    
    bars = plt.bar(subfolders, counts, color=sns.color_palette("viridis", len(subfolders)))
    
    # Add count labels
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribution of "Unknown" Class by Subfolder', fontsize=16, fontweight='bold')
    plt.xlabel('Subfolder', fontsize=12)
    plt.ylabel('Number of Files', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('data_analysis_outputs/unknown_subfolder_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sample and analyze audio from each unknown subfolder
    sample_count = 3  # Number of samples per subfolder
    unknown_samples = {}
    
    for subfolder, files in unknown_subfolders.items():
        if len(files) > sample_count:
            unknown_samples[subfolder] = random.sample(files, sample_count)
        else:
            unknown_samples[subfolder] = files
    
    # Extract features
    unknown_features = defaultdict(list)
    
    for subfolder, files in tqdm(unknown_samples.items(), desc="Analyzing unknown subfolders"):
        for file_path in files:
            features, success = extract_features(file_path)
            if success:
                unknown_features[subfolder].append(features)
    
    # Plot sample spectrograms from each unknown subfolder
    rows = len(unknown_features)
    if rows == 0:
        print("No features extracted from unknown class")
        return
        
    fig, axes = plt.subplots(rows, 2, figsize=(16, 5*rows))
    
    # Handle case with only one subfolder
    if rows == 1:
        axes = np.array([axes])
    
    for i, (subfolder, features_list) in enumerate(unknown_features.items()):
        if not features_list:
            continue
            
        example = features_list[0]
        
        # Mel spectrogram
        if 'mel_spectrogram' in example:
            mel_spec = example['mel_spectrogram']
            img = librosa.display.specshow(mel_spec, sr=Config.SAMPLE_RATE, 
                                         hop_length=Config.HOP_LENGTH,
                                         x_axis='time', y_axis='mel',
                                         ax=axes[i, 0], fmax=Config.FMAX)
            axes[i, 0].set_title(f'{subfolder} - Mel Spectrogram', fontsize=12, fontweight='bold')
            fig.colorbar(img, ax=axes[i, 0], format='%+2.0f dB')
        
        # Waveform
        if 'waveform' in example:
            waveform = example['waveform']
            time = np.linspace(0, len(waveform)/Config.SAMPLE_RATE, len(waveform))
            
            axes[i, 1].plot(time, waveform, linewidth=0.5)
            axes[i, 1].set_title(f'{subfolder} - Waveform', fontsize=12, fontweight='bold')
            axes[i, 1].set_xlabel('Time (s)')
            axes[i, 1].set_ylabel('Amplitude')
            axes[i, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Audio Visualization of "Unknown" Class Subfolders', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('data_analysis_outputs/unknown_subfolder_samples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'unknown_subfolders': unknown_subfolders,
        'subfolder_counts': subfolder_counts,
        'unknown_features': unknown_features
    }

def visualize_feature_space(data):
    """Visualize feature space using dimensionality reduction"""
    features_by_class = data['features_by_class']
    
    # Collect features for visualization
    feature_vectors = []
    labels = []
    
    for instrument, features_list in features_by_class.items():
        for features in features_list:
            if 'mfcc_mean' in features:
                # Use MFCC means as a feature vector
                feature_vectors.append(features['mfcc_mean'])
                labels.append(instrument)
    
    if not feature_vectors:
        print("No feature vectors available for visualization")
        return
    
    # Convert to numpy array
    X = np.array(feature_vectors)
    
    # Use PCA for dimensionality reduction
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Use t-SNE for better separation
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # Create DataFrame for plotting
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2] if X_pca.shape[1] > 2 else np.zeros(X_pca.shape[0]),
        'instrument': labels
    })
    
    df_tsne = pd.DataFrame({
        'tSNE1': X_tsne[:, 0],
        'tSNE2': X_tsne[:, 1],
        'instrument': labels
    })
    
    # Create PCA 3D plot using plotly
    fig_pca = px.scatter_3d(df_pca, x='PC1', y='PC2', z='PC3', color='instrument',
                           title='PCA Feature Space Visualization',
                           labels={'instrument': 'Instrument Class'},
                           width=1000, height=800)
    
    fig_pca.update_layout(
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3'
        )
    )
    
    fig_pca.write_html('data_analysis_outputs/pca_3d_visualization.html')
    
    # Create t-SNE plot using plotly
    fig_tsne = px.scatter(df_tsne, x='tSNE1', y='tSNE2', color='instrument',
                         title='t-SNE Feature Space Visualization',
                         labels={'instrument': 'Instrument Class'},
                         width=1000, height=800)
    
    fig_tsne.update_layout(
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2'
    )
    
    fig_tsne.write_html('data_analysis_outputs/tsne_visualization.html')
    
    # Also create matplotlib versions
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x='tSNE1', y='tSNE2', hue='instrument', data=df_tsne, s=100, alpha=0.7)
    plt.title('t-SNE Feature Space Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(title='Instrument Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('data_analysis_outputs/tsne_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

def write_summary_report(data, unknown_analysis=None):
    """Write a summary report of the dataset analysis"""
    with open('data_analysis_outputs/dataset_summary.md', 'w') as f:
        f.write("# Lao Instrument Dataset Analysis Summary\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Classes**: {len(data['class_names'])}\n")
        f.write(f"- **Class Names**: {', '.join(data['class_names'])}\n")
        
        # File counts
        total_files = sum(len(files) for files in data['file_paths'].values())
        f.write(f"- **Total Files**: {total_files}\n\n")
        
        f.write("### File Counts by Class\n\n")
        f.write("| Class | File Count | Percentage |\n")
        f.write("|-------|------------|------------|\n")
        
        for instrument, files in data['file_paths'].items():
            count = len(files)
            percentage = count / total_files * 100
            f.write(f"| {instrument} | {count} | {percentage:.1f}% |\n")
        
        # Duration statistics
        f.write("\n## Audio Duration Statistics\n\n")
        
        if 'file_df' in data and not data['file_df'].empty:
            duration_stats = data['file_df']['duration'].describe()
            
            f.write("| Statistic | Value (seconds) |\n")
            f.write("|-----------|----------------|\n")
            f.write(f"| Mean | {duration_stats['mean']:.2f} |\n")
            f.write(f"| Std Dev | {duration_stats['std']:.2f} |\n")
            f.write(f"| Min | {duration_stats['min']:.2f} |\n")
            f.write(f"| 25% | {duration_stats['25%']:.2f} |\n")
            f.write(f"| Median | {duration_stats['50%']:.2f} |\n")
            f.write(f"| 75% | {duration_stats['75%']:.2f} |\n")
            f.write(f"| Max | {duration_stats['max']:.2f} |\n")
            
            # Duration by class
            f.write("\n### Duration Statistics by Class\n\n")
            f.write("| Class | Mean (s) | Std Dev (s) | Min (s) | Max (s) |\n")
            f.write("|-------|----------|-------------|---------|--------|\n")
            
            for instrument in data['class_names']:
                if instrument in data['file_df']['instrument'].values:
                    class_stats = data['file_df'][data['file_df']['instrument'] == instrument]['duration'].describe()
                    f.write(f"| {instrument} | {class_stats['mean']:.2f} | {class_stats['std']:.2f} | {class_stats['min']:.2f} | {class_stats['max']:.2f} |\n")
        
        # Unknown class analysis
        if unknown_analysis and 'unknown' in data['class_names']:
            f.write("\n## Unknown Class Analysis\n\n")
            
            if 'subfolder_counts' in unknown_analysis:
                f.write("### Unknown Class Subfolders\n\n")
                f.write("| Subfolder | File Count |\n")
                f.write("|-----------|------------|\n")
                
                for subfolder, count in unknown_analysis['subfolder_counts'].items():
                    f.write(f"| {subfolder} | {count} |\n")
        
        # Visualization links
        f.write("\n## Visualization Links\n\n")
        f.write("- [Class Distribution](class_distribution.png)\n")
        f.write("- [Duration Distribution](duration_distribution.png)\n")
        f.write("- [Interactive Duration Analysis](duration_distribution_interactive.html)\n")
        f.write("- [Audio Properties](audio_properties.png)\n")
        f.write("- [Mel Spectrograms by Class](mel_spectrograms_by_class.png)\n")
        f.write("- [Waveforms by Class](waveforms_by_class.png)\n")
        
        if unknown_analysis:
            f.write("- [Unknown Subfolder Distribution](unknown_subfolder_distribution.png)\n")
            f.write("- [Unknown Subfolder Samples](unknown_subfolder_samples.png)\n")
        
        f.write("- [t-SNE Visualization](tsne_visualization.png)\n")
        f.write("- [Interactive t-SNE Visualization](tsne_visualization.html)\n")
        f.write("- [Interactive PCA 3D Visualization](pca_3d_visualization.html)\n")

def main():
    """Main function to run the dataset analysis"""
    print("🔍 Starting comprehensive dataset analysis...")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(Config.DATA_PATH):
        print(f"⚠️ Dataset path '{Config.DATA_PATH}' not found!")
        print("Please update Config.DATA_PATH to point to your dataset directory.")
        return
    
    # Analyze dataset
    try:
        data = analyze_dataset()
        
        # Check if any data was found
        if not data['file_paths']:
            print("⚠️ No valid audio files found in the dataset!")
            return
        
        # Generate visualizations
        print("\n🎨 Creating visualizations...")
        
        # Basic dataset statistics
        plot_class_distribution(data)
        plot_duration_distribution(data)
        plot_audio_properties(data)
        
        # Audio visualizations
        plot_mel_spectrograms_by_class(data)
        plot_waveforms_by_class(data)
        
        # Feature space visualization
        visualize_feature_space(data)
        
        # Analyze unknown class specifically
        unknown_analysis = None
        if 'unknown' in data['class_names']:
            print("\n🔍 Analyzing 'unknown' class...")
            unknown_analysis = analyze_unknown_class(data)
        
        # Write summary report
        write_summary_report(data, unknown_analysis)
        
        print("=" * 70)
        print("✅ Dataset analysis completed successfully!")
        print("📁 Check the 'data_analysis_outputs' folder for all generated visualizations and reports.")
        
    except Exception as e:
        print(f"⚠️ Error during dataset analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()