import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd

def analyze_saw_confusion(data_path="dataset"):
    """Specific analysis for saw instrument confusion with pin and khean"""
    
    # Load data for saw, pin, and khean specifically
    target_instruments = ['saw', 'pin', 'khean']
    instrument_data = {inst: [] for inst in target_instruments}
    
    print("🎻 Analyzing saw instrument confusion...")
    
    for folder in os.listdir(data_path):
        if not os.path.isdir(os.path.join(data_path, folder)):
            continue
            
        # Map folder name to standard instrument name
        folder_lower = folder.lower()
        instrument = None
        
        if any(x in folder_lower for x in ['saw', 'so', 'ຊໍ']):
            instrument = 'saw'
        elif any(x in folder_lower for x in ['pin', 'ພິນ']):
            instrument = 'pin'  
        elif any(x in folder_lower for x in ['khean', 'khaen', 'ແຄນ']):
            instrument = 'khean'
        
        if instrument not in target_instruments:
            continue
            
        folder_path = os.path.join(data_path, folder)
        audio_files = [f for f in os.listdir(folder_path) 
                      if f.endswith(('.wav', '.mp3', '.flac'))]
        
        for audio_file in audio_files:
            file_path = os.path.join(folder_path, audio_file)
            try:
                audio, sr = librosa.load(file_path, sr=44100)
                if len(audio) < sr * 0.5:  # Skip very short files
                    continue
                    
                # Extract detailed features for comparison
                features = extract_detailed_features(audio, sr)
                features['instrument'] = instrument
                features['file_name'] = audio_file
                features['file_path'] = file_path
                
                instrument_data[instrument].append(features)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame for analysis
    all_data = []
    for instrument, data_list in instrument_data.items():
        all_data.extend(data_list)
    
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("❌ No data found for target instruments!")
        return
    
    print(f"📊 Data loaded:")
    for instrument in target_instruments:
        count = len(df[df['instrument'] == instrument])
        print(f"   • {instrument}: {count} files")
    
    # 1. Feature similarity analysis
    analyze_feature_similarity(df)
    
    # 2. Spectral analysis
    analyze_spectral_characteristics(df)
    
    # 3. Temporal pattern analysis  
    analyze_temporal_patterns(df)
    
    # 4. Clustering analysis
    analyze_clustering_patterns(df)
    
    return df

def extract_detailed_features(audio, sr):
    """Extract detailed features for confusion analysis"""
    features = {}
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    features['spectral_centroid_range'] = np.max(spectral_centroids) - np.min(spectral_centroids)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    
    # Harmonic-percussive separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio)
    features['harmonic_ratio'] = np.sum(y_harmonic**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2))
    
    # Spectral contrast (important for distinguishing bowed vs plucked)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
    
    # Chroma features (pitch class profiles)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    features['chroma_range'] = np.max(chroma) - np.min(chroma)
    
    # Zero crossing rate (texture indicator)
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # MFCC features (timbre)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    # Tempo and rhythm
    try:
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        # Beat strength variation
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            features['beat_consistency'] = 1.0 / (np.std(beat_intervals) + 1e-6)
        else:
            features['beat_consistency'] = 0
    except:
        features['tempo'] = 0
        features['beat_consistency'] = 0
    
    # Energy characteristics
    features['rms_energy'] = np.sqrt(np.mean(audio**2))
    features['energy_std'] = np.std(librosa.feature.rms(y=audio)[0])
    
    return features

def analyze_feature_similarity(df):
    """Analyze feature similarity between confused instruments"""
    print("\n🔍 Feature Similarity Analysis:")
    
    # Select numeric features only
    feature_cols = [col for col in df.columns 
                   if col not in ['instrument', 'file_name', 'file_path'] 
                   and df[col].dtype in ['float64', 'int64']]
    
    # Compare saw vs pin and saw vs khean
    comparisons = [('saw', 'pin'), ('saw', 'khean'), ('pin', 'khean')]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (inst1, inst2) in enumerate(comparisons):
        data1 = df[df['instrument'] == inst1][feature_cols]
        data2 = df[df['instrument'] == inst2][feature_cols]
        
        if len(data1) == 0 or len(data2) == 0:
            continue
            
        # Calculate mean feature differences
        mean1 = data1.mean()
        mean2 = data2.mean()
        
        # Find most discriminative features
        feature_diffs = abs(mean1 - mean2) / (mean1.std() + mean2.std() + 1e-6)
        top_discriminative = feature_diffs.nlargest(10)
        
        print(f"\n{inst1.upper()} vs {inst2.upper()}:")
        print("Top discriminative features:")
        for feature, diff in top_discriminative.items():
            print(f"  • {feature}: {diff:.3f}")
            print(f"    {inst1}: {mean1[feature]:.3f}, {inst2}: {mean2[feature]:.3f}")
        
        # Plot feature distributions for top discriminative features
        top_features = top_discriminative.head(3).index
        
        for i, feature in enumerate(top_features):
            if i < 3:  # Only plot top 3
                axes[idx].hist(data1[feature], alpha=0.5, label=f'{inst1}', bins=20)
                axes[idx].hist(data2[feature], alpha=0.5, label=f'{inst2}', bins=20)
                axes[idx].set_title(f'{inst1} vs {inst2}: {feature}')
                axes[idx].legend()
                break
    
    plt.tight_layout()
    plt.savefig('feature_similarity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_spectral_characteristics(df):
    """Analyze spectral characteristics that might cause confusion"""
    print("\n🌈 Spectral Characteristics Analysis:")
    
    instruments = df['instrument'].unique()
    colors = ['red', 'blue', 'green']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Spectral centroid comparison
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[0, 0].hist(data['spectral_centroid_mean'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[0, 0].set_title('Spectral Centroid Distribution')
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].legend()
    
    # Harmonic ratio comparison (key for bowed vs plucked)
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[0, 1].hist(data['harmonic_ratio'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[0, 1].set_title('Harmonic Ratio Distribution')
    axes[0, 1].set_xlabel('Harmonic Ratio')
    axes[0, 1].legend()
    
    # Zero crossing rate (texture)
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[0, 2].hist(data['zcr_mean'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[0, 2].set_title('Zero Crossing Rate Distribution')
    axes[0, 2].set_xlabel('ZCR')
    axes[0, 2].legend()
    
    # Spectral bandwidth
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[1, 0].hist(data['spectral_bandwidth_mean'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[1, 0].set_title('Spectral Bandwidth Distribution')
    axes[1, 0].set_xlabel('Bandwidth (Hz)')
    axes[1, 0].legend()
    
    # Spectral rolloff
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[1, 1].hist(data['spectral_rolloff_mean'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[1, 1].set_title('Spectral Rolloff Distribution')
    axes[1, 1].set_xlabel('Rolloff Frequency (Hz)')
    axes[1, 1].legend()
    
    # Energy characteristics
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[1, 2].hist(data['rms_energy'], alpha=0.6, 
                       label=instrument, color=colors[i], bins=20)
    axes[1, 2].set_title('RMS Energy Distribution')
    axes[1, 2].set_xlabel('RMS Energy')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('spectral_characteristics_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistical analysis
    print("\nSpectral Statistics:")
    feature_cols = ['spectral_centroid_mean', 'harmonic_ratio', 'zcr_mean', 
                   'spectral_bandwidth_mean', 'spectral_rolloff_mean', 'rms_energy']
    
    for feature in feature_cols:
        print(f"\n{feature}:")
        for instrument in instruments:
            data = df[df['instrument'] == instrument][feature]
            print(f"  {instrument}: mean={data.mean():.3f}, std={data.std():.3f}")

def analyze_temporal_patterns(df):
    """Analyze temporal patterns that might cause confusion"""
    print("\n⏱️ Temporal Pattern Analysis:")
    
    instruments = df['instrument'].unique()
    colors = ['red', 'blue', 'green']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Tempo distribution
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        valid_tempo = data[data['tempo'] > 0]['tempo']  # Filter out invalid tempo values
        if len(valid_tempo) > 0:
            axes[0].hist(valid_tempo, alpha=0.6, label=instrument, 
                        color=colors[i], bins=20)
    axes[0].set_title('Tempo Distribution')
    axes[0].set_xlabel('BPM')
    axes[0].legend()
    
    # Beat consistency
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        valid_consistency = data[data['beat_consistency'] > 0]['beat_consistency']
        if len(valid_consistency) > 0:
            axes[1].hist(valid_consistency, alpha=0.6, label=instrument, 
                        color=colors[i], bins=20)
    axes[1].set_title('Beat Consistency Distribution')
    axes[1].set_xlabel('Consistency Score')
    axes[1].legend()
    
    # Energy variation
    for i, instrument in enumerate(instruments):
        data = df[df['instrument'] == instrument]
        axes[2].hist(data['energy_std'], alpha=0.6, label=instrument, 
                    color=colors[i], bins=20)
    axes[2].set_title('Energy Variation Distribution')
    axes[2].set_xlabel('Energy Std Dev')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_clustering_patterns(df):
    """Analyze how instruments cluster in feature space"""
    print("\n🎯 Clustering Analysis:")
    
    # Select numeric features
    feature_cols = [col for col in df.columns 
                   if col not in ['instrument', 'file_name', 'file_path'] 
                   and df[col].dtype in ['float64', 'int64']]
    
    X = df[feature_cols].fillna(0)  # Handle any NaN values
    y = df['instrument']
    
    # Perform t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
    X_tsne = tsne.fit_transform(X)
    
    # Plot t-SNE results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    instruments = y.unique()
    colors = ['red', 'blue', 'green']
    
    # t-SNE colored by true labels
    for i, instrument in enumerate(instruments):
        mask = y == instrument
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[i], label=instrument, alpha=0.7, s=50)
    ax1.set_title('t-SNE: True Instrument Labels')
    ax1.legend()
    
    # K-means clustering to see natural groupings
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # t-SNE colored by clusters
    for i in range(3):
        mask = cluster_labels == i
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)
    ax2.set_title('t-SNE: K-means Clusters')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze cluster-label agreement
    print("\nCluster-Label Analysis:")
    cluster_instrument_map = {}
    for cluster_id in range(3):
        cluster_mask = cluster_labels == cluster_id
        cluster_instruments = y[cluster_mask]
        most_common = cluster_instruments.mode()
        if len(most_common) > 0:
            cluster_instrument_map[cluster_id] = most_common[0]
            purity = (cluster_instruments == most_common[0]).sum() / len(cluster_instruments)
            print(f"Cluster {cluster_id}: mostly {most_common[0]} ({purity:.2%} purity)")
    
    # Calculate overall clustering accuracy
    predicted_labels = [cluster_instrument_map.get(cluster, 'unknown') for cluster in cluster_labels]
    accuracy = sum(pred == true for pred, true in zip(predicted_labels, y)) / len(y)
    print(f"\nClustering accuracy: {accuracy:.2%}")
    
    return X_tsne, cluster_labels

def generate_saw_specific_recommendations(df):
    """Generate specific recommendations for saw classification issues"""
    print("\n💡 Saw-Specific Recommendations:")
    
    saw_data = df[df['instrument'] == 'saw']
    pin_data = df[df['instrument'] == 'pin']
    khean_data = df[df['instrument'] == 'khean']
    
    recommendations = []
    
    # Check if saw has distinctive harmonic characteristics
    if len(saw_data) > 0:
        saw_harmonic = saw_data['harmonic_ratio'].mean()
        pin_harmonic = pin_data['harmonic_ratio'].mean() if len(pin_data) > 0 else 0
        khean_harmonic = khean_data['harmonic_ratio'].mean() if len(khean_data) > 0 else 0
        
        if abs(saw_harmonic - pin_harmonic) < 0.1:
            recommendations.append({
                'issue': 'Saw-Pin Harmonic Confusion',
                'description': f'Saw and Pin have similar harmonic ratios ({saw_harmonic:.3f} vs {pin_harmonic:.3f})',
                'solution': 'Focus on bow articulation patterns, vibrato detection, or attack characteristics'
            })
        
        if abs(saw_harmonic - khean_harmonic) < 0.1:
            recommendations.append({
                'issue': 'Saw-Khean Harmonic Confusion', 
                'description': f'Saw and Khean have similar harmonic ratios ({saw_harmonic:.3f} vs {khean_harmonic:.3f})',
                'solution': 'Focus on continuous vs. intermittent playing patterns'
            })
    
    # Check spectral characteristics
    if len(saw_data) > 0:
        saw_centroid = saw_data['spectral_centroid_mean'].mean()
        pin_centroid = pin_data['spectral_centroid_mean'].mean() if len(pin_data) > 0 else 0
        
        if abs(saw_centroid - pin_centroid) < 500:  # Less than 500Hz difference
            recommendations.append({
                'issue': 'Similar Spectral Centroids',
                'description': f'Saw and Pin have similar frequency content ({saw_centroid:.0f}Hz vs {pin_centroid:.0f}Hz)',
                'solution': 'Record different playing techniques (arco vs pizzicato for saw, different plucking styles for pin)'
            })
    
    # Data collection recommendations
    if len(saw_data) < 50:
        recommendations.append({
            'issue': 'Insufficient Saw Data',
            'description': f'Only {len(saw_data)} saw samples found',
            'solution': 'Collect more diverse saw recordings: different bowing techniques, dynamics, articulations'
        })
    
    for rec in recommendations:
        print(f"\n🔍 {rec['issue']}:")
        print(f"   Problem: {rec['description']}")
        print(f"   Solution: {rec['solution']}")

if __name__ == "__main__":
    # Run the saw confusion analysis
    df = analyze_saw_confusion()
    if df is not None and len(df) > 0:
        generate_saw_specific_recommendations(df)
        print(f"\n📊 Analysis complete! Generated visualization files:")
        print("   • feature_similarity_analysis.png")
        print("   • spectral_characteristics_analysis.png") 
        print("   • temporal_patterns_analysis.png")
        print("   • clustering_analysis.png")