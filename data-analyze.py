import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from collections import defaultdict, Counter
import hashlib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AdvancedDatasetAnalyzer:
    def __init__(self, data_path="dataset", sample_rate=44100):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.instrument_data = defaultdict(list)
        self.file_hashes = {}
        self.spectral_features = defaultdict(list)
        
    def compute_audio_hash(self, audio_data):
        """Compute hash for audio similarity detection"""
        # Use spectral features for similarity detection
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio_data))
        
        # Create a feature vector for hashing
        feature_vector = f"{spectral_centroid:.6f}_{spectral_rolloff:.6f}_{zero_crossing_rate:.6f}"
        return hashlib.md5(feature_vector.encode()).hexdigest()
    
    def extract_comprehensive_features(self, audio_data):
        """Extract comprehensive audio features for analysis"""
        features = {}
        
        # Basic audio properties
        features['duration'] = len(audio_data) / self.sample_rate
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['max_amplitude'] = np.max(np.abs(audio_data))
        features['dynamic_range'] = np.max(audio_data) - np.min(audio_data)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0
        
        return features
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🔍 Starting comprehensive dataset analysis...")
        
        # Get all instrument folders
        instrument_folders = [d for d in os.listdir(self.data_path) 
                             if os.path.isdir(os.path.join(self.data_path, d))]
        
        all_features = []
        all_labels = []
        duplicate_groups = defaultdict(list)
        
        for folder in tqdm(instrument_folders, desc="Processing folders"):
            folder_path = os.path.join(self.data_path, folder)
            audio_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            
            for audio_file in tqdm(audio_files, desc=f"Processing {folder}", leave=False):
                file_path = os.path.join(folder_path, audio_file)
                
                try:
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=self.sample_rate)
                    
                    # Skip very short files
                    if len(audio) < sr * 0.5:
                        continue
                    
                    # Extract features
                    features = self.extract_comprehensive_features(audio)
                    features['instrument'] = folder
                    features['file_path'] = file_path
                    features['file_name'] = audio_file
                    
                    # Compute hash for duplicate detection
                    audio_hash = self.compute_audio_hash(audio)
                    duplicate_groups[audio_hash].append(file_path)
                    
                    all_features.append(features)
                    all_labels.append(folder)
                    
                    # Store for instrument-specific analysis
                    self.instrument_data[folder].append({
                        'features': features,
                        'audio': audio,
                        'file_path': file_path
                    })
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(all_features)
        
        return df, duplicate_groups
    
    def detect_data_leakage(self, df, duplicate_groups):
        """Detect potential data leakage issues"""
        print("\n🚨 Analyzing potential data leakage...")
        
        leakage_report = {
            'duplicate_files': [],
            'similar_recordings': [],
            'cross_contamination': []
        }
        
        # 1. Exact duplicates
        duplicates = {k: v for k, v in duplicate_groups.items() if len(v) > 1}
        if duplicates:
            print(f"⚠️ Found {len(duplicates)} groups of duplicate/very similar files:")
            for hash_val, files in duplicates.items():
                print(f"  - {len(files)} files: {[os.path.basename(f) for f in files[:3]]}")
                leakage_report['duplicate_files'].append(files)
        
        # 2. Cross-instrument similarity analysis
        feature_cols = [col for col in df.columns if col not in ['instrument', 'file_path', 'file_name']]
        X = df[feature_cols].values
        
        # Compute pairwise similarities
        similarities = cosine_similarity(X)
        
        # Find high similarity across different instruments
        high_similarity_threshold = 0.95
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                if (similarities[i, j] > high_similarity_threshold and 
                    df.iloc[i]['instrument'] != df.iloc[j]['instrument']):
                    leakage_report['cross_contamination'].append({
                        'file1': df.iloc[i]['file_path'],
                        'file2': df.iloc[j]['file_path'],
                        'instrument1': df.iloc[i]['instrument'],
                        'instrument2': df.iloc[j]['instrument'],
                        'similarity': similarities[i, j]
                    })
        
        return leakage_report
    
    def analyze_class_separability(self, df):
        """Analyze how well classes can be separated"""
        print("\n📊 Analyzing class separability...")
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in ['instrument', 'file_path', 'file_name']]
        X = df[feature_cols].values
        y = df['instrument'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) == 0:
            print("❌ No data available for separability analysis")
            return {}
        
        instruments = np.unique(y)
        
        if len(instruments) < 2:
            print("❌ Need at least 2 classes for separability analysis")
            return {}
        
        try:
            # PCA analysis
            pca = PCA(n_components=min(2, X.shape[1]))
            X_pca = pca.fit_transform(X)
        except Exception as e:
            print(f"❌ PCA failed: {e}")
            return {}
        
        try:
            # t-SNE analysis (with error handling)
            n_samples = len(X)
            perplexity = min(30, max(5, n_samples // 4))
            
            if n_samples > 3:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                X_tsne = tsne.fit_transform(X)
            else:
                print("⚠️ Too few samples for t-SNE")
                X_tsne = X_pca
        except Exception as e:
            print(f"⚠️ t-SNE failed, using PCA: {e}")
            X_tsne = X_pca
        
        # Plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # PCA plot
        colors = plt.cm.Set3(np.linspace(0, 1, len(instruments)))
        
        for i, instrument in enumerate(instruments):
            mask = y == instrument
            if np.any(mask):  # Only plot if class has samples
                ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                           c=[colors[i]], label=instrument, alpha=0.7)
        
        ax1.set_title('PCA: Feature Space Visualization')
        if pca.explained_variance_ratio_.shape[0] > 0:
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        if pca.explained_variance_ratio_.shape[0] > 1:
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.legend()
        
        # t-SNE plot
        for i, instrument in enumerate(instruments):
            mask = y == instrument
            if np.any(mask):  # Only plot if class has samples
                ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                           c=[colors[i]], label=instrument, alpha=0.7)
        ax2.set_title('t-SNE: Non-linear Feature Space')
        ax2.legend()
        
        # Inter-class distance analysis
        class_centroids = {}
        for instrument in instruments:
            mask = y == instrument
            class_centroids[instrument] = np.mean(X[mask], axis=0)
        
        # Compute inter-class distances
        distances = []
        pairs = []
        for i, inst1 in enumerate(instruments):
            for j, inst2 in enumerate(instruments):
                if i < j:
                    dist = np.linalg.norm(class_centroids[inst1] - class_centroids[inst2])
                    distances.append(float(dist))  # Ensure it's a scalar
                    pairs.append(f"{inst1}-{inst2}")
        
        # Only plot if we have data
        if distances and pairs:
            ax3.barh(pairs, distances)
            ax3.set_title('Inter-class Centroid Distances')
            ax3.set_xlabel('Euclidean Distance')
        else:
            ax3.text(0.5, 0.5, 'Not enough classes for distance analysis', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        # Intra-class variance analysis
        intra_variances = []
        for instrument in instruments:
            mask = y == instrument
            X_class = X[mask]
            if len(X_class) > 0:  # Check if class has samples
                centroid = class_centroids[instrument]
                variance = np.mean([np.linalg.norm(x - centroid) for x in X_class])
                intra_variances.append(float(variance))  # Ensure it's a scalar
            else:
                intra_variances.append(0.0)
        
        # Only plot if we have data
        if intra_variances:
            ax4.bar(instruments, intra_variances)
            ax4.set_title('Intra-class Variance')
            ax4.set_ylabel('Average Distance to Centroid')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No variance data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('class_separability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'pca_explained_variance': pca.explained_variance_ratio_,
            'inter_class_distances': dict(zip(pairs, distances)) if pairs and distances else {},
            'intra_class_variances': dict(zip(instruments, intra_variances)) if intra_variances else {}
        }
    
    def analyze_recording_quality(self, df):
        """Analyze recording quality metrics"""
        print("\n🎙️ Analyzing recording quality...")
        
        quality_metrics = {}
        
        # Group by instrument
        for instrument in df['instrument'].unique():
            inst_data = df[df['instrument'] == instrument]
            
            quality_metrics[instrument] = {
                'avg_rms_energy': inst_data['rms_energy'].mean(),
                'avg_dynamic_range': inst_data['dynamic_range'].mean(),
                'avg_max_amplitude': inst_data['max_amplitude'].mean(),
                'energy_std': inst_data['rms_energy'].std(),
                'clipping_risk': (inst_data['max_amplitude'] > 0.95).sum() / len(inst_data)
            }
        
        # Create quality comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        instruments = list(quality_metrics.keys())
        
        # RMS Energy comparison
        energies = [quality_metrics[inst]['avg_rms_energy'] for inst in instruments]
        ax1.bar(instruments, energies)
        ax1.set_title('Average RMS Energy by Instrument')
        ax1.tick_params(axis='x', rotation=45)
        
        # Dynamic Range comparison
        ranges = [quality_metrics[inst]['avg_dynamic_range'] for inst in instruments]
        ax2.bar(instruments, ranges)
        ax2.set_title('Average Dynamic Range by Instrument')
        ax2.tick_params(axis='x', rotation=45)
        
        # Energy consistency (lower std = more consistent)
        consistency = [quality_metrics[inst]['energy_std'] for inst in instruments]
        ax3.bar(instruments, consistency)
        ax3.set_title('Energy Consistency (lower = more consistent)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Clipping risk
        clipping = [quality_metrics[inst]['clipping_risk'] * 100 for inst in instruments]
        ax4.bar(instruments, clipping)
        ax4.set_title('Clipping Risk (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('recording_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return quality_metrics
    
    def generate_recommendations(self, df, leakage_report, separability_metrics, quality_metrics):
        """Generate recommendations based on analysis"""
        print("\n💡 Generating Recommendations...")
        
        recommendations = []
        
        # Data leakage recommendations
        if leakage_report['duplicate_files']:
            recommendations.append({
                'issue': 'Data Leakage - Duplicate Files',
                'severity': 'HIGH',
                'description': f"Found {len(leakage_report['duplicate_files'])} groups of duplicate/similar files",
                'solution': 'Remove duplicates or ensure they are in the same split (all train or all test)'
            })
        
        if leakage_report['cross_contamination']:
            recommendations.append({
                'issue': 'Data Leakage - Cross-contamination',
                'severity': 'HIGH', 
                'description': f"Found {len(leakage_report['cross_contamination'])} highly similar files across different instruments",
                'solution': 'Review these file pairs - they might be the same recording labeled differently'
            })
        
        # Class separability recommendations
        if separability_metrics['pca_explained_variance'][0] < 0.3:
            recommendations.append({
                'issue': 'Poor Feature Separability',
                'severity': 'MEDIUM',
                'description': f"First PC explains only {separability_metrics['pca_explained_variance'][0]:.2%} of variance",
                'solution': 'Consider different features or more diverse data collection'
            })
        
        # Data distribution recommendations
        instrument_counts = df['instrument'].value_counts()
        min_samples = instrument_counts.min()
        max_samples = instrument_counts.max()
        
        if max_samples / min_samples > 3:
            recommendations.append({
                'issue': 'Class Imbalance',
                'severity': 'MEDIUM',
                'description': f"Largest class has {max_samples/min_samples:.1f}x more samples than smallest",
                'solution': 'Balance classes through undersampling majority or collecting more minority samples'
            })
        
        # Quality recommendations
        for instrument, metrics in quality_metrics.items():
            if metrics['clipping_risk'] > 0.1:  # More than 10% clipping risk
                recommendations.append({
                    'issue': f'Recording Quality - {instrument}',
                    'severity': 'LOW',
                    'description': f"{instrument} has {metrics['clipping_risk']*100:.1f}% clipping risk",
                    'solution': 'Re-record with lower input gain to avoid clipping'
                })
        
        return recommendations

def main():
    """Run comprehensive dataset analysis"""
    analyzer = AdvancedDatasetAnalyzer()
    
    # Run analysis
    df, duplicate_groups = analyzer.analyze_dataset()
    leakage_report = analyzer.detect_data_leakage(df, duplicate_groups)
    separability_metrics = analyzer.analyze_class_separability(df)
    quality_metrics = analyzer.analyze_recording_quality(df)
    recommendations = analyzer.generate_recommendations(df, leakage_report, separability_metrics, quality_metrics)
    
    # Print summary report
    print("\n" + "="*80)
    print("📋 DATASET ANALYSIS SUMMARY REPORT")
    print("="*80)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   • Total files analyzed: {len(df)}")
    print(f"   • Instruments found: {df['instrument'].nunique()}")
    print(f"   • Average duration: {df['duration'].mean():.2f}s")
    
    print(f"\n🚨 Issues Found:")
    high_severity = [r for r in recommendations if r['severity'] == 'HIGH']
    medium_severity = [r for r in recommendations if r['severity'] == 'MEDIUM']
    low_severity = [r for r in recommendations if r['severity'] == 'LOW']
    
    for rec in high_severity:
        print(f"   🔴 {rec['issue']}: {rec['description']}")
        print(f"      Solution: {rec['solution']}")
    
    for rec in medium_severity:
        print(f"   🟡 {rec['issue']}: {rec['description']}")
        print(f"      Solution: {rec['solution']}")
    
    for rec in low_severity:
        print(f"   🟢 {rec['issue']}: {rec['description']}")
        print(f"      Solution: {rec['solution']}")
    
    # Save detailed results
    df.to_csv('dataset_analysis_detailed.csv', index=False)
    
    print(f"\n💾 Detailed analysis saved to 'dataset_analysis_detailed.csv'")
    print("📊 Visualization plots saved as PNG files")

if __name__ == "__main__":
    main()