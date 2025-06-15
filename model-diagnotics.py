import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import onnxruntime as ort
import json
import os
from tqdm import tqdm
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedDiagnostics:
    def __init__(self, model_path, label_mapping_path, data_path):
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.data_path = data_path
        
        # Load model and mappings
        self.session = ort.InferenceSession(model_path)
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        self.idx_to_label = {int(idx): label for label, idx in self.label_mapping.items()}
        
        # Audio processing config
        self.SAMPLE_RATE = 44100
        self.SEGMENT_DURATION = 6.0
        self.N_FFT = 2048
        self.HOP_LENGTH = 512
        self.N_MELS = 128
        
    def extract_features_from_dataset(self, max_files_per_class=50):
        """Extract features from dataset for analysis"""
        print("Extracting features from dataset...")
        
        features_data = []
        
        # Process each class
        for class_name in self.idx_to_label.values():
            if class_name == 'unknown':
                continue
                
            # Find files for this class
            class_files = []
            for root, dirs, files in os.walk(self.data_path):
                if class_name.lower() in os.path.basename(root).lower():
                    for file in files:
                        if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                            class_files.append(os.path.join(root, file))
            
            # Limit files per class for analysis
            class_files = class_files[:max_files_per_class]
            
            print(f"Processing {len(class_files)} files for {class_name}")
            
            for file_path in tqdm(class_files, desc=f"Processing {class_name}"):
                try:
                    # Load and process audio
                    audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE)
                    
                    # Extract multiple features
                    features = self.extract_comprehensive_features(audio, sr)
                    features['class'] = class_name
                    features['file_path'] = file_path
                    features['file_name'] = os.path.basename(file_path)
                    
                    # Get model prediction
                    prediction = self.predict_audio(audio, sr)
                    if prediction:
                        features.update({
                            'predicted_class': prediction['instrument'],
                            'confidence': prediction['confidence'],
                            'entropy': prediction['entropy']
                        })
                    
                    features_data.append(features)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
        
        return pd.DataFrame(features_data)
    
    def extract_comprehensive_features(self, audio, sr):
        """Extract comprehensive features for analysis"""
        # Clean and process audio
        best_segment = self.process_audio_with_best_segment(audio, sr)
        
        features = {}
        
        # Basic audio statistics
        features['duration'] = len(audio) / sr
        features['rms'] = np.sqrt(np.mean(best_segment**2))
        features['max_amplitude'] = np.max(np.abs(best_segment))
        features['min_amplitude'] = np.min(np.abs(best_segment))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=best_segment, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=best_segment, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=best_segment, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=best_segment)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(best_segment)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # Spectral contrast (harmonic vs noise content)
        spectral_contrast = librosa.feature.spectral_contrast(y=best_segment, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=best_segment, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(best_segment)
        features['harmonic_ratio'] = np.mean(harmonic**2) / (np.mean(best_segment**2) + 1e-8)
        features['percussive_ratio'] = np.mean(percussive**2) / (np.mean(best_segment**2) + 1e-8)
        
        # Pitch features
        pitches, magnitudes = librosa.piptrack(y=best_segment, sr=sr)
        pitch_values = pitches[pitches > 0]
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # Rhythm features
        try:
            tempo, beats = librosa.beat.beat_track(y=best_segment, sr=sr)
            features['tempo'] = tempo
            features['beat_strength'] = len(beats) / (len(best_segment) / sr) if len(beats) > 0 else 0
        except:
            features['tempo'] = 0
            features['beat_strength'] = 0
        
        # Frequency domain statistics
        fft = np.abs(np.fft.fft(best_segment))
        freqs = np.fft.fftfreq(len(best_segment), 1/sr)
        
        # Focus on relevant frequency range (20 Hz - 8000 Hz)
        valid_idx = (freqs >= 20) & (freqs <= 8000)
        fft_valid = fft[valid_idx]
        freqs_valid = freqs[valid_idx]
        
        if len(fft_valid) > 0:
            # Spectral peaks
            peaks_idx = np.argsort(fft_valid)[-10:]  # Top 10 peaks
            features['dominant_freq'] = freqs_valid[peaks_idx[-1]] if len(peaks_idx) > 0 else 0
            features['freq_peak_concentration'] = np.std(freqs_valid[peaks_idx]) if len(peaks_idx) > 1 else 0
            
            # Energy distribution across frequency bands
            low_freq = (freqs_valid <= 250)
            mid_freq = (freqs_valid > 250) & (freqs_valid <= 2000)
            high_freq = (freqs_valid > 2000)
            
            total_energy = np.sum(fft_valid**2) + 1e-8
            features['low_freq_energy'] = np.sum(fft_valid[low_freq]**2) / total_energy
            features['mid_freq_energy'] = np.sum(fft_valid[mid_freq]**2) / total_energy
            features['high_freq_energy'] = np.sum(fft_valid[high_freq]**2) / total_energy
        
        return features
    
    def process_audio_with_best_segment(self, audio, sr, segment_duration=6.0):
        """Extract best segment (reuse your logic)"""
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
            score = rms + 0.3 * contrast
            metrics.append(score)
        
        best_idx = np.argmax(metrics)
        return segments[best_idx]
    
    def predict_audio(self, audio, sr):
        """Make prediction using the loaded model"""
        try:
            if sr != self.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
            
            best_segment = self.process_audio_with_best_segment(audio, self.SAMPLE_RATE)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=best_segment,
                sr=self.SAMPLE_RATE,
                n_fft=self.N_FFT,
                hop_length=self.HOP_LENGTH,
                n_mels=self.N_MELS
            )
            
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_normalized = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Prepare for model
            features_batch = np.expand_dims(np.expand_dims(mel_spec_normalized, axis=-1), axis=0).astype(np.float32)
            
            # Get prediction
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: features_batch})
            probabilities = outputs[0][0]
            
            # Process results
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            instrument = self.idx_to_label[max_prob_idx]
            
            # Calculate entropy
            epsilon = 1e-10
            entropy = -np.sum(probabilities * np.log2(probabilities + epsilon)) / np.log2(len(probabilities))
            
            return {
                'instrument': instrument,
                'confidence': float(max_prob),
                'entropy': float(entropy),
                'probabilities': {self.idx_to_label[i]: float(prob) for i, prob in enumerate(probabilities)}
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def analyze_class_separability(self, features_df):
        """Analyze how separable the classes are in feature space"""
        print("Analyzing class separability...")
        
        # Select numerical features only
        numerical_features = features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_features if col not in ['confidence', 'entropy']]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['class']
        
        # Calculate inter-class distances
        class_centers = {}
        for class_name in y.unique():
            class_data = X[y == class_name]
            class_centers[class_name] = class_data.mean()
        
        # Calculate distances between class centers
        distances = {}
        classes = list(class_centers.keys())
        
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i < j:
                    dist = np.linalg.norm(class_centers[class1] - class_centers[class2])
                    distances[f"{class1}_vs_{class2}"] = dist
        
        # Create distance matrix visualization
        dist_matrix = np.zeros((len(classes), len(classes)))
        for i, class1 in enumerate(classes):
            for j, class2 in enumerate(classes):
                if i != j:
                    key1 = f"{class1}_vs_{class2}"
                    key2 = f"{class2}_vs_{class1}"
                    if key1 in distances:
                        dist_matrix[i, j] = distances[key1]
                    elif key2 in distances:
                        dist_matrix[i, j] = distances[key2]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(dist_matrix, annot=True, xticklabels=classes, yticklabels=classes, 
                   cmap='viridis', fmt='.2f')
        plt.title('Inter-Class Distance Matrix (Feature Space)')
        plt.tight_layout()
        plt.savefig('class_separability_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return distances, class_centers
    
    def identify_problematic_pairs(self, features_df):
        """Identify which class pairs are most often confused"""
        print("Identifying problematic class pairs...")
        
        # Create confusion matrix
        y_true = features_df['class']
        y_pred = features_df['predicted_class']
        
        # Remove rows where prediction failed
        valid_predictions = features_df.dropna(subset=['predicted_class'])
        y_true_valid = valid_predictions['class']
        y_pred_valid = valid_predictions['predicted_class']
        
        cm = confusion_matrix(y_true_valid, y_pred_valid, labels=sorted(y_true_valid.unique()))
        
        # Create confusion matrix heatmap
        plt.figure(figsize=(12, 10))
        classes = sorted(y_true_valid.unique())
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.title('Confusion Matrix - Model Predictions on Dataset')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig('dataset_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify most problematic pairs
        problematic_pairs = []
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                if i != j and cm[i, j] > 0:
                    error_rate = cm[i, j] / np.sum(cm[i, :])
                    problematic_pairs.append({
                        'true_class': true_class,
                        'predicted_class': pred_class,
                        'count': cm[i, j],
                        'error_rate': error_rate
                    })
        
        # Sort by error rate
        problematic_pairs.sort(key=lambda x: x['error_rate'], reverse=True)
        
        print("\nMost problematic class pairs:")
        for pair in problematic_pairs[:10]:
            print(f"{pair['true_class']} → {pair['predicted_class']}: "
                  f"{pair['count']} errors ({pair['error_rate']:.2%})")
        
        return problematic_pairs
    
    def analyze_feature_importance(self, features_df):
        """Analyze which features best discriminate between problematic classes"""
        print("Analyzing feature importance for problematic pairs...")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # Focus on khean vs pin (your main problem)
        khean_pin_data = features_df[features_df['class'].isin(['khean', 'pin'])].copy()
        
        if len(khean_pin_data) == 0:
            print("No khean/pin data found")
            return
        
        # Select numerical features
        numerical_features = khean_pin_data.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_features if col not in ['confidence', 'entropy']]
        
        X = khean_pin_data[feature_cols].fillna(0)
        y = khean_pin_data['class']
        
        # Train random forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Features for Khean vs Pin Classification')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_khean_pin.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Mutual information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        print("\nTop discriminative features (Mutual Information):")
        print(mi_df.head(15))
        
        return importance_df, mi_df
    
    def create_feature_distribution_plots(self, features_df):
        """Create plots showing feature distributions by class"""
        print("Creating feature distribution plots...")
        
        # Focus on most important features
        important_features = [
            'spectral_centroid_mean', 'spectral_contrast_mean', 'harmonic_ratio',
            'rms', 'pitch_mean', 'low_freq_energy', 'mid_freq_energy', 'high_freq_energy'
        ]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(important_features):
            if feature in features_df.columns:
                sns.boxplot(data=features_df, x='class', y=feature, ax=axes[i])
                axes[i].set_title(f'{feature} by Class')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendations(self, features_df, problematic_pairs):
        """Generate specific recommendations for improving the model"""
        print("\n" + "="*80)
        print("DIAGNOSTIC RESULTS AND RECOMMENDATIONS")
        print("="*80)
        
        # Analyze the main problems
        khean_pin_issues = [p for p in problematic_pairs if 
                           ('khean' in p['true_class'] and 'pin' in p['predicted_class']) or
                           ('pin' in p['true_class'] and 'khean' in p['predicted_class'])]
        
        print("\n1. MAIN ISSUES IDENTIFIED:")
        if khean_pin_issues:
            print("   - Khean and Pin are frequently confused")
            for issue in khean_pin_issues:
                print(f"     • {issue['true_class']} → {issue['predicted_class']}: {issue['error_rate']:.1%}")
        
        # Check confidence distributions
        if 'confidence' in features_df.columns:
            low_confidence = features_df[features_df['confidence'] < 0.6]
            print(f"   - {len(low_confidence)} samples have confidence < 60%")
            
            # Check which classes have low confidence
            low_conf_by_class = low_confidence['class'].value_counts()
            print("   - Classes with most low-confidence predictions:")
            for class_name, count in low_conf_by_class.head(3).items():
                print(f"     • {class_name}: {count} samples")
        
        print("\n2. SPECIFIC RECOMMENDATIONS:")
        
        print("\n   A. DATA QUALITY IMPROVEMENTS:")
        print("      - Filter out files with background music/accompaniment")
        print("      - Focus on solo instrument recordings")
        print("      - Ensure consistent recording quality across classes")
        print("      - Add more diverse playing techniques for each instrument")
        
        print("\n   B. FEATURE ENGINEERING:")
        print("      - Use harmonic-percussive separation more aggressively")
        print("      - Add temporal dynamics features (attack, decay, sustain)")
        print("      - Include pitch stability measures")
        print("      - Use multi-scale spectral analysis")
        
        print("\n   C. MODEL ARCHITECTURE:")
        print("      - Implement attention mechanisms to focus on discriminative regions")
        print("      - Use ensemble methods combining different architectures")
        print("      - Add uncertainty estimation to flag ambiguous cases")
        print("      - Try transformer-based architectures for better temporal modeling")
        
        print("\n   D. TRAINING STRATEGY:")
        print("      - Use focal loss to handle difficult examples")
        print("      - Implement curriculum learning (easy to hard examples)")
        print("      - Use mixup augmentation between different classes")
        print("      - Add contrastive loss to increase inter-class separation")
        
        print("\n   E. POST-PROCESSING:")
        print("      - Implement confidence thresholding")
        print("      - Use ensemble predictions from multiple segments")
        print("      - Add domain adaptation for different recording conditions")
        
        print("\n3. IMMEDIATE ACTIONS:")
        print("   1. Clean your dataset - remove files with obvious background music")
        print("   2. Retrain with the enhanced pipeline provided")
        print("   3. Implement confidence-based rejection for uncertain predictions")
        print("   4. Collect more high-quality solo recordings for khean and pin")
        
        print("\n" + "="*80)

def main():
    """Main diagnostic function"""
    # Initialize diagnostics
    diagnostics = AdvancedDiagnostics(
        model_path='model/model.onnx',
        label_mapping_path='model/label_mapping.json',
        data_path='dataset'
    )
    
    # Extract features from dataset
    features_df = diagnostics.extract_features_from_dataset(max_files_per_class=100)
    
    # Save features for later analysis
    features_df.to_csv('comprehensive_features_analysis.csv', index=False)
    
    # Perform various analyses
    distances, centers = diagnostics.analyze_class_separability(features_df)
    problematic_pairs = diagnostics.identify_problematic_pairs(features_df)
    importance_df, mi_df = diagnostics.analyze_feature_importance(features_df)
    diagnostics.create_feature_distribution_plots(features_df)
    
    # Generate recommendations
    diagnostics.generate_recommendations(features_df, problematic_pairs)
    
    print(f"\nAnalysis complete! Results saved to current directory.")
    print("Key files generated:")
    print("- comprehensive_features_analysis.csv")
    print("- class_separability_heatmap.png")
    print("- dataset_confusion_matrix.png")
    print("- feature_importance_khean_pin.png")
    print("- feature_distributions_by_class.png")

if __name__ == "__main__":
    main()