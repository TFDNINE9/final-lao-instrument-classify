import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import hashlib
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SimpleDatasetAnalyzer:
    def __init__(self, data_path="dataset", sample_rate=44100):
        self.data_path = data_path
        self.sample_rate = sample_rate
        
    def analyze_basic_stats(self):
        """Basic dataset statistics"""
        print("🔍 Analyzing basic dataset statistics...")
        
        stats = {
            'total_files': 0,
            'instruments': defaultdict(int),
            'file_durations': [],
            'file_sizes': [],
            'extensions': defaultdict(int)
        }
        
        # Get all folders
        folders = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d))]
        
        for folder in tqdm(folders, desc="Analyzing folders"):
            folder_path = os.path.join(self.data_path, folder)
            
            # Get audio files
            audio_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            
            stats['instruments'][folder] = len(audio_files)
            stats['total_files'] += len(audio_files)
            
            # Analyze each file
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                
                # File extension
                ext = os.path.splitext(audio_file)[1].lower()
                stats['extensions'][ext] += 1
                
                # File size
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    stats['file_sizes'].append(file_size)
                except:
                    pass
                
                # Duration (sample a few files to avoid too much processing)
                if len(stats['file_durations']) < 100:  # Sample first 100 files
                    try:
                        duration = librosa.get_duration(path=file_path)
                        stats['file_durations'].append(duration)
                    except:
                        pass
        
        return stats
    
    def detect_potential_duplicates(self):
        """Simple duplicate detection based on file names and sizes"""
        print("🔍 Detecting potential duplicates...")
        
        files_info = []
        
        folders = [d for d in os.listdir(self.data_path) 
                  if os.path.isdir(os.path.join(self.data_path, d))]
        
        for folder in folders:
            folder_path = os.path.join(self.data_path, folder)
            audio_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
            
            for audio_file in audio_files:
                file_path = os.path.join(folder_path, audio_file)
                try:
                    file_size = os.path.getsize(file_path)
                    files_info.append({
                        'path': file_path,
                        'name': audio_file,
                        'size': file_size,
                        'instrument': folder
                    })
                except:
                    pass
        
        # Group by size and name similarity
        potential_duplicates = []
        
        # Simple name-based detection
        name_groups = defaultdict(list)
        for file_info in files_info:
            # Create a simple key based on filename without extension
            name_key = os.path.splitext(file_info['name'])[0].lower()
            name_groups[name_key].append(file_info)
        
        for name_key, group in name_groups.items():
            if len(group) > 1:
                potential_duplicates.append({
                    'type': 'similar_names',
                    'files': group,
                    'key': name_key
                })
        
        # Size-based detection
        size_groups = defaultdict(list)
        for file_info in files_info:
            size_groups[file_info['size']].append(file_info)
        
        for size, group in size_groups.items():
            if len(group) > 1:
                potential_duplicates.append({
                    'type': 'same_size',
                    'files': group,
                    'size': size
                })
        
        return potential_duplicates
    
    def analyze_data_distribution(self, stats):
        """Analyze data distribution and balance"""
        print("📊 Analyzing data distribution...")
        
        instruments = list(stats['instruments'].keys())
        counts = list(stats['instruments'].values())
        
        # Calculate balance metrics
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Instrument distribution
        plt.subplot(2, 2, 1)
        bars = plt.bar(instruments, counts)
        plt.title('Files per Instrument')
        plt.xlabel('Instrument')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        # Subplot 2: Duration distribution
        if stats['file_durations']:
            plt.subplot(2, 2, 2)
            plt.hist(stats['file_durations'], bins=20, alpha=0.7)
            plt.title('Duration Distribution (Sample)')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Number of Files')
            
            # Add statistics
            mean_duration = np.mean(stats['file_durations'])
            plt.axvline(mean_duration, color='red', linestyle='--', 
                       label=f'Mean: {mean_duration:.2f}s')
            plt.legend()
        
        # Subplot 3: File size distribution
        if stats['file_sizes']:
            plt.subplot(2, 2, 3)
            plt.hist(stats['file_sizes'], bins=20, alpha=0.7)
            plt.title('File Size Distribution')
            plt.xlabel('Size (MB)')
            plt.ylabel('Number of Files')
            
            mean_size = np.mean(stats['file_sizes'])
            plt.axvline(mean_size, color='red', linestyle='--', 
                       label=f'Mean: {mean_size:.2f}MB')
            plt.legend()
        
        # Subplot 4: File format distribution
        plt.subplot(2, 2, 4)
        extensions = list(stats['extensions'].keys())
        ext_counts = list(stats['extensions'].values())
        plt.pie(ext_counts, labels=extensions, autopct='%1.1f%%')
        plt.title('File Format Distribution')
        
        plt.tight_layout()
        plt.savefig('basic_dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'min_samples': min_count,
            'max_samples': max_count,
            'total_instruments': len(instruments)
        }
    
    def generate_simple_report(self, stats, duplicates, distribution_metrics):
        """Generate a simple analysis report"""
        print("\n" + "="*80)
        print("📋 SIMPLE DATASET ANALYSIS REPORT")
        print("="*80)
        
        print(f"\n📊 Basic Statistics:")
        print(f"   • Total audio files: {stats['total_files']}")
        print(f"   • Number of instruments: {len(stats['instruments'])}")
        print(f"   • File formats: {dict(stats['extensions'])}")
        
        if stats['file_durations']:
            print(f"   • Average duration: {np.mean(stats['file_durations']):.2f}s")
            print(f"   • Duration range: {np.min(stats['file_durations']):.2f}s - {np.max(stats['file_durations']):.2f}s")
        
        if stats['file_sizes']:
            print(f"   • Average file size: {np.mean(stats['file_sizes']):.2f}MB")
        
        print(f"\n🎵 Instrument Distribution:")
        for instrument, count in stats['instruments'].items():
            percentage = (count / stats['total_files']) * 100
            print(f"   • {instrument}: {count} files ({percentage:.1f}%)")
        
        print(f"\n⚖️ Data Balance:")
        print(f"   • Imbalance ratio: {distribution_metrics['imbalance_ratio']:.2f}")
        print(f"   • Smallest class: {distribution_metrics['min_samples']} files")
        print(f"   • Largest class: {distribution_metrics['max_samples']} files")
        
        if distribution_metrics['imbalance_ratio'] > 3:
            print("   ⚠️ HIGH IMBALANCE DETECTED - Consider balancing classes")
        
        print(f"\n🔍 Potential Issues:")
        duplicate_count = len(duplicates)
        if duplicate_count > 0:
            print(f"   • Found {duplicate_count} groups of potential duplicates")
            print("   ⚠️ POTENTIAL DATA LEAKAGE - Review duplicate files")
        else:
            print("   ✅ No obvious duplicates detected")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if distribution_metrics['imbalance_ratio'] > 3:
            print("   🔴 HIGH PRIORITY: Balance your dataset")
            print("     - Collect more samples for underrepresented instruments")
            print("     - Or reduce samples from overrepresented instruments")
        
        if duplicate_count > 0:
            print("   🔴 HIGH PRIORITY: Remove duplicates")
            print("     - Check the duplicate groups listed above")
            print("     - Ensure same recordings aren't in both train and test sets")
        
        if stats['total_files'] < 100:
            print("   🟡 MEDIUM PRIORITY: Collect more data")
            print("     - Current dataset might be too small for deep learning")
            print("     - Aim for at least 50-100 samples per instrument")
        
        if len(set(stats['extensions'].keys())) > 1:
            print("   🟢 LOW PRIORITY: Standardize file formats")
            print("     - Consider converting all files to WAV for consistency")
        
        return {
            'total_files': stats['total_files'],
            'instruments': len(stats['instruments']),
            'imbalance_ratio': distribution_metrics['imbalance_ratio'],
            'duplicate_groups': duplicate_count,
            'recommendations': []
        }

def main():
    """Run simple dataset analysis"""
    analyzer = SimpleDatasetAnalyzer()
    
    try:
        # Basic statistics
        stats = analyzer.analyze_basic_stats()
        
        # Duplicate detection
        duplicates = analyzer.detect_potential_duplicates()
        
        # Distribution analysis
        distribution_metrics = analyzer.analyze_data_distribution(stats)
        
        # Generate report
        report = analyzer.generate_simple_report(stats, duplicates, distribution_metrics)
        
        # Print duplicate details
        if duplicates:
            print(f"\n🔍 Duplicate Analysis Details:")
            for i, dup_group in enumerate(duplicates[:5]):  # Show first 5 groups
                print(f"\nGroup {i+1} ({dup_group['type']}):")
                for file_info in dup_group['files'][:3]:  # Show first 3 files per group
                    print(f"   • {file_info['instrument']}: {file_info['name']}")
        
        print(f"\n💾 Analysis complete! Generated 'basic_dataset_analysis.png'")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()