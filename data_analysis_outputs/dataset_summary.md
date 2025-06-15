# Lao Instrument Dataset Analysis Summary

## Dataset Overview

- **Total Classes**: 7
- **Class Names**: khean, khong_vong, pin, ranad, saw, sing, unknown
- **Total Files**: 2113

### File Counts by Class

| Class | File Count | Percentage |
|-------|------------|------------|
| khean | 295 | 14.0% |
| khong_vong | 314 | 14.9% |
| pin | 285 | 13.5% |
| ranad | 302 | 14.3% |
| saw | 288 | 13.6% |
| sing | 295 | 14.0% |
| unknown | 334 | 15.8% |

## Audio Duration Statistics

| Statistic | Value (seconds) |
|-----------|----------------|
| Mean | 7.08 |
| Std Dev | 1.02 |
| Min | 3.46 |
| 25% | 6.80 |
| Median | 7.00 |
| 75% | 7.50 |
| Max | 13.80 |

### Duration Statistics by Class

| Class | Mean (s) | Std Dev (s) | Min (s) | Max (s) |
|-------|----------|-------------|---------|--------|
| khean | 7.32 | 0.60 | 6.00 | 8.60 |
| khong_vong | 6.67 | 1.47 | 4.19 | 13.80 |
| pin | 7.07 | 0.87 | 5.50 | 11.20 |
| ranad | 6.92 | 0.86 | 5.30 | 10.30 |
| saw | 6.92 | 0.75 | 5.00 | 8.56 |
| sing | 7.28 | 1.47 | 3.46 | 10.40 |
| unknown | 7.41 | 0.54 | 6.80 | 9.00 |

## Unknown Class Analysis

### Unknown Class Subfolders

| Subfolder | File Count |
|-----------|------------|
| unknown-env_indoor | 51 |
| unknown-env_outdoor | 56 |
| unknown-human_speech | 82 |
| unknown-non-instru | 37 |
| unknown-non_speech | 28 |
| unknown-processed_sound | 26 |
| unknown-slience | 21 |
| unknown-unique | 33 |

## Visualization Links

- [Class Distribution](class_distribution.png)
- [Duration Distribution](duration_distribution.png)
- [Interactive Duration Analysis](duration_distribution_interactive.html)
- [Audio Properties](audio_properties.png)
- [Mel Spectrograms by Class](mel_spectrograms_by_class.png)
- [Waveforms by Class](waveforms_by_class.png)
- [Unknown Subfolder Distribution](unknown_subfolder_distribution.png)
- [Unknown Subfolder Samples](unknown_subfolder_samples.png)
- [t-SNE Visualization](tsne_visualization.png)
- [Interactive t-SNE Visualization](tsne_visualization.html)
- [Interactive PCA 3D Visualization](pca_3d_visualization.html)
