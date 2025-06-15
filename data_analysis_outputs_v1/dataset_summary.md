# Lao Instrument Dataset Analysis Summary

## Dataset Overview

- **Total Classes**: 7
- **Class Names**: khean, khong_vong, pin, ranad, saw, sing, unknown
- **Total Files**: 1031

### File Counts by Class

| Class | File Count | Percentage |
|-------|------------|------------|
| khean | 141 | 13.7% |
| khong_vong | 160 | 15.5% |
| pin | 139 | 13.5% |
| ranad | 146 | 14.2% |
| saw | 142 | 13.8% |
| sing | 143 | 13.9% |
| unknown | 160 | 15.5% |

## Audio Duration Statistics

| Statistic | Value (seconds) |
|-----------|----------------|
| Mean | 6.90 |
| Std Dev | 1.36 |
| Min | 3.09 |
| 25% | 6.33 |
| Median | 7.00 |
| 75% | 7.60 |
| Max | 13.80 |

### Duration Statistics by Class

| Class | Mean (s) | Std Dev (s) | Min (s) | Max (s) |
|-------|----------|-------------|---------|--------|
| khean | 7.07 | 0.69 | 4.30 | 8.40 |
| khong_vong | 6.44 | 1.48 | 4.02 | 13.80 |
| pin | 7.05 | 1.28 | 5.00 | 11.20 |
| ranad | 6.45 | 1.18 | 4.40 | 9.40 |
| saw | 7.05 | 1.09 | 4.92 | 12.30 |
| sing | 6.48 | 2.12 | 3.09 | 10.04 |
| unknown | 7.78 | 0.50 | 6.80 | 9.30 |

## Unknown Class Analysis

### Unknown Class Subfolders

| Subfolder | File Count |
|-----------|------------|
| unknown-env_indoor | 25 |
| unknown-env_outdoor | 25 |
| unknown-human_speech | 35 |
| unknown-non-instru | 20 |
| unknown-non_speech | 15 |
| unknown-processed_sound | 15 |
| unknown-slience | 10 |
| unknown-unique | 15 |

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
