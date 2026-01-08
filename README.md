# EEG Preprocessing Pipeline for Imagined Speech

This pipeline provides a robust preprocessing solution for EEG data from imagined speech experiments, specifically designed for datasets organized by word classes.

## Dataset Status ✅

**PREPROCESSING COMPLETED SUCCESSFULLY**
- **Input**: 360 EDF files in `Pilot_2/` (raw data)
- **Output**: 360 preprocessed EDF files in `Pilot_2_Preprocessed/` (cleaned data)
- **Success Rate**: 100% (360/360 files processed)
- **Quality**: All subdirectories contain at least 10 files per word class

## Features

- **Comprehensive filtering**: Band-pass (0.5-32.5 Hz) and notch filtering (50 Hz + harmonics)
- **ICA-based artifact removal**: Automatic detection of EOG, muscle, and cardiac artifacts
- **Quality control**: Extensive validation and logging of all preprocessing steps
- **Robust handling**: Supports different trial lengths and handles missing/corrupted files gracefully
- **Detailed reporting**: Generates comprehensive logs and statistics

## Requirements

```bash
pip install mne numpy scipy matplotlib
# Optional for enhanced artifact detection:
pip install mne-icalabel
```

## Usage

### Basic Usage

```python
from preprocessing import run_preprocessing_pipeline

# Run preprocessing
results = run_preprocessing_pipeline(
    input_dir="Pilot_2",
    output_dir="Pilot_2_Preprocessed"
)
```

### Direct Execution

```bash
python preprocessing.py
```

## Pipeline Steps

1. **Dataset Validation**: Scans input directory and validates EDF files
2. **ICA Preparation**: Concatenates trials for robust ICA decomposition
3. **ICA Decomposition**: Fits ICA on concatenated data
4. **Individual Processing**: Processes each file using the fitted ICA:
   - Band-pass filtering (0.5-32.5 Hz)
   - Notch filtering (50 Hz + harmonics for UAE powerline)
   - Bad channel interpolation
   - ICA artifact removal
   - Data export
5. **Report Generation**: Creates comprehensive processing reports

## Configuration

Key parameters can be modified in the `PreprocessingConfig` class:

```python
class PreprocessingConfig:
    # Filtering
    BANDPASS_LOW = 0.5      # Hz
    BANDPASS_HIGH = 32.5    # Hz
    NOTCH_FREQ = 50.0       # Hz (UAE powerline)
    
    # ICA
    ICA_METHOD = 'infomax'
    ICA_N_COMPONENTS = None  # Auto-determine
    
    # Artifact detection thresholds
    EOG_THRESHOLD = 0.8
    MUSCLE_THRESHOLD = 0.8
    CARDIAC_THRESHOLD = 0.8
    
    # Quality control
    MIN_CHANNELS = 16
    MAX_BAD_CHANNELS_RATIO = 0.2
```

## Output Structure

```
Pilot_2_Preprocessed/
├── al3amil/
│   ├── C11_T1_W1_al3amil_preprocessed.edf
│   └── ...
├── alard/
│   └── ...
├── logs/
│   └── preprocessing_YYYYMMDD_HHMMSS.log
├── ica_reports/
│   ├── concatenated_data_ica.fif
│   ├── concatenated_data_ica_components.png
│   └── concatenated_data_ica_sources.png
└── preprocessing_summary.json
```

## Logging and Quality Control

The pipeline provides extensive logging:

- **File-level logs**: Processing status for each file
- **Channel logs**: Removed/interpolated channels
- **ICA logs**: Removed components and their classifications
- **Quality metrics**: Filter effectiveness, processing times
- **Summary statistics**: Overall preprocessing success rates

## Artifact Detection

The pipeline uses multiple methods for artifact detection:

1. **ICLabel** (if available): Machine learning-based component classification
2. **Heuristic methods**: Rule-based detection using:
   - Topographical patterns
   - Frequency content analysis
   - Temporal characteristics

### Detected Artifacts:
- **EOG (Eye movements)**: Frontal topography + low-frequency content
- **Muscle**: High-frequency content + temporal/peripheral topography  
- **Cardiac**: Regular rhythmic patterns in 0.8-2.2 Hz range
- **Line noise**: 50 Hz and harmonics

## Data Quality Safeguards

- **Bad channel detection**: Flat channels, extreme values, statistical outliers
- **Bad channel interpolation**: Spherical spline interpolation
- **Trial validation**: Minimum duration, sufficient channels
- **Filter quality assessment**: Power spectral analysis before/after filtering

## Error Handling

- Graceful handling of corrupted/missing files
- Automatic fallback from ICLabel to heuristic methods
- Comprehensive error logging with specific failure reasons
- Partial processing support (continues with remaining files if some fail)

## Example Results

After preprocessing, you can expect:
- Removal of powerline noise (50 Hz + harmonics)
- Clean EEG signals free from eye, muscle, and cardiac artifacts
- Preserved neural signals in relevant frequency bands
- Detailed logs for reproducibility and quality assessment

## Troubleshooting

### Common Issues:

1. **"No valid files found"**: Check EDF file format and minimum channel requirements
2. **"ICA failed"**: Ensure sufficient data duration (concatenated trials should be >60s)
3. **"Too many bad channels"**: Check electrode impedances and recording quality

### Performance Tips:

- For large datasets, consider processing subsets separately
- Monitor memory usage with many long trials
- Use SSD storage for faster I/O with large EDF files

## Citation

This preprocessing pipeline follows best practices from:
- MNE-Python documentation
- Jas et al. (2018). "Autoreject: Automated artifact rejection for MEG and EEG data"
- Pion-Tonachini et al. (2019). "ICLabel: An automated electroencephalographic independent component classifier"