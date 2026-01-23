
# EEG Preprocessing Pipeline for Imagined Speech

This repository provides a comprehensive and robust preprocessing pipeline for EEG data collected during imagined speech experiments. The pipeline is designed to handle datasets organized by word classes and supports multi-subject, multi-trial structures. It ensures high data quality through advanced filtering, artifact removal, and extensive logging.

## Project Overview

This pipeline converts the subject folders into classification ready folder structure ready to be preprocessed.

Then it processes raw EEG data (EDF format) and outputs cleaned, artifact-reduced EEG files suitable for further analysis. It is tailored for research in imagined speech and related cognitive neuroscience applications.

## Dataset Example

The pipeline has been validated on datasets containing hundreds of EDF files, organized by word class. For example:

- **Input Directory**: Contains raw EDF files grouped by word class (e.g., `Pilot_2/word_class/subject_trial.edf`).
- **Output Directory**: Contains preprocessed EDF files with artifacts removed and quality control applied (e.g., `Pilot_2_Preprocessed/word_class/subject_trial_preprocessed.edf`).

## Key Features

- Band-pass filtering (0.5–32.5 Hz) and notch filtering (50 Hz and harmonics)
- ICA-based artifact removal (EOG, muscle, cardiac artifacts)
- Quality control and validation of all preprocessing steps
- Robust handling of variable trial lengths and missing/corrupted files
- Detailed logging and reporting for reproducibility

## Requirements

Install the required Python packages:

```bash
pip install mne numpy scipy matplotlib edfio pandas
# For enhanced artifact detection (optional):
pip install mne-icalabel
```

## Usage

### Python API

```python
from preprocessing import run_preprocessing_pipeline

# Run preprocessing
results = run_preprocessing_pipeline(
    input_dir="Pilot_2",
    output_dir="Pilot_2_Preprocessed"
)
```


### Command Line

```bash
python setup.py S1 S2 S3
python preprocessing.py --input_dir <input_directory> --output_dir <output_directory>
```

## Pipeline Workflow

1. **Dataset Validation**: Scans the input directory and validates EDF files for minimum quality requirements.
2. **ICA Preparation**: Concatenates trials to enable robust ICA decomposition (if enabled).
3. **ICA Decomposition**: Fits ICA on concatenated data to identify artifact components (if enabled).
4. **Individual File Processing**: Applies filtering, artifact removal, and bad channel interpolation to each file.
5. **Reporting**: Generates logs and summary statistics for all preprocessing steps.

## Configuration

Key parameters can be modified in the `PreprocessingConfig` class within `preprocessing.py`:

```python
class PreprocessingConfig:
    BANDPASS_LOW = 0.5      # Hz
    BANDPASS_HIGH = 32.5    # Hz
    NOTCH_FREQ = 50.0       # Hz
    ICA_METHOD = 'infomax'
    ICA_N_COMPONENTS = None  # Auto-determine
    EOG_THRESHOLD = 0.8
    MUSCLE_THRESHOLD = 0.8
    CARDIAC_THRESHOLD = 0.8
    MIN_CHANNELS = 16
    MAX_BAD_CHANNELS_RATIO = 0.2
```

## Input Directory Structure

The input directory should be organized as follows:

```
input_dir/
├── word_class_1/
│   ├── trial1.edf
│   ├── trial2.edf
│   └── ...
├── word_class_2/
│   ├── trial1.edf
│   └── ...
├── ...
```

Each EDF file should meet minimum channel and duration requirements. The pipeline will scan these subdirectories, validate the files, and process them accordingly.

## Output Directory Structure

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


## Notes

- The script now requires you to specify the input and output directories using command-line arguments:

    ```bash
    python preprocessing.py --input_dir <input_directory> --output_dir <output_directory>
    ```

- You can still use the Python API as shown above for integration into other workflows.

- Key configuration parameters can be changed in the `PreprocessingConfig` class in `preprocessing.py`.


## Logging and Quality Control

The pipeline provides detailed logging and reporting:

- File-level logs: Processing status and errors for each file
- Channel logs: Channels removed or interpolated during preprocessing
- ICA logs: Artifact components detected and removed (if enabled)
- Quality metrics: Filter effectiveness, processing times, and summary statistics

## Artifact Detection

Artifact detection is performed using both machine learning (ICLabel, if available) and heuristic methods:

- **ICLabel**: Machine learning-based component classification (optional)
- **Heuristic methods**: Rule-based detection using topographical patterns, frequency content, and temporal characteristics

Artifacts detected include:
- Eye movements (EOG): Frontal topography, low-frequency content
- Muscle: High-frequency content, peripheral topography
- Cardiac: Rhythmic patterns in 0.8–2.2 Hz range
- Line noise: 50 Hz and harmonics

## Data Quality Safeguards

- Bad channel detection: Identifies flat channels, extreme values, and statistical outliers
- Bad channel interpolation: Uses spherical spline interpolation
- Trial validation: Ensures minimum duration and sufficient channels
- Filter quality assessment: Power spectral analysis before and after filtering

## Error Handling

- Handles corrupted or missing files gracefully
- Falls back to heuristic artifact detection if ICLabel is unavailable
- Logs specific failure reasons for each file
- Supports partial processing (continues with remaining files if some fail)

## Expected Results

After preprocessing, the output EEG data will:
- Have powerline noise (50 Hz and harmonics) removed
- Be free from eye, muscle, and cardiac artifacts
- Preserve neural signals in relevant frequency bands
- Include detailed logs for reproducibility and quality assessment

## Troubleshooting and Performance Tips

Common issues and solutions:

1. "No valid files found": Check EDF file format and minimum channel requirements
2. "ICA failed": Ensure sufficient data duration (concatenated trials should be >60s)
3. "Too many bad channels": Check electrode impedances and recording quality

Performance tips:
- For large datasets, process subsets separately
- Monitor memory usage for long trials
- Use SSD storage for faster I/O

## References

This pipeline follows best practices from:
- MNE-Python documentation
- Jas et al. (2018). "Autoreject: Automated artifact rejection for MEG and EEG data"
- Pion-Tonachini et al. (2019). "ICLabel: An automated electroencephalographic independent component classifier"