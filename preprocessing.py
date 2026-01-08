"""
EEG Preprocessing Pipeline for Imagined Speech Task
==================================================

This module provides a streamlined preprocessing pipeline for EEG data
from imagined speech experiments. The pipeline includes:
- Band-pass and notch filtering
- DC offset removal
- Basic quality control and logging
- Robust handling of multi-subject, multi-class datasets

Note: ICA artifact removal and advanced quality control have been disabled.

Author: EEG Preprocessing Expert
Date: October 2025
"""

import os
import sys
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore

import mne
from mne.preprocessing import ICA

# Try to import ICLabel, but don't fail if not available
try:
    from mne_icalabel import label_components
    ICALABEL_AVAILABLE = True
except ImportError:
    ICALABEL_AVAILABLE = False

# Suppress MNE warnings for cleaner output
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', category=FutureWarning)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

class PreprocessingConfig:
    """Configuration parameters for EEG preprocessing pipeline."""
    
    # Filtering parameters
    BANDPASS_LOW = 0.5      # Hz - High-pass filter cutoff
    BANDPASS_HIGH = 32.5    # Hz - Low-pass filter cutoff
    NOTCH_FREQ = 50.0       # Hz - Powerline frequency (UAE grid)
    NOTCH_HARMONICS = [100, 150]  # Hz - Powerline harmonics
    FILTER_METHOD = 'fir'   # Filter method ('fir' or 'iir')
    
    # ICA parameters (DISABLED)
    # ICA_N_COMPONENTS = None  # Auto-determine based on data rank
    # ICA_METHOD = 'infomax'   # ICA algorithm
    # ICA_MAX_ITER = 1000     # Maximum iterations
    # ICA_RANDOM_STATE = 42   # For reproducibility
    
    # Artifact detection thresholds (DISABLED)
    # EOG_THRESHOLD = 0.8     # ICLabel confidence for eye artifacts
    # MUSCLE_THRESHOLD = 0.8  # ICLabel confidence for muscle artifacts
    # CARDIAC_THRESHOLD = 0.8 # ICLabel confidence for cardiac artifacts
    
    # Data quality thresholds (RELAXED)
    MIN_CHANNELS = 8        # Minimum required EEG channels (adjusted for 14-channel system)
    # MAX_BAD_CHANNELS_RATIO = 0.5  # Maximum ratio of bad channels allowed (DISABLED)
    MIN_TRIAL_DURATION = 1.0      # Minimum trial duration in seconds
    
    # Output parameters
    SAVE_PREPROCESSING_REPORT = True
    # SAVE_ICA_COMPONENTS = True  # DISABLED
    EXPORT_FORMAT = 'edf'   # Output format ('edf', 'fif', 'set')


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Setup comprehensive logging for preprocessing pipeline.
    
    Parameters
    ----------
    output_dir : Path
        Directory to save log files
        
    Returns
    -------
    logger : logging.Logger
        Configured logger instance
    """
    log_dir = output_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'preprocessing_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('EEG_Preprocessing')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


class PreprocessingLogger:
    """Enhanced logging for tracking preprocessing statistics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.stats = {
            'files_processed': 0,
            'files_failed': 0,
            'channels_removed': {},
            'channels_interpolated': {},
            'ica_components_removed': {},
            'trials_dropped': [],
            'processing_times': {}
        }
    
    def log_file_start(self, filepath: str):
        """Log start of file processing."""
        self.logger.info(f"Processing file: {filepath}")
    
    def log_file_success(self, filepath: str, processing_time: float):
        """Log successful file processing."""
        self.stats['files_processed'] += 1
        self.stats['processing_times'][filepath] = processing_time
        self.logger.info(f"Successfully processed {filepath} in {processing_time:.2f}s")
    
    def log_file_failure(self, filepath: str, error: str):
        """Log file processing failure."""
        self.stats['files_failed'] += 1
        self.logger.error(f"Failed to process {filepath}: {error}")
    
    def log_channels_removed(self, filepath: str, channels: List[str]):
        """Log removed channels."""
        self.stats['channels_removed'][filepath] = channels
        if channels:
            self.logger.warning(f"Removed channels in {filepath}: {channels}")
    
    def log_channels_interpolated(self, filepath: str, channels: List[str]):
        """Log interpolated channels."""
        self.stats['channels_interpolated'][filepath] = channels
        if channels:
            self.logger.info(f"Interpolated channels in {filepath}: {channels}")
    
    def log_ica_components(self, filepath: str, removed_components: List[int], 
                          component_labels: Dict):
        """Log ICA component removal."""
        self.stats['ica_components_removed'][filepath] = {
            'components': removed_components,
            'labels': component_labels
        }
        self.logger.info(f"Removed ICA components in {filepath}: {removed_components}")
        for comp, label in component_labels.items():
            self.logger.info(f"  Component {comp}: {label}")
    
    def log_trial_dropped(self, filepath: str, reason: str):
        """Log dropped trial."""
        self.stats['trials_dropped'].append({'file': filepath, 'reason': reason})
        self.logger.warning(f"Dropped trial {filepath}: {reason}")
    
    def save_summary(self, output_dir: Path):
        """Save preprocessing summary to JSON file."""
        summary_file = output_dir / 'preprocessing_summary.json'
        
        # Add overall statistics
        self.stats['summary'] = {
            'total_files': self.stats['files_processed'] + self.stats['files_failed'],
            'success_rate': self.stats['files_processed'] / 
                          (self.stats['files_processed'] + self.stats['files_failed']) 
                          if (self.stats['files_processed'] + self.stats['files_failed']) > 0 else 0,
            'total_channels_removed': sum(len(ch) for ch in self.stats['channels_removed'].values()),
            'total_channels_interpolated': sum(len(ch) for ch in self.stats['channels_interpolated'].values()),
            'total_ica_components_removed': sum(len(comp['components']) 
                                              for comp in self.stats['ica_components_removed'].values()),
            'average_processing_time': np.mean(list(self.stats['processing_times'].values())) 
                                     if self.stats['processing_times'] else 0
        }
        
        with open(summary_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        self.logger.info(f"Preprocessing summary saved to: {summary_file}")
        
        # Log summary statistics
        self.logger.info("="*60)
        self.logger.info("PREPROCESSING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Files processed successfully: {self.stats['files_processed']}")
        self.logger.info(f"Files failed: {self.stats['files_failed']}")
        self.logger.info(f"Success rate: {self.stats['summary']['success_rate']:.2%}")
        self.logger.info(f"Total channels removed: {self.stats['summary']['total_channels_removed']}")
        self.logger.info(f"Total channels interpolated: {self.stats['summary']['total_channels_interpolated']}")
        self.logger.info(f"Total ICA components removed: {self.stats['summary']['total_ica_components_removed']}")
        self.logger.info(f"Average processing time: {self.stats['summary']['average_processing_time']:.2f}s")
        self.logger.info("="*60)


# =============================================================================
# DIRECTORY AND FILE MANAGEMENT
# =============================================================================

def scan_input_directory(input_dir: Union[str, Path]) -> Dict[str, List[Path]]:
    """
    Scan input directory and organize files by relative parent folder.

    Supports datasets where word classes live directly under the input
    directory as well as datasets organised into subject folders that
    contain the word class subdirectories.

    Parameters
    ----------
    input_dir : str or Path
        Root directory containing EEG files, optionally nested by subject
        and word class

    Returns
    -------
    files_dict : dict
        Dictionary mapping relative parent folder (e.g. "al3amil" or
        "S0/al3amil") to list of EDF files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    files_dict: Dict[str, List[Path]] = {}

    for edf_file in sorted(input_path.rglob("*.edf")):
        relative_parent = edf_file.parent.relative_to(input_path)
        key = relative_parent.as_posix()
        files_dict.setdefault(key, []).append(edf_file)

    return files_dict


def create_output_structure(input_dir: Union[str, Path], 
                          output_dir: Union[str, Path]) -> Path:
    """
    Create output directory structure mirroring input structure.
    
    Parameters
    ----------
    input_dir : str or Path
        Input directory to mirror
    output_dir : str or Path
        Output root directory
        
    Returns
    -------
    output_path : Path
        Created output directory path
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create main output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Mirror directory structure for all EDF-containing folders
    for edf_file in input_path.rglob("*.edf"):
        target_dir = output_path / edf_file.parent.relative_to(input_path)
        target_dir.mkdir(parents=True, exist_ok=True)
    
    return output_path


def get_output_filepath(input_filepath: Path, input_dir: Path, 
                       output_dir: Path, suffix: str = "_preprocessed") -> Path:
    """
    Generate output filepath maintaining directory structure.
    
    Parameters
    ----------
    input_filepath : Path
        Original file path
    input_dir : Path
        Input root directory
    output_dir : Path
        Output root directory
    suffix : str
        Suffix to add to filename
        
    Returns
    -------
    output_filepath : Path
        Generated output file path
    """
    # Get relative path from input root
    rel_path = input_filepath.relative_to(input_dir)
    
    # Create output path
    output_filepath = output_dir / rel_path
    
    # Add suffix to filename
    stem = output_filepath.stem
    suffix_name = f"{stem}{suffix}{output_filepath.suffix}"
    output_filepath = output_filepath.parent / suffix_name
    
    return output_filepath


def validate_dataset_structure(files_dict: Dict[str, List[Path]], 
                             logger: logging.Logger) -> Dict[str, List[Path]]:
    """
    Validate dataset structure and filter valid files.
    
    Parameters
    ----------
    files_dict : dict
        Dictionary of word classes and their files
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    validated_files : dict
        Dictionary of validated files
    """
    validated_files = {}
    
    for word_class, files in files_dict.items():
        valid_files = []
        
        for file_path in files:
            try:
                # Basic file validation
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                if file_path.stat().st_size == 0:
                    logger.warning(f"Empty file: {file_path}")
                    continue
                
                # Quick EDF header check
                try:
                    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
                    raw.crop(tmin=0.5, tmax=5.999)
                    if len(raw.ch_names) < PreprocessingConfig.MIN_CHANNELS:
                        logger.warning(f"Insufficient channels ({len(raw.ch_names)}) in {file_path}")
                        continue
                    
                    if raw.times[-1] < PreprocessingConfig.MIN_TRIAL_DURATION:
                        logger.warning(f"Trial too short ({raw.times[-1]:.2f}s) in {file_path}")
                        continue
                    
                    valid_files.append(file_path)
                    
                except Exception as e:
                    logger.error(f"Cannot read EDF file {file_path}: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error validating {file_path}: {e}")
                continue
        
        if valid_files:
            validated_files[word_class] = valid_files
            logger.info(f"Word class '{word_class}': {len(valid_files)} valid files")
        else:
            logger.warning(f"No valid files found for word class '{word_class}'")
    
    logger.info(f"Dataset validation complete. {len(validated_files)} word classes, "
                f"{sum(len(files) for files in validated_files.values())} total files")
    
    return validated_files


# =============================================================================
# EDF READING AND DATA VALIDATION
# =============================================================================

def load_and_validate_edf(filepath: Path, logger: logging.Logger) -> Optional[mne.io.Raw]:
    """
    Load EDF file and perform comprehensive data validation.
    
    Parameters
    ----------
    filepath : Path
        Path to EDF file
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    raw : mne.io.Raw or None
        Loaded and validated raw data, or None if validation fails
    """
    try:
        # Load EDF file
        logger.info(f"Loading EDF file: {filepath.name}")
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        # Basic validation
        logger.info(f"Data shape: {raw.get_data().shape} (channels x samples)")
        logger.info(f"Sampling frequency: {raw.info['sfreq']} Hz")
        logger.info(f"Duration: {raw.times[-1]:.2f} seconds")
        logger.info(f"Channels: {raw.ch_names}")
        
        # Check for sufficient channels
        if len(raw.ch_names) < PreprocessingConfig.MIN_CHANNELS:
            logger.error(f"Insufficient channels: {len(raw.ch_names)} < {PreprocessingConfig.MIN_CHANNELS}")
            return None
        
        # Check duration
        if raw.times[-1] < PreprocessingConfig.MIN_TRIAL_DURATION:
            logger.error(f"Trial too short: {raw.times[-1]:.2f}s < {PreprocessingConfig.MIN_TRIAL_DURATION}s")
            return None
        
        # Set channel types (assume all are EEG for now)
        # In practice, you might want to detect EOG, ECG channels automatically
        eeg_channels = [ch for ch in raw.ch_names if not ch.lower().startswith(('eog', 'ecg', 'emg'))]
        raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
        
        # Check for flat channels
        data = raw.get_data()
        flat_channels = []
        for i, ch_name in enumerate(raw.ch_names):
            if raw.get_channel_types([ch_name])[0] == 'eeg':
                ch_data = data[i]
                if np.std(ch_data) < 1e-12:  # Essentially flat
                    flat_channels.append(ch_name)
        
        if flat_channels:
            logger.warning(f"Detected flat channels: {flat_channels}")
            raw.info['bads'].extend(flat_channels)
        
        # Remove DC offset first (subtract mean from each channel)
        logger.info("Removing DC offset from each channel")
        for i, ch_name in enumerate(raw.ch_names):
            if raw.get_channel_types([ch_name])[0] == 'eeg':
                ch_data = data[i]
                data[i] = ch_data - np.mean(ch_data)
        
        # Update the raw object with DC-corrected data
        raw._data = data
        
        # QUALITY CONTROL DISABLED - No extreme value checking
        # QUALITY CONTROL DISABLED - No bad channel ratio validation
        
        # Set montage if available (10-20 system)
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage, match_case=False, on_missing='warn')
            logger.info("Applied standard 10-20 montage")
        except Exception as e:
            logger.warning(f"Could not apply montage: {e}")
        
        logger.info("EDF loading successful")
        
        return raw
        
    except Exception as e:
        logger.error(f"Failed to load EDF file {filepath}: {e}")
        return None


def interpolate_bad_channels(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Interpolate bad channels using spherical splines.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with bad channels marked
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    raw : mne.io.Raw
        Data with interpolated channels
    """
    if not raw.info['bads']:
        logger.info("No bad channels to interpolate")
        return raw
    
    logger.info(f"Interpolating {len(raw.info['bads'])} bad channels: {raw.info['bads']}")
    
    try:
        # Store original bad channels for logging
        original_bads = raw.info['bads'].copy()
        
        # Interpolate bad channels
        raw_interp = raw.copy()
        raw_interp.interpolate_bads(reset_bads=True)
        
        logger.info(f"Successfully interpolated channels: {original_bads}")
        return raw_interp
        
    except Exception as e:
        logger.error(f"Failed to interpolate bad channels: {e}")
        return raw


def detect_channel_artifacts(raw: mne.io.Raw, logger: logging.Logger) -> List[str]:
    """
    Detect channels with persistent artifacts using statistical methods.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    artifact_channels : list
        List of channel names with persistent artifacts
    """
    artifact_channels = []
    
    # Get EEG channels only
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    
    # Z-score threshold for outlier detection
    z_threshold = 4.0
    
    for i, ch_name in enumerate(ch_names):
        ch_data = data[i]
        
        # Check for extreme z-scores
        z_scores = np.abs(zscore(ch_data))
        if np.sum(z_scores > z_threshold) > 0.01 * len(ch_data):  # >1% of samples
            artifact_channels.append(ch_name)
            logger.warning(f"Channel {ch_name}: {np.sum(z_scores > z_threshold)} extreme values")
        
        # Check variance compared to other channels
        var_ratios = np.var(ch_data) / np.median([np.var(data[j]) for j in range(len(ch_names)) if j != i])
        if var_ratios > 5.0 or var_ratios < 0.2:
            if ch_name not in artifact_channels:
                artifact_channels.append(ch_name)
            logger.warning(f"Channel {ch_name}: abnormal variance ratio {var_ratios:.2f}")
    
    if artifact_channels:
        logger.warning(f"Detected {len(artifact_channels)} channels with persistent artifacts: {artifact_channels}")
    else:
        logger.info("No channels with persistent artifacts detected")
    
    return artifact_channels


# =============================================================================
# FILTERING PIPELINE
# =============================================================================

def apply_bandpass_filter(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Apply band-pass filter to remove low-frequency drifts and high-frequency noise.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered data
    """
    logger.info(f"Applying band-pass filter: {PreprocessingConfig.BANDPASS_LOW}-{PreprocessingConfig.BANDPASS_HIGH} Hz")
    
    try:
        # Check Nyquist frequency
        nyquist = raw.info['sfreq'] / 2
        if PreprocessingConfig.BANDPASS_HIGH >= nyquist:
            logger.warning(f"High-pass cutoff ({PreprocessingConfig.BANDPASS_HIGH} Hz) too close to Nyquist "
                          f"frequency ({nyquist} Hz). Adjusting to {nyquist * 0.9:.1f} Hz")
            high_freq = nyquist * 0.9
        else:
            high_freq = PreprocessingConfig.BANDPASS_HIGH
        
        # Apply filter
        raw_filtered = raw.copy()
        raw_filtered.filter(
            l_freq=PreprocessingConfig.BANDPASS_LOW,
            h_freq=high_freq,
            method=PreprocessingConfig.FILTER_METHOD,
            phase='zero',
            fir_window='hamming',
            fir_design='firwin',
            verbose=False
        )
        
        logger.info(f"Band-pass filter applied successfully: {PreprocessingConfig.BANDPASS_LOW}-{high_freq:.1f} Hz")
        return raw_filtered
        
    except Exception as e:
        logger.error(f"Failed to apply band-pass filter: {e}")
        return raw


def apply_notch_filter(raw: mne.io.Raw, logger: logging.Logger) -> mne.io.Raw:
    """
    Apply notch filter to remove powerline noise and harmonics.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    raw_filtered : mne.io.Raw
        Filtered data
    """
    # Determine frequencies to filter
    notch_freqs = [PreprocessingConfig.NOTCH_FREQ] + PreprocessingConfig.NOTCH_HARMONICS
    
    # Filter only frequencies below Nyquist
    nyquist = raw.info['sfreq'] / 2
    valid_freqs = [f for f in notch_freqs if f < nyquist * 0.95]
    
    if not valid_freqs:
        logger.warning("No valid notch frequencies below Nyquist frequency")
        return raw
    
    logger.info(f"Applying notch filter at frequencies: {valid_freqs} Hz")
    
    try:
        raw_filtered = raw.copy()
        raw_filtered.notch_filter(
            freqs=valid_freqs,
            method=PreprocessingConfig.FILTER_METHOD,
            phase='zero',
            fir_window='hamming',
            fir_design='firwin',
            verbose=False
        )
        
        logger.info(f"Notch filter applied successfully at {valid_freqs} Hz")
        return raw_filtered
        
    except Exception as e:
        logger.error(f"Failed to apply notch filter: {e}")
        return raw


def check_filter_quality(raw_before: mne.io.Raw, raw_after: mne.io.Raw, 
                        logger: logging.Logger) -> Dict[str, float]:
    """
    Assess filtering quality by comparing power spectra before and after filtering.
    
    Parameters
    ----------
    raw_before : mne.io.Raw
        Data before filtering
    raw_after : mne.io.Raw
        Data after filtering
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    quality_metrics : dict
        Dictionary containing quality metrics
    """
    try:
        # Pick EEG channels
        picks = mne.pick_types(raw_before.info, eeg=True, exclude='bads')
        
        # Compute PSDs
        psd_before, freqs = mne.time_frequency.psd_welch(
            raw_before, picks=picks, fmin=0.5, fmax=100, verbose=False
        )
        psd_after, _ = mne.time_frequency.psd_welch(
            raw_after, picks=picks, fmin=0.5, fmax=100, verbose=False
        )
        
        # Calculate metrics
        power_reduction_50hz = np.mean(psd_before[:, np.abs(freqs - 50) < 1]) / \
                              np.mean(psd_after[:, np.abs(freqs - 50) < 1])
        
        power_preservation_alpha = np.mean(psd_after[:, (freqs >= 8) & (freqs <= 12)]) / \
                                  np.mean(psd_before[:, (freqs >= 8) & (freqs <= 12)])
        
        total_power_ratio = np.mean(psd_after) / np.mean(psd_before)
        
        quality_metrics = {
            'power_reduction_50hz': power_reduction_50hz,
            'power_preservation_alpha': power_preservation_alpha,
            'total_power_ratio': total_power_ratio
        }
        
        logger.info(f"Filter quality metrics:")
        logger.info(f"  50Hz power reduction: {power_reduction_50hz:.2f}x")
        logger.info(f"  Alpha band preservation: {power_preservation_alpha:.2f}")
        logger.info(f"  Total power ratio: {total_power_ratio:.2f}")
        
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Failed to assess filter quality: {e}")
        return {}


# =============================================================================
# ICA PREPROCESSING
# =============================================================================

def prepare_ica_data(files_dict: Dict[str, List[Path]], logger: logging.Logger) -> Tuple[mne.io.Raw, Dict]:
    """
    Prepare concatenated data for ICA by combining multiple trials.
    
    For robust ICA, we need sufficient data. This function concatenates
    trials from the same subject to create a longer continuous recording.
    
    Parameters
    ----------
    files_dict : dict
        Dictionary of word classes and their files
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    concatenated_raw : mne.io.Raw
        Concatenated raw data for ICA
    file_info : dict
        Information about file contributions to concatenated data
    """
    logger.info("Preparing data for ICA decomposition")
    
    raw_list = []
    file_info = {}
    total_duration = 0
    
    # Load and filter each file
    for word_class, files in files_dict.items():
        for filepath in files:
            logger.info(f"Loading {filepath.name} for ICA preparation")
            
            # Load and validate
            raw = load_and_validate_edf(filepath, logger)
            if raw is None:
                continue
            
            # Apply filters for ICA
            raw_filtered = apply_bandpass_filter(raw, logger)
            raw_filtered = apply_notch_filter(raw_filtered, logger)
            
            # Interpolate bad channels
            raw_filtered = interpolate_bad_channels(raw_filtered, logger)
            
            raw_list.append(raw_filtered)
            file_info[str(filepath)] = {
                'word_class': word_class,
                'duration': raw_filtered.times[-1],
                'n_channels': len(raw_filtered.ch_names)
            }
            total_duration += raw_filtered.times[-1]
    
    if not raw_list:
        raise RuntimeError("No valid files found for ICA preparation")
    
    logger.info(f"Concatenating {len(raw_list)} files for ICA (total duration: {total_duration:.1f}s)")
    
    # Concatenate all raw data
    concatenated_raw = mne.concatenate_raws(raw_list, preload=True)
    
    logger.info(f"ICA data prepared: {concatenated_raw.get_data().shape} (channels x samples)")
    logger.info(f"Total duration: {concatenated_raw.times[-1]:.1f} seconds")
    
    return concatenated_raw, file_info


def run_ica_decomposition(raw: mne.io.Raw, logger: logging.Logger) -> ICA:
    """
    Perform ICA decomposition on the prepared data.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Prepared raw data for ICA
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    logger.info("Starting ICA decomposition")
    
    # Determine number of components
    n_components = PreprocessingConfig.ICA_N_COMPONENTS
    if n_components is None:
        # Use data rank (accounting for interpolated channels)
        rank = mne.compute_rank(raw, tol=1e-6, tol_kind='relative')
        n_components = rank['eeg']
        logger.info(f"Auto-determined ICA components: {n_components} (based on data rank)")
    
    # Initialize ICA
    ica = ICA(
        n_components=n_components,
        method=PreprocessingConfig.ICA_METHOD,
        max_iter=PreprocessingConfig.ICA_MAX_ITER,
        random_state=PreprocessingConfig.ICA_RANDOM_STATE,
        verbose=False
    )
    
    # Fit ICA
    logger.info(f"Fitting ICA with {n_components} components using {PreprocessingConfig.ICA_METHOD} method")
    start_time = datetime.now()
    
    try:
        ica.fit(raw, picks='eeg', verbose=False)
        
        fit_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"ICA decomposition completed in {fit_time:.1f} seconds")
        logger.info(f"Explained variance: {ica.pca_explained_variance_[:5]} (first 5 components)")
        
        return ica
        
    except Exception as e:
        logger.error(f"ICA decomposition failed: {e}")
        raise


def save_ica_report(ica: ICA, raw: mne.io.Raw, output_dir: Path, 
                   subject_id: str, logger: logging.Logger):
    """
    Save ICA decomposition report and plots.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
    output_dir : Path
        Output directory
    subject_id : str
        Subject identifier
    logger : logging.Logger
        Logger instance
    """
    if not PreprocessingConfig.SAVE_ICA_COMPONENTS:
        return
    
    try:
        ica_dir = output_dir / 'ica_reports'
        ica_dir.mkdir(exist_ok=True)
        
        # Save ICA object
        ica_file = ica_dir / f'{subject_id}_ica.fif'
        ica.save(ica_file, overwrite=True)
        logger.info(f"ICA object saved to: {ica_file}")
        
        # Generate and save component plots (first 20 components)
        n_components_plot = min(20, ica.n_components_)
        
        # Component topographies
        fig_topo = ica.plot_components(picks=range(n_components_plot), show=False)
        topo_file = ica_dir / f'{subject_id}_ica_components.png'
        fig_topo.savefig(topo_file, dpi=150, bbox_inches='tight')
        plt.close(fig_topo)
        
        # Component time series
        fig_sources = ica.plot_sources(raw, picks=range(min(10, n_components_plot)), 
                                     start=0, stop=30, show=False)  # First 30 seconds
        sources_file = ica_dir / f'{subject_id}_ica_sources.png'
        fig_sources.savefig(sources_file, dpi=150, bbox_inches='tight')
        plt.close(fig_sources)
        
        logger.info(f"ICA component plots saved to: {ica_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to save ICA report: {e}")


# =============================================================================
# ARTIFACT DETECTION AND REMOVAL
# =============================================================================

def detect_artifact_components_heuristics(ica: ICA, raw: mne.io.Raw, 
                                         logger: logging.Logger) -> Dict[str, List[int]]:
    """
    Detect artifact components using heuristic methods.
    
    This function uses multiple heuristics to identify different types of artifacts:
    - EOG artifacts: High correlation with frontal electrodes, specific frequency content
    - Muscle artifacts: High frequency content, peripheral electrode activity
    - Cardiac artifacts: Regular rhythmic patterns around 60-100 BPM
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    artifact_components : dict
        Dictionary mapping artifact type to component indices
    """
    logger.info("Detecting artifact components using heuristic methods")
    
    artifact_components = {
        'eog': [],
        'muscle': [],
        'cardiac': [],
        'other': []
    }
    
    # Get ICA components and their properties
    sources = ica.get_sources(raw)
    components_data = sources.get_data()
    
    for comp_idx in range(ica.n_components_):
        comp_data = components_data[comp_idx]
        
        # Get component topography
        comp_topo = ica.mixing_matrix_[:, comp_idx]
        
        # Detect EOG artifacts
        if _is_eog_component(comp_data, comp_topo, raw, logger):
            artifact_components['eog'].append(comp_idx)
            continue
        
        # Detect muscle artifacts
        if _is_muscle_component(comp_data, comp_topo, raw, logger):
            artifact_components['muscle'].append(comp_idx)
            continue
        
        # Detect cardiac artifacts
        if _is_cardiac_component(comp_data, raw, logger):
            artifact_components['cardiac'].append(comp_idx)
            continue
    
    # Log results
    total_artifacts = sum(len(comps) for comps in artifact_components.values())
    logger.info(f"Detected {total_artifacts} artifact components:")
    for artifact_type, components in artifact_components.items():
        if components:
            logger.info(f"  {artifact_type.upper()}: {components}")
    
    return artifact_components


def _is_eog_component(comp_data: np.ndarray, comp_topo: np.ndarray, 
                     raw: mne.io.Raw, logger: logging.Logger) -> bool:
    """Check if component is EOG-related."""
    # Look for frontal topography (high weights in frontal electrodes)
    frontal_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'AF3', 'AF4']
    frontal_indices = []
    
    # Only use channel indices that exist in the topography array
    for ch in frontal_channels:
        try:
            idx = raw.ch_names.index(ch)
            # Check if this index is valid for the topography array
            if idx < len(comp_topo):
                frontal_indices.append(idx)
        except ValueError:
            continue
    
    if frontal_indices:
        frontal_weights = np.abs(comp_topo[frontal_indices])
        max_frontal = np.max(frontal_weights)
        mean_weight = np.mean(np.abs(comp_topo))
        
        # Strong frontal topography
        if max_frontal > 2 * mean_weight:
            # Check frequency content (EOG typically < 10 Hz)
            freqs, psd = signal.welch(comp_data, fs=raw.info['sfreq'], nperseg=2048)
            low_freq_power = np.sum(psd[freqs < 10]) / np.sum(psd)
            
            if low_freq_power > 0.7:  # >70% power below 10 Hz
                return True
    
    return False


def _is_muscle_component(comp_data: np.ndarray, comp_topo: np.ndarray, 
                        raw: mne.io.Raw, logger: logging.Logger) -> bool:
    """Check if component is muscle-related."""
    # Muscle artifacts typically have high frequency content and peripheral topography
    temporal_channels = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8']
    temporal_indices = []
    
    # Only use channel indices that exist in the topography array
    for ch in temporal_channels:
        try:
            idx = raw.ch_names.index(ch)
            # Check if this index is valid for the topography array
            if idx < len(comp_topo):
                temporal_indices.append(idx)
        except ValueError:
            continue
    
    # Check frequency content (muscle activity typically > 20 Hz)
    freqs, psd = signal.welch(comp_data, fs=raw.info['sfreq'], nperseg=2048)
    high_freq_power = np.sum(psd[freqs > 20]) / np.sum(psd)
    
    if high_freq_power > 0.4:  # >40% power above 20 Hz
        # Check for peripheral topography
        if temporal_indices:
            temporal_weights = np.abs(comp_topo[temporal_indices])
            max_temporal = np.max(temporal_weights)
            mean_weight = np.mean(np.abs(comp_topo))
            
            if max_temporal > 1.5 * mean_weight:
                return True
    
    return False


def _is_cardiac_component(comp_data: np.ndarray, raw: mne.io.Raw, 
                         logger: logging.Logger) -> bool:
    """Check if component is cardiac-related."""
    # Look for regular rhythmic activity in the cardiac frequency range (50-120 BPM)
    # Convert to Hz: 50-120 BPM = 0.83-2.0 Hz
    
    freqs, psd = signal.welch(comp_data, fs=raw.info['sfreq'], nperseg=4096)
    cardiac_band = (freqs >= 0.8) & (freqs <= 2.2)
    cardiac_power = np.sum(psd[cardiac_band])
    total_power = np.sum(psd[freqs < 10])  # Consider low frequencies only
    
    if cardiac_power / total_power > 0.3:  # >30% power in cardiac band
        # Check for regularity using autocorrelation
        autocorr = np.correlate(comp_data, comp_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Look for peaks corresponding to heart rate
        fs = raw.info['sfreq']
        hr_samples_60 = int(fs * 60 / 60)    # 60 BPM
        hr_samples_120 = int(fs * 60 / 120)  # 120 BPM
        
        if len(autocorr) > hr_samples_60:
            hr_region = autocorr[hr_samples_120:hr_samples_60]
            if np.max(hr_region) > 0.3 * autocorr[0]:  # Strong periodic component
                return True
    
    return False


def detect_artifact_components_iclabel(ica: ICA, raw: mne.io.Raw, 
                                     logger: logging.Logger) -> Dict[str, List[int]]:
    """
    Detect artifact components using ICLabel (if available).
    
    ICLabel is an automatic component classifier that uses machine learning
    to identify artifact components.
    
    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.io.Raw
        Raw data used for ICA
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    artifact_components : dict
        Dictionary mapping artifact type to component indices
    """
    if not ICALABEL_AVAILABLE:
        logger.warning("ICLabel not available, using heuristic methods")
        return detect_artifact_components_heuristics(ica, raw, logger)
    
    try:
        logger.info("Using ICLabel for automatic component classification")
        
        # Label components
        component_labels = label_components(raw, ica, method='iclabel')
        
        artifact_components = {
            'eog': [],
            'muscle': [],
            'cardiac': [],
            'other': []
        }
        
        # Extract components based on confidence thresholds
        predictions = component_labels['predictions']
        
        for comp_idx in range(ica.n_components_):
            probs = predictions[comp_idx]
            
            # ICLabel classes: ['brain', 'muscle', 'eye', 'heart', 'line_noise', 'channel_noise', 'other']
            if probs[2] > PreprocessingConfig.EOG_THRESHOLD:  # eye
                artifact_components['eog'].append(comp_idx)
            elif probs[1] > PreprocessingConfig.MUSCLE_THRESHOLD:  # muscle
                artifact_components['muscle'].append(comp_idx)
            elif probs[3] > PreprocessingConfig.CARDIAC_THRESHOLD:  # heart
                artifact_components['cardiac'].append(comp_idx)
            elif probs[4] > 0.8 or probs[5] > 0.8:  # line_noise or channel_noise
                artifact_components['other'].append(comp_idx)
        
        logger.info("ICLabel classification completed")
        return artifact_components
        
    except Exception as e:
        logger.warning(f"ICLabel failed ({e}), using heuristic methods")
        return detect_artifact_components_heuristics(ica, raw, logger)


def apply_ica_correction(raw: mne.io.Raw, ica: ICA, 
                        artifact_components: Dict[str, List[int]], 
                        logger: logging.Logger) -> mne.io.Raw:
    """
    Apply ICA correction by removing artifact components.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data to correct
    ica : mne.preprocessing.ICA
        Fitted ICA object
    artifact_components : dict
        Dictionary of artifact components to remove
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    raw_corrected : mne.io.Raw
        ICA-corrected data
    """
    # Flatten all artifact components
    exclude_components = []
    for artifact_type, components in artifact_components.items():
        exclude_components.extend(components)
    
    if not exclude_components:
        logger.info("No artifact components to remove")
        return raw.copy()
    
    logger.info(f"Removing {len(exclude_components)} ICA components: {sorted(exclude_components)}")
    
    # Apply ICA correction
    ica.exclude = exclude_components
    raw_corrected = ica.apply(raw.copy())
    
    logger.info("ICA correction applied successfully")
    return raw_corrected


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_single_file(filepath: Path, output_dir: Path, 
                          input_dir: Path, preprocessing_logger: PreprocessingLogger) -> bool:
    """
    Preprocess a single EDF file using the fitted ICA.
    
    Parameters
    ----------
    filepath : Path
        Path to EDF file to preprocess
    ica : mne.preprocessing.ICA
        Fitted ICA object
    output_dir : Path
        Output directory
    input_dir : Path
        Input directory root
    preprocessing_logger : PreprocessingLogger
        Preprocessing logger instance
        
    Returns
    -------
    success : bool
        True if preprocessing succeeded
    """
    start_time = datetime.now()
    
    try:
        preprocessing_logger.log_file_start(str(filepath))
        
        # Load and validate EDF
        raw = load_and_validate_edf(filepath, preprocessing_logger.logger)
        if raw is None:
            preprocessing_logger.log_trial_dropped(str(filepath), "Failed validation")
            return False
        
        # Store original data for comparison
        raw_original = raw.copy()
        
        # Apply filters
        raw_filtered = apply_bandpass_filter(raw, preprocessing_logger.logger)
        raw_filtered = apply_notch_filter(raw_filtered, preprocessing_logger.logger)
        
        # Check filter quality
        filter_metrics = check_filter_quality(raw_original, raw_filtered, preprocessing_logger.logger)
        
        # ICA DISABLED - Skip artifact removal
        # Interpolate bad channels (if any exist)
        if raw_filtered.info['bads']:
            preprocessing_logger.log_channels_interpolated(str(filepath), raw_filtered.info['bads'].copy())
            raw_filtered = interpolate_bad_channels(raw_filtered, preprocessing_logger.logger)
        
        # Final cleaned data (no ICA correction)
        raw_clean = raw_filtered
        
        # Generate output filepath
        output_filepath = get_output_filepath(filepath, input_dir, output_dir)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed data
        if PreprocessingConfig.EXPORT_FORMAT.lower() == 'edf':
            try:
                # Ensure edfio is available for EDF export
                import edfio
                raw_clean.export(output_filepath, fmt='edf', overwrite=True)
                
            except ImportError as e:
                preprocessing_logger.logger.error(f"edfio not available for EDF export: {e}")
                preprocessing_logger.logger.info("Falling back to FIF format")
                output_filepath = output_filepath.with_suffix('.fif')
                raw_clean.save(output_filepath, overwrite=True)
            except Exception as e:
                preprocessing_logger.logger.error(f"EDF export failed: {e}")
                preprocessing_logger.logger.info("Falling back to FIF format")
                output_filepath = output_filepath.with_suffix('.fif')
                raw_clean.save(output_filepath, overwrite=True)
        elif PreprocessingConfig.EXPORT_FORMAT.lower() == 'fif':
            output_filepath = output_filepath.with_suffix('.fif')
            raw_clean.save(output_filepath, overwrite=True)
        else:
            preprocessing_logger.logger.warning(f"Unsupported export format: {PreprocessingConfig.EXPORT_FORMAT}")
            output_filepath = output_filepath.with_suffix('.fif')
            raw_clean.save(output_filepath, overwrite=True)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        preprocessing_logger.log_file_success(str(filepath), processing_time)
        
        preprocessing_logger.logger.info(f"Saved preprocessed file: {output_filepath}")
        
        return True
        
    except Exception as e:
        preprocessing_logger.log_file_failure(str(filepath), str(e))
        return False


def run_preprocessing_pipeline(input_dir: Union[str, Path], 
                             output_dir: Union[str, Path]) -> Dict:
    """
    Run the complete EEG preprocessing pipeline.
    
    This is the main function that orchestrates the streamlined preprocessing workflow:
    1. Scan and validate input files
    2. Process each file individually with filtering
    3. Generate comprehensive reports
    
    Note: ICA artifact removal has been disabled.
    
    Parameters
    ----------
    input_dir : str or Path
        Input directory containing word class subdirectories with EDF files
    output_dir : str or Path
        Output directory for preprocessed files
        
    Returns
    -------
    results : dict
        Dictionary containing processing results and statistics
    """
    print("="*80)
    print("EEG PREPROCESSING PIPELINE FOR IMAGINED SPEECH (SIMPLIFIED)")
    print("="*80)
    
    # Convert to Path objects
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Setup logging
    logger = setup_logging(output_path)
    preprocessing_logger = PreprocessingLogger(logger)
    
    logger.info(f"Starting preprocessing pipeline")
    logger.info(f"Input directory: {input_path}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Configuration: {PreprocessingConfig.__dict__}")
    
    try:
        # Step 1: Scan and validate dataset
        logger.info("Step 1: Scanning and validating dataset...")
        files_dict = scan_input_directory(input_path)
        validated_files = validate_dataset_structure(files_dict, logger)
        
        if not validated_files:
            raise RuntimeError("No valid files found in input directory")
        
        # Create output directory structure
        create_output_structure(input_path, output_path)
        
        # ICA STEPS DISABLED - Skip to direct file processing
        # Step 2: Process individual files
        logger.info("Step 2: Processing individual files (ICA DISABLED)...")
        total_files = sum(len(files) for files in validated_files.values())
        processed_files = 0

        for relative_folder, files in validated_files.items():
            logger.info(f"Processing folder: {relative_folder}")

            for filepath in files:
                success = preprocess_single_file(
                    filepath, output_path, input_path, preprocessing_logger
                )
                if success:
                    processed_files += 1

                # Progress update
                progress = (processed_files / total_files) * 100
                logger.info(f"Progress: {processed_files}/{total_files} ({progress:.1f}%)")
        
        # Step 3: Generate final report
        logger.info("Step 3: Generating final report...")
        preprocessing_logger.save_summary(output_path)
        
        # Prepare results
        results = {
            'success': True,
            'total_files': total_files,
            'processed_files': processed_files,
            'failed_files': total_files - processed_files,
            'output_directory': str(output_path),
            'statistics': preprocessing_logger.stats
        }
        
        logger.info("="*80)
        logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Processed {processed_files}/{total_files} files successfully")
        logger.info(f"Results saved to: {output_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        preprocessing_logger.save_summary(output_path)
        
        results = {
            'success': False,
            'error': str(e),
            'statistics': preprocessing_logger.stats
        }
        
        return results


def main():
    """
    Main entry point for the preprocessing pipeline.
    
    Example usage:
    python preprocessing.py
    """
    # Configuration
    input_directory = r"c:\Users\Hi\Documents\GitHub\EEG Preprocessing\Preliminary"
    output_directory = r"c:\Users\Hi\Documents\GitHub\EEG Preprocessing\Preliminary_Preprocessed_Simplified"
    
    # Run preprocessing
    results = run_preprocessing_pipeline(input_directory, output_directory)
    
    if results['success']:
        print(f"\nPreprocessing completed successfully!")
        print(f"Processed {results['processed_files']}/{results['total_files']} files")
        print(f"Output directory: {results['output_directory']}")
    else:
        print(f"\nPreprocessing failed: {results['error']}")
    
    return results


if __name__ == "__main__":
    # Import matplotlib for ICA plotting (only when running as main)
    import matplotlib.pyplot as plt
    plt.style.use('default')
    
    results = main()
