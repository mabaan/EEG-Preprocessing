# Process subjects - Extract imagination segments and create classification folders
import mne
import pandas as pd
import os
import shutil
from pathlib import Path
import sys

# Define the sentence mapping
sentences = {
    1: "المريض يشعر بالجوع",
    2: "المريض يشعر بالألم",
    3: "الممرض يحضر الطعام",
    4: "الممرض يحضر الماء",
    5: "السرير غير مريح",
    6: "الطعام غير لذيذ",
    7: "الطبيب يغادر المستشفى",
    8: "الطبيب يصف الدواء",
    9: "الصيدلي يعد الدواء",
    10: "الصيدلي يجيب الهاتف",
    11: "العامل يمسح الأرض",
    12: "العامل يرتب السرير",
}

translit_map = {
    "المريض يشعر بالجوع": "almareed_yash3ur_biljoo3",
    "المريض يشعر بالألم": "almareed_yash3ur_bilalam",
    "الممرض يحضر الطعام": "almumarid_yu7dar_alta3am",
    "الممرض يحضر الماء": "almumarid_yu7dar_alma2",
    "السرير غير مريح": "alsareer_gheir_muree7",
    "الطعام غير لذيذ": "alta3am_gheir_latheeth",
    "الطبيب يغادر المستشفى": "altabeeb_yughadir_almustashfa",
    "الطبيب يصف الدواء": "altabeeb_yasif_aldawa2",
    "الصيدلي يعد الدواء": "alsaydalee_yu3id_aldawa2",
    "الصيدلي يجيب الهاتف": "alsaydalee_yujeeb_alhatif",
    "العامل يمسح الأرض": "al3amil_yamsa7_alard",
    "العامل يرتب السرير": "al3amil_yuratib_alsareer",
}

subjects=sys.argv[1:]  # List of subject folder names passed as arguments

# Process subjects
for subject in subjects:
    print(f"\n{'='*60}")
    print(f"Processing Subject: {subject}")
    print(f"{'='*60}")
    
    root_dir = Path(subject)
    temp_imagined = Path(f"{subject}_imagined_temp")
    classification_root = Path(f"{subject}_classification")
    
    # Create temporary folder for imagined segments
    temp_imagined.mkdir(exist_ok=True)
    classification_root.mkdir(exist_ok=True)
    
    # Step 1: Extract imagination segments
    print(f"\n Step 1: Extracting imagination segments from {subject}...")
    
    for c in range(1, 13):  # C1..C12
        class_dir = root_dir / f"C{c}"
        if not class_dir.is_dir():
            continue

        for t in range(1, 11):  # T1..T10 (to handle extra trials in S3/C2)
            trial_dir = class_dir / f"T{t}"
            if not trial_dir.is_dir():
                continue

            # Find EDF and CSV files
            edf_file = None
            csv_file = None
            for f in os.listdir(trial_dir):
                if f.endswith(".edf"):
                    edf_file = trial_dir / f
                elif f.endswith(".csv"):
                    csv_file = trial_dir / f

            if edf_file is None or csv_file is None:
                continue

            try:
                # Load EDF and CSV
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                
                # Keep only EEG channels
                eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
                channels_to_keep = [ch for ch in eeg_channels if ch in raw.ch_names]
                raw.pick(channels_to_keep)
                
                events_df = pd.read_csv(csv_file)

                # Filter imagination phases (~6s)
                imagination_phases = events_df[
                    events_df['type'].str.contains("phase_Imagine")
                    & (events_df['duration'] > 5)
                    & (events_df['duration'] < 7)
                ].sort_values('latency')
                
                if len(imagination_phases) != 3:
                    print(f"⚠️  Skipping {subject}/C{c}/T{t}: found {len(imagination_phases)} phases (expected 3)")
                    continue

                # Get sentence and split into words
                arabic_sentence = sentences[c]
                full_translit = translit_map.get(arabic_sentence, f"sentence{c}")
                arabic_words = arabic_sentence.split()
                translit_words = full_translit.split("_")

                # Process each of the 3 words
                for idx, (_, phase_row) in enumerate(imagination_phases.iterrows(), start=1):
                    onset = phase_row['latency']
                    duration = phase_row['duration']
                    
                    arabic_word = arabic_words[idx-1] if idx-1 < len(arabic_words) else f"word{idx}"
                    translit = translit_words[idx-1] if idx-1 < len(translit_words) else f"word{idx}"

                    # Extract full ~6s imagination phase
                    # Note: Crop first 0.5s during model training (raw.crop(tmin=0.5)) to remove eye blink
                    word_raw = raw.copy().crop(tmin=onset, tmax=onset+duration-1e-6)

                    # Output filename
                    out_name = f"{subject}_C{c}_T{t}_W{idx}_{translit}.edf"
                    
                    # Save to temp folder organized by word
                    word_temp_dir = temp_imagined / translit
                    word_temp_dir.mkdir(exist_ok=True)
                    out_path = word_temp_dir / out_name

                    # Export EDF
                    word_raw.export(str(out_path), fmt="edf", overwrite=True)

            except Exception as e:
                print(f" Error processing {subject}/C{c}/T{t}: {e}")
                continue
    
    # Step 2: Move to classification folders
    print(f"\n Step 2: Organizing into {subject}_classification...")
    
    for word_folder in temp_imagined.iterdir():
        if word_folder.is_dir():
            word_name = word_folder.name
            dest_word_dir = classification_root / word_name
            dest_word_dir.mkdir(exist_ok=True)
            
            # Move all EDF files
            for edf_file in word_folder.glob("*.edf"):
                dest_file = dest_word_dir / edf_file.name
                shutil.copy(edf_file, dest_file)
    
    # Clean up temp folder
    shutil.rmtree(temp_imagined)
    
    # Count files per word
    print(f"\n {subject}_classification created:")
    for word_folder in sorted(classification_root.iterdir()):
        if word_folder.is_dir():
            count = len(list(word_folder.glob("*.edf")))
            print(f"   {word_folder.name}: {count} trials")

print("\n" + "="*60)
print("✅ All subjects processed successfully!")
print("="*60)