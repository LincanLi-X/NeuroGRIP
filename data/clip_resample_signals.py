import os
import sys
import argparse
import numpy as np
import pyedflib
import h5py
from tqdm import tqdm

sys.path.append("../")
from constants import INCLUDED_CHANNELS, FREQUENCY
from data.data_utils import resampleData, getEDFsignals, getOrderedChannels

def read_edf_with_retry(file_name, max_retries=3, retry_delay=1):
    import time
    for attempt in range(max_retries):
        try:
            return pyedflib.EdfReader(file_name)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay)
    raise Exception(f"All {max_retries} retries failed for {file_name}")

def process_edf_file(edf_path, save_dir, clip_len=12, time_step_size=1):
    edf = read_edf_with_retry(edf_path)
    signal_labels = edf.getSignalLabels()

    ordered_channels = getOrderedChannels(edf_path, False, signal_labels, INCLUDED_CHANNELS)
    signals = getEDFsignals(edf)
    signal_array = np.array(signals[ordered_channels, :])
    sample_freq = edf.getSampleFrequency(0)

    if sample_freq != FREQUENCY:
        signal_array = resampleData(
            signal_array,
            to_freq=FREQUENCY,
            window_size=int(signal_array.shape[1] / sample_freq),
        )

    total_length = signal_array.shape[1]
    clip_samples = int(FREQUENCY * clip_len)

    base_name = os.path.basename(edf_path)  # e.g. 00000404_s002_t000.edf
    base_name_with_ext = base_name  # includes .edf

    num_clips = total_length // clip_samples

    for clip_idx in range(num_clips):
        start = clip_idx * clip_samples
        end = start + clip_samples
        clip_data = signal_array[:, start:end]

        if clip_data.shape[1] < clip_samples:
            continue

        steps = []
        step_size = int(FREQUENCY * time_step_size)
        for i in range(0, clip_data.shape[1], step_size):
            if i + step_size > clip_data.shape[1]:
                break
            steps.append(clip_data[:, i:i + step_size])

        if len(steps) == 0:
            continue

        eeg_clip = np.stack(steps, axis=0)

        # FIX: Correct .h5 filename format to match expected: <name>.edf_<idx>.h5
        save_name = f"{base_name_with_ext}_{clip_idx}.h5"
        save_path = os.path.join(save_dir, save_name)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with h5py.File(save_path, 'w') as hf:
            hf.create_dataset("resampled_signal", data=eeg_clip)
            hf.create_dataset("resample_freq", data=FREQUENCY)

def process_all(raw_edf_dir, save_dir, clip_len, time_step_size):
    edf_files = []
    for path, _, files in os.walk(raw_edf_dir):
        for f in files:
            if f.endswith(".edf"):
                edf_files.append(os.path.join(path, f))

    print(f"Found {len(edf_files)} .edf files.")
    for edf_path in tqdm(edf_files):
        try:
            process_edf_file(edf_path, save_dir, clip_len, time_step_size)
        except Exception as e:
            print(f"Failed to process {edf_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clip-level .h5 files from EDF")
    parser.add_argument("--raw_edf_dir", type=str, required=True, help="Directory containing EDF files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save .h5 clips")
    parser.add_argument("--clip_len", type=int, default=12, help="Clip length in seconds")
    parser.add_argument("--time_step_size", type=int, default=1, help="Size of time step per frame in seconds")
    args = parser.parse_args()

    process_all(args.raw_edf_dir, args.save_dir, args.clip_len, args.time_step_size)
