#!/usr/bin/env python3
import argparse
import os
import glob
import hashlib
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# --- Helper Functions ---
def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def write_wav(path, y, sr):
    sf.write(path, y.astype(np.float32), sr, subtype='PCM_16')

def compute_log_mel(y, sr, n_mels=64, n_fft=1024, hop_length=256):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = (log_S - np.mean(log_S)) / (np.std(log_S) + 1e-9)
    return log_S.astype(np.float32)

def synthesize_demo(raw_dir, n=16, sr=16000):
    ensure_dir(raw_dir)
    dur = 3.0
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    for i in range(n):
        freqs = [220 + i*10, 440 + i*5, 660 + (i%3)*20]
        y = sum(0.25*np.sin(2*np.pi*f*t) for f in freqs)
        y = y * np.linspace(0.01, 1.0, len(t)) * 0.9 + 0.005 * np.random.randn(len(y))
        fname = os.path.join(raw_dir, f"demo_spk{(i%4)+1}_utt{i:03d}.wav")
        sf.write(fname, y.astype(np.float32), sr)
    print(f"Synthesized {n} demo wavs into {raw_dir}")

# --- Main Processing Function (UPDATED) ---
def process_file(path, index, out_processed_dir, out_features_dir, sample_rate, duration, raw_dir):
    # Deterministic splitting
    if index % 10 < 7:
        split = 'train'
    elif index % 10 < 9:
        split = 'val'
    else:
        split = 'test'
    
    # --- THIS IS THE KEY CHANGE ---
    # Label half the files as 'fake' to create a balanced dataset
    label = 'fake' if index % 2 == 0 else 'real'
    
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / (np.max(np.abs(y)) + 1e-9) * 0.99
    
    target_len = int(duration * sample_rate)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = max(0, (len(y) - target_len) // 2)
        y = y[start:start+target_len]

    base = Path(path).stem
    out_label_dir = os.path.join(out_processed_dir, label)
    ensure_dir(out_label_dir)
    out_wav_path = os.path.join(out_label_dir, f"{base}.wav")
    write_wav(out_wav_path, y, sample_rate)

    feat = compute_log_mel(y, sample_rate)
    ensure_dir(out_features_dir)
    out_feat_path = os.path.join(out_features_dir, f"{base}.npy")
    np.save(out_feat_path, feat)

    return {
        "filename": out_wav_path.replace(os.sep, '/'), "label": label, "split": split,
        "speaker_id": "demo_speaker", "duration_sec": f"{duration:.2f}", "sample_rate": sample_rate,
        "md5": md5_file(out_wav_path), "original_path": path.replace(os.sep, '/'), "notes": "demo_data"
    }, {
        "feature_file": out_feat_path.replace(os.sep, '/'), "audio_file": out_wav_path.replace(os.sep, '/'),
        "n_frames": feat.shape[1], "n_mels": feat.shape[0], "feature_dtype": str(feat.dtype),
        "shape": str(feat.shape), "created_at": datetime.utcnow().isoformat() + "Z"
    }

def main():
    p = argparse.ArgumentParser(description="Preprocess raw audio and extract log-mel features.")
    p.add_argument("--raw-dir", required=True, help="Directory containing raw WAVs.")
    p.add_argument("--out-processed", default="data/processed", help="Where to write standardized WAVs.")
    p.add_argument("--out-features", default="data/features", help="Where to write .npy feature files.")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--generate-demo", action="store_true", help="Synthesize demo WAVs if raw dir is empty.")
    args = p.parse_args()

    ensure_dir(args.raw_dir)
    wavs = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.wav"), recursive=True))
    if not wavs and args.generate_demo:
        synthesize_demo(args.raw_dir, n=16, sr=args.sample_rate)
        wavs = sorted(glob.glob(os.path.join(args.raw_dir, "**", "*.wav"), recursive=True))
    
    if not wavs:
        print(f"No WAV files found in {args.raw_dir}. Exiting.")
        return

    metadata_rows, feat_manifest_rows = [], []
    for i, path in enumerate(wavs):
        try:
            meta_row, feat_row = process_file(path, i, args.out_processed, args.out_features, args.sample_rate, args.duration, args.raw_dir)
            metadata_rows.append(meta_row)
            feat_manifest_rows.append(feat_row)
            print(f"Processed {path} -> Label: {meta_row['label']}, Split: {meta_row['split']}")
        except Exception as e:
            print(f"Failed to process {path}: {e}")

    meta_csv_path = os.path.join("data", "metadata.csv")
    with open(meta_csv_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
        w.writeheader()
        w.writerows(metadata_rows)
    print(f"Wrote metadata to {meta_csv_path}")

    feat_manifest_path = os.path.join(args.out_features, "manifest.csv")
    with open(feat_manifest_path, "w", newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=feat_manifest_rows[0].keys())
        w.writeheader()
        w.writerows(feat_manifest_rows)
    print(f"Wrote feature manifest to {feat_manifest_path}")

if __name__ == "__main__":
    main()