import os
import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

# --- Constants ---
DEFAULT_SR = 16000
DEFAULT_DURATION = 3.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
LABEL_MAP = {"real": 0, "fake": 1}

def compute_log_mel(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Computes a standardized log-Mel spectrogram from a waveform."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = (log_S - np.mean(log_S)) / (np.std(log_S) + 1e-9)
    return log_S.astype(np.float32)

class DeepfakeDataset(Dataset):
    """
    A PyTorch Dataset for loading audio features for deepfake detection.
    It reads a metadata.csv file and loads pre-computed features if available.
    """
    def __init__(self, metadata_csv, split="train", features_manifest=None,
                 sample_rate=DEFAULT_SR, duration=DEFAULT_DURATION, n_mels=N_MELS):
        
        self.df = pd.read_csv(metadata_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        if self.df.empty:
            raise ValueError(f"No data found for split '{split}' in {metadata_csv}. Check the 'split' column.")
            
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        
        self.feature_map = {}
        if features_manifest and os.path.exists(features_manifest):
            fdf = pd.read_csv(features_manifest)
            self.feature_map = pd.Series(fdf.feature_file.values, index=fdf.audio_file).to_dict()

    def __len__(self):
        return len(self.df)

    def _load_feature(self, audio_path):
        """Loads a feature from a .npy file or computes it from the audio file."""
        normalized_audio_path = os.path.normpath(audio_path)
        if normalized_audio_path in self.feature_map:
            feature_path = self.feature_map[normalized_audio_path]
            if os.path.exists(feature_path):
                return np.load(feature_path)
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        target_len = int(self.duration * self.sample_rate)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            start = (len(y) - target_len) // 2
            y = y[start:start + target_len]
            
        return compute_log_mel(y, self.sample_rate, n_mels=self.n_mels)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row['filename']
        label_str = row['label']
        
        feature = self._load_feature(audio_path)
        label = LABEL_MAP.get(label_str, 0)
        
        feature_tensor = torch.tensor(feature).unsqueeze(0)
        label_tensor = torch.tensor(float(label), dtype=torch.float32)
        
        return feature_tensor, label_tensor