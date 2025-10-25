import librosa
import numpy as np

N_MELS = 64
DEFAULT_SR = 16000
DEFAULT_DURATION = 3.0

def compute_log_mel(y, sr, n_mels=N_MELS):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels, power=2.0)
    log_S = librosa.power_to_db(S, ref=np.max)
    log_S = (log_S - np.mean(log_S)) / (np.std(log_S) + 1e-9)
    return log_S.astype(np.float32)

def load_audio_feature(path, sample_rate=DEFAULT_SR, duration=DEFAULT_DURATION):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    target_len = int(duration * sample_rate)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    feature = compute_log_mel(y, sample_rate)
    return feature