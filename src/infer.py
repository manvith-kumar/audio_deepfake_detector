#!/usr/bin/env python3
import argparse
import os
import torch
from .model import DeepfakeCNN
from .dataloader import compute_log_mel
import librosa
import numpy as np

def load_feature_from_wav(path, sample_rate=16000, duration=3.0):
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    target_len = int(duration * sample_rate)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    return compute_log_mel(y, sample_rate)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wav", required=True, help="Path to WAV file")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    
    device = torch.device(args.device)
    model = DeepfakeCNN(n_mels=64).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat = load_feature_from_wav(args.wav)
    xb = torch.tensor(feat).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(xb)
        prob = torch.sigmoid(logits).item()
        
    label = "FAKE" if prob > 0.5 else "REAL"
    print(f"Probability of being FAKE: {prob:.4f} -> Prediction: {label}")

if __name__ == "__main__":
    main()