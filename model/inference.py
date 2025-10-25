import os
import torch
from .model import DeepfakeCNN
from .helpers import load_audio_feature

DEVICE = torch.device("cpu")
MODEL = None

def load_model(checkpoint_path="checkpoints/model_best.pth"):
    global MODEL, DEVICE
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{checkpoint_path}'")
    
    MODEL = DeepfakeCNN(n_mels=64).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    MODEL.load_state_dict(checkpoint["model_state"])
    MODEL.eval()

def get_prediction(audio_path):
    global MODEL, DEVICE
    if MODEL is None:
        raise RuntimeError("Model is not loaded. Ensure training is complete and restart the server.")

    feature = load_audio_feature(audio_path)
    feature_tensor = torch.tensor(feature).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = MODEL(feature_tensor)
        probability = torch.sigmoid(logits).item()

    prediction = "Fake" if probability > 0.5 else "Real"
    confidence = probability if prediction == "Fake" else 1 - probability

    return {"prediction": prediction, "confidence": float(f"{confidence:.4f}")}