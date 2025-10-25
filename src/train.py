#!/usr/bin/env python3
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataloader import DeepfakeDataset
from .model import DeepfakeCNN
from .metrics import compute_auc
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds = [], []
    for xb, yb in tqdm(loader, desc="eval", leave=False):
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds.extend(probs.tolist())
        ys.extend(yb.numpy().tolist())
    return compute_auc(ys, preds)

def save_ckpt(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def main():
    p = argparse.ArgumentParser(description="Main training script for the deepfake detection model.")
    p.add_argument("--metadata", required=True, help="Path to data/metadata.csv")
    p.add_argument("--features-manifest", default="data/features/manifest.csv", help="Path to features manifest CSV")
    p.add_argument("--out-dir", default="checkpoints", help="Directory to save model checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    
    train_ds = DeepfakeDataset(args.metadata, split="train", features_manifest=args.features_manifest)
    val_ds = DeepfakeDataset(args.metadata, split="val", features_manifest=args.features_manifest)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = DeepfakeCNN(n_mels=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, device, loss_fn)
        val_auc = evaluate(model, val_loader, device)
        t1 = time.time()
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_auc={val_auc:.4f}  time={t1-t0:.1f}s")
        
        ckpt = {"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict(), "val_auc": val_auc}
        
        if val_auc > best_auc:
            best_auc = val_auc
            save_ckpt(ckpt, os.path.join(args.out_dir, "model_best.pth"))
            
    print(f"Training complete. Best val AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()