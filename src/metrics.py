import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_auc(ys, preds):
    """Computes the Area Under the Curve (AUC) score."""
    try:
        return roc_auc_score(ys, preds)
    except ValueError:
        return 0.5

def eer_from_scores(ys, preds):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thr = roc_curve(ys, preds)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return eer, fpr[idx], thr[idx]