import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score

def probability_transform(scores, mu):
    """
    Eq (7): Exponential CDF
    mu: scaling parameter (1 / mean anomaly score of training set)
    """
    # F(x; mu) = 1 - exp(-mu * x)
    probs = 1.0 - np.exp(-mu * scores)
    probs = np.clip(probs, 0, 1)
    return probs

def evaluate_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    
    auc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    brier = brier_score_loss(y_true, y_probs)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "AUC": auc,
        "Brier": brier,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }
