import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def _ensure_dir(save_path):
    """Ensure parent directory exists."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_loss_curve(train_losses, val_losses, save_path):
    _ensure_dir(save_path)

    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()





def plot_roc_curve(labels, scores, save_path):
    _ensure_dir(save_path)

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve(labels, scores, save_path):
    _ensure_dir(save_path)

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_score_histogram(labels, scores, save_path):
    _ensure_dir(save_path)

    labels = np.asarray(labels)
    scores = np.asarray(scores)

    plt.figure(figsize=(6, 4))
    plt.hist(scores[labels == 0], bins=30, alpha=0.7, label="Normal")
    plt.hist(scores[labels == 1], bins=30, alpha=0.7, label="Anomaly")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
