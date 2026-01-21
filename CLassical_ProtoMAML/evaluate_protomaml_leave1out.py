#!/usr/bin/env python3
"""
Classical ProtoMAML Evaluation
Reports mean ± standard error over episodes
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

from utils.utils_data_fedcontrast_leave1out import (
    load_client_dataset_leave1out,
    load_images,
    transform_eval,
)
from utils.utils_plotting import (
    plot_roc_curve,
    plot_pr_curve,
    plot_score_histogram,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = (
    "/Users/sohamsarkar/Desktop/Projects/"
    "Research Projects/MAML/ML Project/Datasets/"
    "MVTech AD Dataset"
)

CLIENTS = [
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper"
]

MODEL_PATH = "runs/protomaml_leave1out/checkpoints/final_model.pth"
OUT_DIR = "runs/protomaml_leave1out/eval"

TEST_EPISODES = 300
K_SHOT = 5
Q_N = 12
Q_A = 4

DEVICE = torch.device("cpu")

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- MODEL ---------------- #

class EmbeddingNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.head(x)

# ---------------- EPISODE ---------------- #

def sample_episode(data):
    support = random.sample(data["test"]["normal"], K_SHOT)
    qn = random.sample(data["test"]["normal"], Q_N)
    qa = random.sample(data["test"]["anomaly"], Q_A)
    return support, qn, qa

def mean_se(values):
    values = np.asarray(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    return mean, se

# ---------------- EVALUATE ONE CLIENT ---------------- #

def evaluate_client(model, client, data):

    print(f"\nEvaluating → {client}")
    print(f"Held-out anomaly subtype → {data['held_out_subtype']}")

    aucs, prs, f1s = [], [], []
    pooled_scores, pooled_labels = [], []

    for _ in range(TEST_EPISODES):

        support, qn, qa = sample_episode(data)
        labels = [0]*len(qn) + [1]*len(qa)

        xs = torch.stack(
            load_images(support, transform_eval)
        ).to(DEVICE)
        xq = torch.stack(
            load_images(qn + qa, transform_eval)
        ).to(DEVICE)

        with torch.no_grad():
            z_sup = model(xs)
            proto = z_sup.mean(dim=0, keepdim=True)
            z_q = model(xq)

        # -------- Correct anomaly score --------
        scores = ((z_q - proto) ** 2).sum(dim=1).cpu().numpy()

        pooled_scores.extend(scores)
        pooled_labels.extend(labels)

        if len(set(labels)) < 2:
            continue

        aucs.append(roc_auc_score(labels, scores))
        prs.append(average_precision_score(labels, scores))

        thresholds = np.linspace(scores.min(), scores.max(), 200)
        f1s.append(
            max(
                f1_score(labels, scores >= t, zero_division=0)
                for t in thresholds
            )
        )

    auc_mean, auc_se = mean_se(aucs)
    pr_mean, pr_se   = mean_se(prs)
    f1_mean, f1_se   = mean_se(f1s)

    print(
        f"AUC = {auc_mean:.4f} ± {auc_se:.4f} | "
        f"PR = {pr_mean:.4f} ± {pr_se:.4f} | "
        f"F1 = {f1_mean:.4f} ± {f1_se:.4f}"
    )

    # -------- Plots (pooled) --------
    client_dir = os.path.join(OUT_DIR, client)
    os.makedirs(client_dir, exist_ok=True)

    pooled_scores = np.asarray(pooled_scores)
    pooled_labels = np.asarray(pooled_labels)

    plot_roc_curve(
        pooled_labels, pooled_scores,
        f"{client_dir}/roc.png"
    )
    plot_pr_curve(
        pooled_labels, pooled_scores,
        f"{client_dir}/pr.png"
    )
    plot_score_histogram(
        pooled_labels, pooled_scores,
        f"{client_dir}/score_hist.png"
    )

    preds = (pooled_scores >= np.median(pooled_scores)).astype(int)
    cm = confusion_matrix(pooled_labels, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal","Anomaly"],
        yticklabels=["Normal","Anomaly"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{client_dir}/confusion_matrix.png")
    plt.close()

    return {
        "auc_mean": auc_mean,
        "auc_se": auc_se,
        "pr_mean": pr_mean,
        "pr_se": pr_se,
        "f1_mean": f1_mean,
        "f1_se": f1_se,
        "n_episodes": len(aucs),
        "held_out": data["held_out_subtype"],
    }

# ---------------- MAIN ---------------- #

def main():

    print("Loading trained ProtoMAML model...")
    model = EmbeddingNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    results = {}

    for c in CLIENTS:
        data = load_client_dataset_leave1out(
            os.path.join(BASE_ROOT, c), seed=42
        )
        if data is None:
            print(f"Skipping {c}")
            continue

        results[c] = evaluate_client(model, c, data)

    np.save(
        f"{OUT_DIR}/results.npy",
        results
    )

    print("\nSaved results to:")
    print(f"{OUT_DIR}/results.npy")

# ---------------- RUN ---------------- #

if __name__ == "__main__":
    main()
