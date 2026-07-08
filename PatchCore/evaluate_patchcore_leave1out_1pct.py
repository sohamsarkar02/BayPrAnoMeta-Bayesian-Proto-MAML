#!/usr/bin/env python3
"""
PatchCore Evaluation Script
Greedy Coreset Selection (1%)
Reports Mean ± Standard Error (SE)
CPU-friendly
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

from utils.utils_data_fedcontrast_leave1out import (
    load_client_dataset_leave1out,
    load_images,
    transform_eval,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = (
    "/.../"
    "MVTech AD Dataset"
)

CLIENTS = [
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper"
]

# 🔹 MUST match the 1% training script
MODEL_DIR = "runs/patchcore_shared_encoder_1pct"
MEMORY_DIR = f"{MODEL_DIR}/memory_bank"
OUT_DIR = f"{MODEL_DIR}/eval_results"

DEVICE = torch.device("cpu")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP = 200   # bootstrap resamples for SE

np.random.seed(42)
torch.manual_seed(42)

# ---------------- MODEL ---------------- #

class SharedEmbeddingNet(nn.Module):
    """
    Same encoder as Bayesian ProtoMAML:
    ResNet-18 + projection head (512 -> 128)
    """
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        backbone.maxpool = nn.AvgPool2d(
            kernel_size=3, stride=2, padding=1
        )

        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.projector = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
        )

    def forward(self, x):
        feats = self.backbone(x)           # [B, 512, H, W]
        B, C, H, W = feats.shape

        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
        patches = self.projector(patches) # [B*H*W, 128]

        return patches

# ---------------- METRICS ---------------- #

def compute_metrics(scores, labels):
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    thresholds = np.unique(scores)
    f1s = [f1_score(labels, scores >= t) for t in thresholds]
    f1_opt = np.max(f1s)

    return auroc, auprc, f1_opt


def bootstrap_metrics(scores, labels, n_bootstrap=200):
    """
    Bootstrap mean ± standard error for AUROC, AUPRC, F1
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    n = len(scores)

    aurocs, auprcs, f1s = [], [], []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        s = scores[idx]
        y = labels[idx]

        # Skip degenerate bootstrap samples
        if len(np.unique(y)) < 2:
            continue

        au, ap, f1 = compute_metrics(s, y)
        aurocs.append(au)
        auprcs.append(ap)
        f1s.append(f1)

    def mean_se(x):
        x = np.array(x)
        return x.mean(), x.std(ddof=1) / np.sqrt(len(x))

    return {
        "auroc": mean_se(aurocs),
        "auprc": mean_se(auprcs),
        "f1": mean_se(f1s),
    }

# ---------------- EVALUATION ---------------- #

def evaluate_client(client):

    print(f"\nEvaluating → {client}")

    data = load_client_dataset_leave1out(
        os.path.join(BASE_ROOT, client), seed=42
    )

    if data is None:
        print(f"Skipping {client}")
        return None

    memory_path = f"{MEMORY_DIR}/{client}_memory.npy"
    memory_bank = np.load(memory_path)

    model = SharedEmbeddingNet().to(DEVICE)
    model.eval()

    scores, labels = [], []

    with torch.no_grad():
        for label, paths in [
            (0, data["test"]["normal"]),
            (1, data["test"]["anomaly"])
        ]:
            for path in paths:
                imgs = load_images([path], transform_eval)
                if len(imgs) == 0:
                    continue

                x = imgs[0].unsqueeze(0).to(DEVICE)
                patches = model(x).cpu().numpy()

                # kNN distance (k = 1)
                dists = np.min(
                    np.linalg.norm(
                        patches[:, None] - memory_bank[None],
                        axis=2
                    ),
                    axis=1
                )

                score = dists.max()
                scores.append(score)
                labels.append(label)

    stats = bootstrap_metrics(
        scores, labels, n_bootstrap=N_BOOTSTRAP
    )

    print(
        f"AUROC = {stats['auroc'][0]:.4f} ± {stats['auroc'][1]:.4f} | "
        f"AUPRC = {stats['auprc'][0]:.4f} ± {stats['auprc'][1]:.4f} | "
        f"F1 = {stats['f1'][0]:.4f} ± {stats['f1'][1]:.4f}"
    )

    return stats

# ---------------- MAIN ---------------- #

def main():

    results = {}

    for client in CLIENTS:
        res = evaluate_client(client)
        if res is not None:
            results[client] = res

    # 🔹 Distinct filename for 1% results
    save_path = f"{OUT_DIR}/patchcore_1pct_results_with_se.npy"
    np.save(save_path, results)

    print("\nSaved results to", save_path)

if __name__ == "__main__":
    main()
