#!/usr/bin/env python3
"""
Evaluation: Classical MAML for Anomaly Detection
Reports mean ± standard error
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import higher
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

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
from utils.utils_eval_stats import aggregate_with_se

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

MODEL_PATH = "runs/maml_leave1out/checkpoints/final_model.pth"
OUT_DIR = "runs/maml_leave1out/eval"

DEVICE = torch.device("cpu")

TEST_EPISODES = 300
INNER_STEPS = 1
INNER_LR = 5e-4

K_SHOT = 5
Q_N = 12
Q_A = 4

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- MODEL ---------------- #

class MAMLNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        z = self.features(x).view(x.size(0), -1)
        return self.classifier(z).squeeze(-1)

# ---------------- EPISODE SAMPLING ---------------- #

def sample_episode(data):
    support = random.sample(data["test"]["normal"], K_SHOT)
    qn = random.sample(data["test"]["normal"], Q_N)
    qa = random.sample(data["test"]["anomaly"], Q_A)
    return support, qn, qa

# ---------------- EVALUATE ONE CLIENT ---------------- #

def evaluate_client(model, client):

    print(f"\nEvaluating → {client}")

    data = load_client_dataset_leave1out(
        os.path.join(BASE_ROOT, client), seed=42
    )
    if data is None:
        print("  Skipped (only one anomaly subtype)")
        return None

    aucs, prs, f1s = [], [], []
    all_scores, all_labels = [], []

    for _ in range(TEST_EPISODES):

        support, qn, qa = sample_episode(data)
        labels = np.array([0]*len(qn) + [1]*len(qa))

        xs = torch.stack(
            load_images(support, transform_eval)
        ).to(DEVICE)
        xq = torch.stack(
            load_images(qn + qa, transform_eval)
        ).to(DEVICE)

        inner_opt = torch.optim.SGD(
            model.parameters(), lr=INNER_LR
        )
        bce = nn.BCEWithLogitsLoss()

        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=True
        ) as (fmodel, diffopt):

            # ----- Inner loop -----
            for _ in range(INNER_STEPS):
                logits_sup = fmodel(xs)
                loss_inner = bce(
                    logits_sup,
                    torch.zeros_like(logits_sup)
                )
                diffopt.step(loss_inner)

            # ----- Scoring -----
            with torch.no_grad():
                logits_q = fmodel(xq)

        scores = logits_q.cpu().numpy()

        if len(np.unique(labels)) > 1:
            aucs.append(roc_auc_score(labels, scores))
            prs.append(average_precision_score(labels, scores))

            thresholds = np.linspace(scores.min(), scores.max(), 200)
            f1s.append(
                max(
                    f1_score(labels, scores >= t, zero_division=0)
                    for t in thresholds
                )
            )

        all_scores.extend(scores)
        all_labels.extend(labels)

    auc_m, auc_se = aggregate_with_se(aucs)
    pr_m, pr_se = aggregate_with_se(prs)
    f1_m, f1_se = aggregate_with_se(f1s)

    print(
        f"AUC = {auc_m:.4f} ± {auc_se:.4f} | "
        f"PR = {pr_m:.4f} ± {pr_se:.4f} | "
        f"F1 = {f1_m:.4f} ± {f1_se:.4f}"
    )

    # ----- Plots -----
    plot_roc_curve(
        all_labels,
        all_scores,
        f"{OUT_DIR}/{client}_roc.png"
    )
    plot_pr_curve(
        all_labels,
        all_scores,
        f"{OUT_DIR}/{client}_pr.png"
    )
    plot_score_histogram(
        all_labels,
        all_scores,
        f"{OUT_DIR}/{client}_hist.png"
    )

    preds = (np.array(all_scores) >= 0).astype(int)
    cm = confusion_matrix(all_labels, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal","Anomaly"],
        yticklabels=["Normal","Anomaly"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{client}_cm.png")
    plt.close()

    return {
        "auc_mean": auc_m,
        "auc_se": auc_se,
        "pr_mean": pr_m,
        "pr_se": pr_se,
        "f1_mean": f1_m,
        "f1_se": f1_se,
    }

# ---------------- MAIN ---------------- #

def main():

    print("Loading trained Classical MAML model...")
    model = MAMLNet().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.eval()

    results = {}

    for client in CLIENTS:
        res = evaluate_client(model, client)
        if res is not None:
            results[client] = res

    np.save(
        f"{OUT_DIR}/results.npy",
        results
    )

    print("\nSaved results to:", f"{OUT_DIR}/results.npy")

if __name__ == "__main__":
    main()
