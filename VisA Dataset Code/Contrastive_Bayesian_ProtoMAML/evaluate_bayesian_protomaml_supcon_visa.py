#!/usr/bin/env python3
"""
Evaluation for Contrastive BayPrAnoMeta (ViSA):
Bayesian ProtoMAML + Supervised Contrastive Learning
CPU friendly
"""

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import seaborn as sns
import higher
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from utils.utils_niw import niw_posterior, log_student_t
from utils.utils_eval_stats import compute_episode_metrics, aggregate_with_se
from utils.utils_plotting import (
    plot_roc_curve,
    plot_pr_curve,
    plot_score_histogram,
)
from utils.utils_data_visa_unseen_anomaly import (   
    load_client_dataset_visa,                        
    load_images,
    transform_eval,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = "/.../ViSA Dataset"   

CLIENTS = [                                             
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum"
]

MODEL_PATH = "runs/bpmaml_supcon_visa/checkpoints/final_model.pth"   
OUT_DIR = "runs/bpmaml_supcon_visa/eval"                             

TEST_EPISODES = 300
INNER_STEPS = 1
INNER_LR = 5e-4

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
            nn.LayerNorm(embed_dim)
        )

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        return self.head(x)

# ---------------- EPISODE SAMPLING ---------------- #

def sample_test_episode(data):
    support = random.sample(data["test"]["normal"], K_SHOT)
    qn = random.sample(data["test"]["normal"], Q_N)
    qa = random.sample(data["test"]["anomaly"], Q_A)
    return support, qn, qa

# ---------------- EVALUATE ONE CLIENT ---------------- #

def evaluate_client(model, client):

    print(f"\nEvaluating → {client}")

    data = load_client_dataset_visa(Path(BASE_ROOT) / client, seed=42)   
    if data is None:
        print(f"Skipping {client}")
        return None

    aucs, prs, f1s = [], [], []
    all_scores, all_labels = [], []
    all_embeddings, all_names = [], []

    for _ in range(TEST_EPISODES):

        support, qn, qa = sample_test_episode(data)
        labels = [0]*len(qn) + [1]*len(qa)
        names = qn + qa

        xs = torch.stack(load_images(support, transform_eval)).to(DEVICE)
        xq = torch.stack(load_images(names, transform_eval)).to(DEVICE)

        inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)

        with higher.innerloop_ctx(
            model,
            inner_opt,
            copy_initial_weights=True
        ) as (fmodel, diffopt):

            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            inner_loss = -log_student_t(z_sup, mu, Sigma, dof).mean()
            diffopt.step(inner_loss)

            zq = fmodel(xq)

        logp_n = log_student_t(zq, mu, Sigma, dof)
        logp_a = log_student_t(
            zq,
            torch.zeros(zq.shape[1], device=DEVICE),
            100.0 * torch.eye(zq.shape[1], device=DEVICE),
            torch.tensor(2.0, device=DEVICE),
        )

        scores = (logp_a - logp_n).detach().cpu().numpy()

        auc, pr, f1 = compute_episode_metrics(scores, labels)
        aucs.append(auc)
        prs.append(pr)
        f1s.append(f1)

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_embeddings.append(zq.detach().cpu().numpy())
        all_names.extend(names)

    # ---------------- Metrics ---------------- #

    auc_m, auc_se = aggregate_with_se(aucs)
    pr_m, pr_se = aggregate_with_se(prs)
    f1_m, f1_se = aggregate_with_se(f1s)

    print(
        f"AUC = {auc_m:.4f} ± {auc_se:.4f} | "
        f"PR = {pr_m:.4f} ± {pr_se:.4f} | "
        f"F1 = {f1_m:.4f} ± {f1_se:.4f}"
    )

    # ---------------- Plots ---------------- #

    out = Path(OUT_DIR) / client
    out.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(all_labels, all_scores, out / "roc.png")
    plot_pr_curve(all_labels, all_scores, out / "pr.png")
    plot_score_histogram(all_labels, all_scores, out / "score_hist.png")

    # ---------------- Confusion Matrix (95% normal quantile) ---------------- #

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    thr = np.percentile(all_scores[all_labels == 0], 95)
    preds = (all_scores >= thr).astype(int)
    cm = confusion_matrix(all_labels, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal","Anomaly"],
        yticklabels=["Normal","Anomaly"]
    )
    plt.title(f"{client}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png")
    plt.close()

    # ---------------- FP / FN examples ---------------- #

    fps = np.where((all_labels == 0) & (preds == 1))[0][:3]
    fns = np.where((all_labels == 1) & (preds == 0))[0][:3]

    print("False Positives (Normal → Anomaly):")
    for i in fps:
        print(f"  {Path(all_names[i]).name} | GT=0 | Pred=1")

    print("False Negatives (Anomaly → Normal):")
    for i in fns:
        print(f"  {Path(all_names[i]).name} | GT=1 | Pred=0")

    # ---------------- t-SNE ---------------- #

    emb = np.vstack(all_embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6,5))
    plt.scatter(
        emb_2d[all_labels==0,0], emb_2d[all_labels==0,1],
        s=5, label="Normal"
    )
    plt.scatter(
        emb_2d[all_labels==1,0], emb_2d[all_labels==1,1],
        s=5, label="Anomaly"
    )
    plt.legend()
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title(f"{client} t-SNE")
    plt.tight_layout()
    plt.savefig(out / "tsne.png")
    plt.close()

    return {
        "auc_mean": auc_m,
        "auc_se": auc_se,
        "pr_mean": pr_m,
        "pr_se": pr_se,
        "f1_mean": f1_m,
        "f1_se": f1_se,
        "held_out_anomaly_count": data["held_out_anomaly_count"]  
    }

# ---------------- MAIN ---------------- #

def main():

    print("Loading trained model...")
    model = EmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = {}

    for c in CLIENTS:
        res = evaluate_client(model, c)
        if res is not None:
            results[c] = res

    np.save(Path(OUT_DIR) / "results.npy", results)
    print("\nSaved results to:", Path(OUT_DIR) / "results.npy")

if __name__ == "__main__":
    main()
