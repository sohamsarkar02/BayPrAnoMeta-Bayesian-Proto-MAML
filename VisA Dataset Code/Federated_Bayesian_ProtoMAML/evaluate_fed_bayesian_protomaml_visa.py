#!/usr/bin/env python3
"""
Evaluation script for Federated BayPrAnoMeta: Federated Bayesian Proto-MAML (ViSA)
CPU friendly
"""

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import higher
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.manifold import TSNE

from utils.utils_niw import niw_posterior, log_student_t
from utils.utils_eval_stats import compute_episode_metrics, aggregate_with_se
from utils.utils_plotting import plot_roc_curve, plot_pr_curve, plot_score_histogram
from utils.utils_data_visa_unseen_anomaly import (   # CHANGED
    load_client_dataset_visa,                        # CHANGED
    load_images,
    transform_eval,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = "/Users/sohamsarkar/Desktop/ViSA Dataset"   # CHANGED

CLIENTS = [                                             # CHANGED
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

MODEL_PATH = "runs/fed_bpmaml_visa/final_model.pth"   # CHANGED
OUT_DIR = "runs/fed_bpmaml_visa/eval"                 # CHANGED

DEVICE = torch.device("cpu")

TEST_EPISODES = 300
INNER_LR = 5e-4
K_SHOT = 5
Q_N = 12
Q_A = 4

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

# ---------------- EPISODE ---------------- #

def make_episode(good, anom):
    return (
        random.sample(good, K_SHOT),
        random.sample(good, Q_N),
        random.sample(anom, Q_A)
    )

# ---------------- EVALUATE ONE CLIENT ---------------- #

def evaluate_client(model, client, pr_curves):

    print(f"\nEvaluating → {client}")

    data = load_client_dataset_visa(Path(BASE_ROOT) / client, seed=42)   # CHANGED
    if data is None:
        print(f"  Skipped (insufficient data)")
        return None

    good = data["test"]["normal"]     # CHANGED
    anom = data["test"]["anomaly"]    # CHANGED

    aucs, prs, f1s = [], [], []
    all_scores, all_labels, all_embeddings, all_paths = [], [], [], []

    for _ in range(TEST_EPISODES):

        support, qn, qa = make_episode(good, anom)
        labels = np.array([0]*len(qn) + [1]*len(qa))
        paths = qn + qa

        xs = torch.stack(load_images(support, transform_eval))
        xq = torch.stack(load_images(paths, transform_eval))

        inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):

            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            diffopt.step(-log_student_t(z_sup, mu, Sigma, dof).mean())

            with torch.no_grad():
                zq = fmodel(xq)

        logp_n = log_student_t(zq, mu, Sigma, dof)
        logp_a = log_student_t(
            zq,
            torch.zeros(zq.shape[1]),
            100.0 * torch.eye(zq.shape[1]),
            torch.tensor(2.0)
        )

        scores = (logp_a - logp_n).detach().cpu().numpy()

        auc, pr, f1 = compute_episode_metrics(scores, labels)
        aucs.append(auc); prs.append(pr); f1s.append(f1)

        all_scores.extend(scores)
        all_labels.extend(labels)
        all_embeddings.append(zq.detach().cpu().numpy())
        all_paths.extend(paths)

    # ----- Aggregate -----
    auc_m, auc_se = aggregate_with_se(aucs)
    pr_m, pr_se   = aggregate_with_se(prs)
    f1_m, f1_se   = aggregate_with_se(f1s)

    print(f"  AUROC = {auc_m:.4f} ± {auc_se:.4f}")
    print(f"  AUPRC = {pr_m:.4f} ± {pr_se:.4f}")
    print(f"  F1(opt)= {f1_m:.4f} ± {f1_se:.4f}")

    client_out = Path(OUT_DIR) / client
    client_out.mkdir(parents=True, exist_ok=True)

    # ----- PR curve -----
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    pr_curves[client] = (recall, precision)

    plot_roc_curve(all_labels, all_scores, client_out / "roc.png")
    plot_pr_curve(all_labels, all_scores, client_out / "pr.png")
    plot_score_histogram(all_labels, all_scores, client_out / "score_hist.png")

    # ----- Confusion Matrix -----
    normal_scores = np.array(all_scores)[np.array(all_labels) == 0]
    thr = np.percentile(normal_scores, 95)
    preds = (np.array(all_scores) >= thr).astype(int)

    cm = confusion_matrix(all_labels, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Anomaly"],
                yticklabels=["Normal","Anomaly"])
    plt.title(f"{client} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(client_out / "confusion_matrix.png")
    plt.close()

    # ----- FP / FN -----
    for tag, cond in [("FP", (preds==1)&(np.array(all_labels)==0)),
                      ("FN", (preds==0)&(np.array(all_labels)==1))]:
        idxs = np.where(cond)[0][:3]
        for i in idxs:
            print(f"  {tag}: {os.path.basename(all_paths[i])} | GT={all_labels[i]} | Pred={preds[i]}")

    # ----- t-SNE -----
    emb = np.vstack(all_embeddings)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(6,5))
    plt.scatter(emb_2d[np.array(all_labels)==0,0],
                emb_2d[np.array(all_labels)==0,1], s=5, label="Normal")
    plt.scatter(emb_2d[np.array(all_labels)==1,0],
                emb_2d[np.array(all_labels)==1,1], s=5, label="Anomaly")
    plt.legend()
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.title(f"{client} t-SNE")
    plt.tight_layout()
    plt.savefig(client_out / "tsne.png")
    plt.close()

    return {
        "auroc_mean": auc_m, "auroc_se": auc_se,
        "auprc_mean": pr_m, "auprc_se": pr_se,
        "f1_mean": f1_m, "f1_se": f1_se,
        "held_out_anomaly_count": data["held_out_anomaly_count"]   # CHANGED
    }

# ---------------- MAIN ---------------- #

def main():

    model = EmbeddingNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    results = {}
    pr_curves = {}

    for client in CLIENTS:
        res = evaluate_client(model, client, pr_curves)
        if res is not None:
            results[client] = res

    # ----- Combined PR curve -----
    plt.figure(figsize=(7,6))
    for c,(r,p) in pr_curves.items():
        plt.plot(r, p, label=c)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision–Recall Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "combined_pr_curve.png")
    plt.close()

    np.save(Path(OUT_DIR) / "results.npy", results)
    print("\nSaved results to:", Path(OUT_DIR) / "results.npy")

if __name__ == "__main__":
    main()