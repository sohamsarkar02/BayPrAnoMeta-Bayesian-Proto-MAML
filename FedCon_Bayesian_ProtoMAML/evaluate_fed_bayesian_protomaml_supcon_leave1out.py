#!/usr/bin/env python3
"""
Evaluation for:
Federated Contrastive BayPrAnoMeta: Federated Contrastive Bayesian ProtoMAML
CPU friendly
"""

import os
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import higher
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve
)
from sklearn.manifold import TSNE

from utils.utils_niw import niw_posterior, log_student_t
from utils.utils_eval_stats import compute_episode_metrics, aggregate_with_se
from utils.utils_plotting import (
    plot_roc_curve,
    plot_pr_curve,
    plot_score_histogram,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = "/Users/sohamsarkar/Desktop/Projects/Research Projects/MAML/ML Project/Datasets/MVTech AD Dataset"

CLIENTS = [
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper"
]

MODEL_PATH = "runs/fed_supcon_leave1out/final_model.pth"
OUT_DIR = "runs/fed_supcon_leave1out/eval"

TEST_EPISODES = 300
INNER_LR = 5e-4
K_SHOT = 5
Q_N = 12
Q_A = 4

DEVICE = torch.device("cpu")

os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- TRANSFORMS ---------------- #

transform_eval = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

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

# ---------------- HELPERS ---------------- #

def list_images(p):
    return sorted([str(x) for x in Path(p).glob("*.png")])

def load_images(paths):
    return [transform_eval(Image.open(p).convert("RGB")) for p in paths]

def make_episode(good, anom):
    return (
        random.sample(good, K_SHOT),
        random.sample(good, Q_N),
        random.sample(anom, Q_A)
    )

# ---------------- EVALUATION ---------------- #

def evaluate_client(model, client, pr_curves):

    print(f"\nEvaluating → {client}")

    test_root = Path(BASE_ROOT) / client / "test"
    good = list_images(test_root / "good")
    anom_dirs = [d for d in test_root.iterdir() if d.is_dir() and d.name != "good"]

    if len(anom_dirs) < 2:
        print(f"Skipping {client} (only one anomaly subtype)")
        return None

    held_out = anom_dirs[-1].name
    anomaly = list_images(anom_dirs[-1])

    print(f"Held-out anomaly subtype → {held_out}")

    aucs, prs, f1s = [], [], []
    all_scores, all_labels, all_embeddings, all_paths = [], [], [], []

    for _ in range(TEST_EPISODES):

        support, qn, qa = make_episode(good, anomaly)
        labels = [0]*len(qn) + [1]*len(qa)
        paths = qn + qa

        xs = torch.stack(load_images(support))
        xq = torch.stack(load_images(paths))

        inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):

            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            diffopt.step(-log_student_t(z_sup, mu, Sigma, dof).mean())

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

    # ---------- Aggregate metrics ----------
    auc_m, auc_se = aggregate_with_se(aucs)
    pr_m, pr_se   = aggregate_with_se(prs)
    f1_m, f1_se   = aggregate_with_se(f1s)

    print(f"AUC={auc_m:.4f}±{auc_se:.4f} | PR={pr_m:.4f}±{pr_se:.4f} | F1(opt)={f1_m:.4f}±{f1_se:.4f}")

    out = Path(OUT_DIR) / client
    out.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(all_labels, all_scores, out / "roc.png")
    plot_pr_curve(all_labels, all_scores, out / "pr.png")
    plot_score_histogram(all_labels, all_scores, out / "score_hist.png")

    # ---------- Optimal-threshold confusion matrix ----------
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    f1_vals = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.nanargmax(f1_vals)
    opt_thr = thresholds[best_idx]

    preds = (np.array(all_scores) >= opt_thr).astype(int)
    cm = confusion_matrix(all_labels, preds)

    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal","Anomaly"],
        yticklabels=["Normal","Anomaly"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{client} Confusion Matrix (Optimal Threshold)")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png")
    plt.close()

    # ---------- FP / FN examples ----------
    for tag, cond in [
        ("FP", (preds==1) & (np.array(all_labels)==0)),
        ("FN", (preds==0) & (np.array(all_labels)==1))
    ]:
        idxs = np.where(cond)[0][:3]
        for i in idxs:
            print(f"{tag}: {os.path.basename(all_paths[i])} | GT={all_labels[i]} | Pred={preds[i]}")

    # ---------- t-SNE ----------
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
    plt.savefig(out / "tsne.png")
    plt.close()

    # ---------- Store PR curve ----------
    pr_curves[client] = (recall, precision)

    return {
        "auc_mean": auc_m, "auc_se": auc_se,
        "pr_mean": pr_m, "pr_se": pr_se,
        "f1_mean": f1_m, "f1_se": f1_se,
        "held_out_subtype": held_out
    }

# ---------------- MAIN ---------------- #

def main():

    model = EmbeddingNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    results = {}
    pr_curves = {}

    for c in CLIENTS:
        res = evaluate_client(model, c, pr_curves)
        if res: results[c] = res

    # ---------- Combined PR curve ----------
    plt.figure(figsize=(7,6))
    for c,(r,p) in pr_curves.items():
        plt.plot(r, p, label=c)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves of all Clients")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(OUT_DIR) / "combined_pr_curve.png")
    plt.close()

    np.save(Path(OUT_DIR) / "results.npy", results)
    print("\nSaved results.")

if __name__ == "__main__":
    main()
