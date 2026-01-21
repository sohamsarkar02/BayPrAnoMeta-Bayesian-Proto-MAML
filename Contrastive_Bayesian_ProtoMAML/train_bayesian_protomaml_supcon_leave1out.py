#!/usr/bin/env python3
"""
Contrastive BayPrAnoMeta: Bayesian ProtoMAML + Supervised Contrastive Learning
CPU friendly
"""

import os
import random
import numpy as np
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import higher
from PIL import Image

from utils.utils_niw import niw_posterior, log_student_t
from utils.utils_plotting import plot_loss_curve
from utils.utils_data_fedcontrast_leave1out import (
    load_client_dataset_leave1out,
    load_images,
    transform_train,
    transform_eval,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = "/Users/sohamsarkar/Desktop/Projects/Research Projects/MAML/ML Project/Datasets/MVTech AD Dataset"

CLIENTS = [
    "bottle","cable","capsule","carpet","grid",
    "hazelnut","leather","metal_nut","pill","screw",
    "tile","toothbrush","transistor","wood","zipper"
]

OUT_DIR = "runs/bpmaml_supcon_leave1out"
CKPT_DIR = f"{OUT_DIR}/checkpoints"
PLOT_DIR = f"{OUT_DIR}/plots"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

# Episodic parameters
K_SHOT = 5
Q_N = 12
Q_A = 4

# Optimization
EPOCHS = 50
EPISODES_PER_EPOCH = 50
VAL_EPISODES = 20

INNER_STEPS = 1
INNER_LR = 5e-4
META_LR = 1e-4

# SupCon
LAMBDA_CON = 0.1
TEMPERATURE = 0.07

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# ---------------- SUPCON LOSS ---------------- #

def supervised_contrastive_loss(z, labels, temperature=TEMPERATURE):
    z = nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    labels = labels.unsqueeze(1)
    mask = labels.eq(labels.T).float()

    logits = sim - torch.max(sim, dim=1, keepdim=True)[0]
    exp_logits = torch.exp(logits) * (1 - torch.eye(len(z), device=z.device))

    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    return -mean_log_prob_pos.mean()

# ---------------- EPISODE SAMPLING ---------------- #

def sample_episode(data):
    support = random.sample(data["train"]["normal"], K_SHOT)
    qn = random.sample(data["train"]["normal"], Q_N)
    qa = random.sample(data["train"]["anomaly"], Q_A)
    return support, qn, qa

# ---------------- TRAIN / VAL STEP ---------------- #

def run_episode(model, data, train=True):
    support, qn, qa = sample_episode(data)
    labels = torch.tensor([0]*len(qn) + [1]*len(qa), device=DEVICE)

    xs = torch.stack(load_images(support, transform_train)).to(DEVICE)
    xq = torch.stack(load_images(qn + qa, transform_train if train else transform_eval)).to(DEVICE)

    inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)

    with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):

        for _ in range(INNER_STEPS):
            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            inner_loss = -log_student_t(z_sup, mu, Sigma, dof).mean()
            diffopt.step(inner_loss)

        zq = fmodel(xq)
        mu, Sigma, dof = niw_posterior(z_sup)

        logp_n = log_student_t(zq, mu, Sigma, dof)
        logp_a = log_student_t(
            zq,
            torch.zeros(zq.shape[1], device=DEVICE),
            100.0 * torch.eye(zq.shape[1], device=DEVICE),
            torch.tensor(2.0, device=DEVICE),
        )

        bayes_loss = -((1-labels)*logp_n + labels*logp_a).mean()
        supcon_loss = supervised_contrastive_loss(zq, labels)

        loss = bayes_loss + LAMBDA_CON * supcon_loss

    return loss

# ---------------- MAIN ---------------- #

def main():
    print("Loading datasets...")
    datasets = {}
    for c in CLIENTS:
        d = load_client_dataset_leave1out(Path(BASE_ROOT)/c, seed=SEED)
        if d is not None:
            datasets[c] = d
        else:
            print(f"Skipping {c}")

    print(f"Training on {len(datasets)} clients")

    model = EmbeddingNet().to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=META_LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_losses = []

        for _ in range(EPISODES_PER_EPOCH):
            client = random.choice(list(datasets.keys()))
            loss = run_episode(model, datasets[client], train=True)
            meta_opt.zero_grad()
            loss.backward()
            meta_opt.step()
            epoch_losses.append(loss.item())

        train_loss = np.mean(epoch_losses)
        train_losses.append(train_loss)

        # -------- Validation --------
        model.eval()
        val_ep_losses = []
        for _ in range(VAL_EPISODES):
            client = random.choice(list(datasets.keys()))
            val_loss = run_episode(model, datasets[client], train=False)
            val_ep_losses.append(val_loss.item())

        val_loss = np.mean(val_ep_losses)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f}"
        )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                f"{CKPT_DIR}/model_epoch_{epoch}.pth"
            )

    plot_loss_curve(
        train_losses,
        val_losses,
        f"{PLOT_DIR}/loss_curve.png"
    )

    torch.save(model.state_dict(), f"{CKPT_DIR}/final_model.pth")
    print("Training finished.")

if __name__ == "__main__":
    main()
