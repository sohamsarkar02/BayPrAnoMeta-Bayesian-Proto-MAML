#!/usr/bin/env python3
"""
Classical MAML for Anomaly Detection
CPU-friendly
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import higher

from utils.utils_data_fedcontrast_leave1out import (
    load_client_dataset_leave1out,
    load_images,
    transform_train,
    transform_eval,
)
from utils.utils_plotting import plot_loss_curve

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

OUT_DIR = "runs/maml_leave1out"
CHECKPOINT_DIR = f"{OUT_DIR}/checkpoints"
PLOT_DIR = f"{OUT_DIR}/plots"

DEVICE = torch.device("cpu")

EPOCHS = 50
EPISODES_PER_EPOCH = 50
VAL_EPISODES = 20

INNER_STEPS = 1
INNER_LR = 5e-4
META_LR = 1e-4

K_SHOT = 5
Q_N = 12
Q_A = 4

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- MODEL ---------------- #

class MAMLNet(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, 1)

    def forward(self, x):
        z = self.features(x).view(x.size(0), -1)
        return self.classifier(z).squeeze(-1)

# ---------------- EPISODE SAMPLING ---------------- #

def sample_episode(data, train=True):
    split = "train" if train else "test"
    support = random.sample(data[split]["normal"], K_SHOT)
    qn = random.sample(data[split]["normal"], Q_N)
    qa = random.sample(data[split]["anomaly"], Q_A)
    return support, qn, qa

# ---------------- RUN ONE EPISODE ---------------- #

def run_episode(model, data, train=True):

    support, qn, qa = sample_episode(data, train)
    labels = torch.tensor(
        [0]*len(qn) + [1]*len(qa),
        dtype=torch.float32,
        device=DEVICE
    )

    transform = transform_train if train else transform_eval

    xs = torch.stack(load_images(support, transform)).to(DEVICE)
    xq = torch.stack(load_images(qn + qa, transform)).to(DEVICE)

    inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)
    bce = nn.BCEWithLogitsLoss()

    with higher.innerloop_ctx(
        model,
        inner_opt,
        copy_initial_weights=True,
        track_higher_grads=True
    ) as (fmodel, diffopt):

        # ---- INNER LOOP (adapt to normal class) ----
        for _ in range(INNER_STEPS):
            logits_sup = fmodel(xs)
            loss_inner = bce(logits_sup, torch.zeros_like(logits_sup))
            diffopt.step(loss_inner)

        # ---- OUTER LOOP ----
        logits_q = fmodel(xq)
        loss_outer = bce(logits_q, labels)

    return loss_outer

# ---------------- TRAINING ---------------- #

def main():

    print("Device:", DEVICE)

    client_data = {}
    for c in CLIENTS:
        data = load_client_dataset_leave1out(
            os.path.join(BASE_ROOT, c), seed=42
        )
        if data is None:
            print(f"Skipping {c}")
            continue
        client_data[c] = data

    print(f"Training on {len(client_data)} clients")

    model = MAMLNet().to(DEVICE)
    meta_opt = torch.optim.Adam(model.parameters(), lr=META_LR)

    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):

        model.train()
        epoch_train = []

        for _ in range(EPISODES_PER_EPOCH):
            c = random.choice(list(client_data.keys()))
            loss = run_episode(model, client_data[c], train=True)

            meta_opt.zero_grad()
            loss.backward()
            meta_opt.step()

            epoch_train.append(loss.item())

        train_losses.append(np.mean(epoch_train))

        # ---- VALIDATION ----
        model.eval()
        epoch_val = []

        for _ in range(VAL_EPISODES):
            c = random.choice(list(client_data.keys()))
            loss = run_episode(model, client_data[c], train=False)
            epoch_val.append(loss.item())

        val_losses.append(np.mean(epoch_val))

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_losses[-1]:.4f} | "
            f"Val Loss {val_losses[-1]:.4f}"
        )

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                f"{CHECKPOINT_DIR}/maml_epoch_{epoch}.pth"
            )

    plot_loss_curve(
        train_losses,
        val_losses,
        f"{PLOT_DIR}/loss_curve.png"
    )

    torch.save(
        model.state_dict(),
        f"{CHECKPOINT_DIR}/final_model.pth"
    )

    print("Training finished.")

if __name__ == "__main__":
    main()
