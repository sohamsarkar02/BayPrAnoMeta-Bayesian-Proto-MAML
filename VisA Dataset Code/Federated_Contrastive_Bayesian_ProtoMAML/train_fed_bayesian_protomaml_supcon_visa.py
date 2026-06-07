#!/usr/bin/env python3
"""
Federated Contrastive BayPrAnoMeta (ViSA):
Federated Contrastive Bayesian ProtoMAML
CPU friendly
"""

import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import higher

from utils.utils_niw import niw_posterior, log_student_t
from utils.utils_plotting import plot_loss_curve
from utils.utils_data_visa_unseen_anomaly import (   
    load_client_dataset_visa,                        
    load_images,
    transform_train,
    transform_eval,
)

# ---------------- CONFIG ---------------- #

BASE_ROOT = "/Users/sohamsarkar/Desktop/ViSA Dataset"   

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

DEVICE = torch.device("cpu")

ROUNDS = 50
EPISODES_PER_CLIENT = 10

INNER_STEPS = 1
INNER_LR = 5e-4
SERVER_LR = 1e-4

K_SHOT = 5
Q_N = 12
Q_A = 4

# Supervised contrastive hyperparameters
LAMBDA_CONTRAST = 0.1
TEMPERATURE = 0.07

OUT_DIR = "runs/fed_supcon_visa"   # CHANGED
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

# ---------------- UTILS ---------------- #

def make_episode(good, anom):
    support = random.sample(good, K_SHOT)
    qn = random.sample(good, Q_N)
    qa = random.sample(anom, Q_A)
    return support, qn, qa

# ---------------- SUPERVISED CONTRASTIVE LOSS ---------------- #

def supervised_contrastive_loss(z, labels, temperature):
    """
    z: (N, d) embeddings
    labels: (N,) {0,1}
    """
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature

    labels = labels.unsqueeze(1)
    mask = torch.eq(labels, labels.T).float()

    logits_mask = 1 - torch.eye(z.size(0), device=z.device)
    mask = mask * logits_mask

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    loss = -mean_log_prob_pos.mean()

    return loss

# ---------------- CLIENT GRADIENT ---------------- #

def client_compute_gradient(global_model, data):
    grads_acc = None

    for _ in range(EPISODES_PER_CLIENT):

        support, qn, qa = make_episode(
            data["train"]["normal"], data["train"]["anomaly"]   
        )

        xs = torch.stack(load_images(support, transform_train)).to(DEVICE)  
        xq = torch.stack(load_images(qn + qa, transform_eval)).to(DEVICE)    
        labels = torch.tensor([0]*len(qn) + [1]*len(qa), device=DEVICE)

        local_model = deepcopy(global_model)
        inner_opt = torch.optim.SGD(local_model.parameters(), lr=INNER_LR)

        with higher.innerloop_ctx(local_model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):

            # ----- Bayesian inner loop -----
            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            inner_loss = -log_student_t(z_sup, mu, Sigma, dof).mean()
            diffopt.step(inner_loss)

            # ----- Outer losses -----
            zq = fmodel(xq)

            logp_n = log_student_t(zq, mu, Sigma, dof)
            logp_a = log_student_t(
                zq,
                torch.zeros(zq.shape[1]),
                100.0 * torch.eye(zq.shape[1]),
                torch.tensor(2.0)
            )

            bayes_loss = -((1-labels)*logp_n + labels*logp_a).mean()
            con_loss = supervised_contrastive_loss(zq, labels, TEMPERATURE)

            total_loss = bayes_loss + LAMBDA_CONTRAST * con_loss

            grads = torch.autograd.grad(total_loss, fmodel.parameters())

        if grads_acc is None:
            grads_acc = [g.detach() for g in grads]
        else:
            for i in range(len(grads_acc)):
                grads_acc[i] += grads[i].detach()

    for i in range(len(grads_acc)):
        grads_acc[i] /= EPISODES_PER_CLIENT

    return grads_acc

# ---------------- VALIDATION ---------------- #

def run_validation(model, client_data):
    losses = []

    for client, data in client_data.items():

        support, qn, qa = make_episode(
            data["test"]["normal"], data["test"]["anomaly"]  
        )

        xs = torch.stack(load_images(support, transform_train)).to(DEVICE)   
        xq = torch.stack(load_images(qn + qa, transform_eval)).to(DEVICE)    
        labels = torch.tensor([0]*len(qn) + [1]*len(qa), device=DEVICE)

        inner_opt = torch.optim.SGD(model.parameters(), lr=INNER_LR)

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=True) as (fmodel, diffopt):

            z_sup = fmodel(xs)
            mu, Sigma, dof = niw_posterior(z_sup)
            inner_loss = -log_student_t(z_sup, mu, Sigma, dof).mean()
            diffopt.step(inner_loss)

            with torch.no_grad():
                zq = fmodel(xq)
                logp_n = log_student_t(zq, mu, Sigma, dof)
                logp_a = log_student_t(
                    zq,
                    torch.zeros(zq.shape[1]),
                    100.0 * torch.eye(zq.shape[1]),
                    torch.tensor(2.0)
                )
                loss = -((1-labels)*logp_n + labels*logp_a).mean()
                losses.append(loss.item())

    return float(np.mean(losses))

# ---------------- MAIN ---------------- #

def main():

    print("Training on CPU")

    model = EmbeddingNet().to(DEVICE)
    server_opt = torch.optim.Adam(model.parameters(), lr=SERVER_LR)

    client_data = {}

    for c in CLIENTS:
        data = load_client_dataset_visa(os.path.join(BASE_ROOT, c), seed=42)   
        if data is None:
            print(f"Skipping {c} (insufficient data)")                          
            continue
        client_data[c] = data

    print(f"Training on {len(client_data)} clients")

    if len(client_data) == 0:                                                   
        raise ValueError("No valid VisA clients found.")

    train_losses, val_losses = [], []

    for r in range(1, ROUNDS+1):
        grads_all = []

        for c in client_data:
            grads = client_compute_gradient(model, client_data[c])
            grads_all.append(grads)

        # ----- FedAvg -----
        for p_idx, p in enumerate(model.parameters()):
            p.grad = torch.stack([g[p_idx] for g in grads_all]).mean(0)

        server_opt.step()
        server_opt.zero_grad()

        train_loss = torch.stack([g[0].norm() for g in grads_all]).mean().item()
        val_loss = run_validation(model, client_data)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Round {r:03d} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}")

    torch.save(model.state_dict(), f"{OUT_DIR}/final_model.pth")
    plot_loss_curve(train_losses, val_losses, f"{OUT_DIR}/loss_curve.png")

    print("Training finished.")

if __name__ == "__main__":
    main()