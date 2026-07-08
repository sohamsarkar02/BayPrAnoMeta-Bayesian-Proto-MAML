#!/usr/bin/env python3
"""
PatchCore Training Script
Greedy Coreset Selection (50%)
CPU-friendly
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

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

DEVICE = torch.device("cpu")


OUT_DIR = "runs/patchcore_shared_encoder_50pct"
MEMORY_DIR = f"{OUT_DIR}/memory_bank"
os.makedirs(MEMORY_DIR, exist_ok=True)

CORESET_FRACTION = 0.50   

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ---------------- MODEL ---------------- #

class SharedEmbeddingNet(nn.Module):
    """
    Same architecture as Bayesian ProtoMAML encoder:
    ResNet-18 backbone + projection head (512 -> 128)
    """
    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Replace maxpool with avgpool
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
        feats = self.backbone(x)          # [B, 512, H, W]
        B, C, H, W = feats.shape

        patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
        patches = self.projector(patches)

        return patches   # [B*H*W, 128]

# ---------------- CORESET SELECTION ---------------- #

def greedy_coreset(X, fraction, seed=42):
    """
    Greedy k-center coreset selection.
    X: [N, D] numpy array
    """
    np.random.seed(seed)

    N = X.shape[0]
    k = max(1, int(N * fraction))

    idx = np.random.randint(0, N)
    selected = [idx]

    dist = np.linalg.norm(X - X[idx], axis=1)

    for _ in range(1, k):
        idx = np.argmax(dist)
        selected.append(idx)

        new_dist = np.linalg.norm(X - X[idx], axis=1)
        dist = np.minimum(dist, new_dist)

    return np.array(selected)

# ---------------- MAIN TRAINING ---------------- #

def main():

    print("Device:", DEVICE)
    print("Training PatchCore (shared encoder, 50% coreset)...")

    model = SharedEmbeddingNet().to(DEVICE)
    model.eval()

    for client in CLIENTS:

        data = load_client_dataset_leave1out(
            os.path.join(BASE_ROOT, client), seed=42
        )

        if data is None:
            print(f"Skipping {client} (only one anomaly subtype)")
            continue

        print(f"\nProcessing → {client}")

        normal_imgs = data["train"]["normal"]
        all_patches = []

        with torch.no_grad():
            for path in normal_imgs:
                imgs = load_images([path], transform_eval)
                if len(imgs) == 0:
                    continue

                x = imgs[0].unsqueeze(0).to(DEVICE)
                patches = model(x)
                all_patches.append(patches.cpu().numpy())

        if len(all_patches) == 0:
            print(f"No data for {client}, skipping.")
            continue

        all_patches = np.vstack(all_patches)
        print(f"Total patches: {all_patches.shape[0]}")

        # -------- CORESET SUBSAMPLING (50%) -------- #

        idx = greedy_coreset(
            all_patches,
            fraction=CORESET_FRACTION
        )

        memory_bank = all_patches[idx]

        save_path = f"{MEMORY_DIR}/{client}_memory.npy"
        np.save(save_path, memory_bank)

        print(
            f"Saved memory bank → {client} | "
            f"patches={memory_bank.shape[0]}"
        )

    print("\nPatchCore training finished.")

# ---------------- ENTRY ---------------- #

if __name__ == "__main__":
    main()
