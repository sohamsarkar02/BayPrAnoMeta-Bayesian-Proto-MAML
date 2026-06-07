# Meta-Learning for Anomaly Detection

This repository contains implementations of various meta-learning approaches for few-shot anomaly detection on the **MVTec Anomaly Detection Dataset** and further we also report training and evaluation scripts for **VisA Dataset**.

## Results Summary

Average performance across 15 object categories (Leave-One-Anomaly-Subtype-Out protocol) of MVTec AD Dataset:

| Method | AUC-ROC | AUPRC | F1 Score |
|--------|---------|-------|----------|
| **Bayesian ProtoMAML** | **0.8408** | **0.7391** | **0.7755** |
| Classical ProtoMAML | 0.8020 | 0.7051 | 0.7429 |
| Federated Bayesian ProtoMAML | 0.6357 | 0.5256 | 0.6051 |
| FedCon Bayesian ProtoMAML | 0.6363 | 0.5278 | 0.6085 |
| Contrastive Bayesian ProtoMAML | 0.6186 | 0.5177 | 0.5995 |
| Classical MAML | 0.5786 | 0.4948 | 0.5697 |
| PatchCore (10% coreset) | 0.5530 | 0.5269 | 0.6054 |

>

---

## Project Structure

```
ML Codes/
├── Bayesian_ProtoMAML/           #  Best performing method
├── CLassical_ProtoMAML/          #  Second best
├── Classical MAML/               # Baseline MAML
├── Contrastive_Bayesian_ProtoMAML/  # +SupCon loss
├── Federated Bayesian_ProtoMAML/    # Federated training
├── FedCon_Bayesian_ProtoMAML/       # Federated + Contrastive
├── PatchCore/                       # Memory bank baseline
└── README.md
```

---

## Methods Overview (For convenience we report the summary for MVTec AD Dataset only, here in this file)

### 1. Bayesian ProtoMAML (`Bayesian_ProtoMAML/`)

**Best performing method** combining Prototypical Networks with MAML and Bayesian inference.

| File | Description |
|------|-------------|
| `train_bayesian_protomaml_leave1out.py` | Training script with NIW posterior |
| `evaluate_bayesian_protomaml_leave1out.py` | Evaluation with metrics & visualizations |

**Key Features:**

- Normal-Inverse-Wishart (NIW) posterior for robust prototype estimation
- Student-t likelihood for uncertainty-aware scoring
- Inner loop: fit NIW posterior to support set
- Outer loop: score queries against learned prototype vs anomaly prior

---

### 2. Classical ProtoMAML (`CLassical_ProtoMAML/`)

Standard ProtoMAML using mean embeddings and Euclidean distance.

| File | Description |
|------|-------------|
| `train_protomaml_leave1out.py` | Training with BCE loss on distance-based logits |
| `evaluate_protomaml_leave1out.py` | Evaluation script |

**Key Features:**

- Prototype = mean of support embeddings
- Inner loop: minimize variance of support around prototype
- Outer loop: BCE loss on squared distance logits

---

### 3. Classical MAML (`Classical MAML/`)

Direct MAML adaptation for binary anomaly classification.

| File | Description |
|------|-------------|
| `train_maml_anomaly_leave1out.py` | MAML training with direct classification |
| `evaluate_maml_anomaly_leave1out.py` | Evaluation script |

**Key Features:**

- Binary classifier head (512 → 1)
- Inner loop: adapt to classify support as normal
- Outer loop: BCE loss on query predictions

---

### 4. Contrastive Bayesian ProtoMAML (`Contrastive_Bayesian_ProtoMAML/`)

Extends Bayesian ProtoMAML with Supervised Contrastive Learning.

| File | Description |
|------|-------------|
| `train_bayesian_protomaml_supcon_leave1out.py` | Training with SupCon loss |
| `evaluate_bayesian_protomaml_supcon_leave1out.py` | Evaluation script |

**Key Features:**

- Combines Bayesian loss + λ × SupCon loss (λ=0.1)
- Temperature-scaled contrastive learning
- Pulls same-class embeddings together, pushes different classes apart

---

### 5. Federated Bayesian ProtoMAML (`Federated Bayesian_ProtoMAML/`)

Federated Learning version for multi-client training.

| File | Description |
|------|-------------|
| `train_fed_bayesian_protomaml_leave1out.py` | Federated training with FedAvg |
| `evaluate_fed_bayesian_protomaml_leave1out.py` | Evaluation script |

**Key Features:**

- Each client computes gradients locally
- Server aggregates gradients via FedAvg
- Simulates privacy-preserving federated learning

---

### 6. FedCon Bayesian ProtoMAML (`FedCon_Bayesian_ProtoMAML/`)

Combines Federated Learning with Contrastive Learning.

| File | Description |
|------|-------------|
| `train_fed_bayesian_protomaml_supcon_leave1out.py` | Federated + SupCon training |
| `evaluate_fed_bayesian_protomaml_supcon_leave1out.py` | Evaluation script |

**Key Features:**

- FedAvg aggregation + Supervised Contrastive loss
- Local Bayesian + Contrastive optimization per client
- Global model synchronization each round

---

### 7. PatchCore (`PatchCore/`)

Memory bank-based baseline using patch-level features.

| File | Description |
|------|-------------|
| `train_patchcore_leave1out.py` | Extract patches, build coreset (10%) |
| `train_patchcore_leave1out_1pct.py` | 1% coreset variant |
| `train_patchcore_leave1out_25pct.py` | 25% coreset variant |
| `train_patchcore_leave1out_50pct.py` | 50% coreset variant |
| `train_patchcore_leave1out_100pct.py` | 100% (no coreset) variant |
| `evaluate_patchcore_leave1out*.py` | Corresponding evaluation scripts |

**Key Features:**

- ResNet-18 backbone extracts 7×7 patch embeddings
- Greedy k-center coreset selection for memory efficiency
- Anomaly score = distance to nearest normal patch in memory bank

---

## Shared Architecture

All methods use the same encoder for fair comparison:

```python
class EmbeddingNet(nn.Module):
    # ResNet-18 (pretrained) → 512-dim
    # Linear(512, 128) → ReLU → LayerNorm → 128-dim
```

---

## Common Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | ResNet-18 (ImageNet pretrained) |
| Embed Dim | 128 |
| K-Shot | 5 |
| Query Normal | 12 |
| Query Anomaly | 4 |
| Inner LR | 5e-4 |
| Meta/Server LR | 1e-4 |
| Epochs/Rounds | 50 |

---

## Requirements

```
torch
torchvision
higher          # Meta-learning library
numpy
matplotlib
seaborn
scikit-learn
pillow
```

---

## Dataset

**MVTec Anomaly Detection Dataset** with 15 object categories:

- bottle, cable, capsule, carpet, grid
- hazelnut, leather, metal_nut, pill, screw
- tile, toothbrush, transistor, wood, zipper

**Evaluation Protocol:** Leave-one-anomaly-subtype-out

- Train on N-1 anomaly subtypes
- Test on held-out subtype

---

## Output Structure

Each method generates:

```
runs/{method_name}/
├── checkpoints/    # Model weights
├── plots/          # Loss curves
└── eval/           # Evaluation results
    ├── {category}/
    │   ├── roc.png
    │   ├── pr.png
    │   ├── confusion_matrix.png
    │   ├── score_hist.png
    │   └── tsne.png
    └── results.npy
```

---
