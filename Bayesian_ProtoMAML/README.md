# BayPrAnoMeta: Bayesian ProtoMAML for Anomaly Detection

A Bayesian Prototypical MAML (Model-Agnostic Meta-Learning) framework for few-shot anomaly detection on the MVTec Anomaly Detection dataset.

## Overview

This project implements **Bayesian ProtoMAML**, a meta-learning approach that combines:
- **Prototypical Networks**: Learning a metric space for few-shot classification
- **MAML**: Gradient-based meta-learning for fast adaptation
- **Bayesian Inference**: Normal-Inverse-Wishart (NIW) posterior for robust prototype estimation

The framework uses a **leave-one-anomaly-subtype-out** protocol, where one anomaly subtype is held out for testing while the model is trained on all other subtypes.

## Project Structure

```
Bayesian_ProtoMAML/
├── train_bayesian_protomaml_leave1out.py   # Training script
├── evaluate_bayesian_protomaml_leave1out.py # Evaluation script
├── utils/
│   ├── utils_data_fedcontrast_leave1out.py # Data loading & preprocessing
│   ├── utils_niw.py                         # NIW posterior & Student-t likelihood
│   ├── utils_eval_stats.py                  # Evaluation metrics (AUC, PR, F1)
│   └── utils_plotting.py                    # Visualization utilities
└── runs/                                    # Checkpoints & results
    └── bpmaml_leave1out_cpu/
        ├── checkpoints/                     # Model checkpoints
        ├── plots/                           # Training loss curves
        └── eval/                            # Evaluation outputs
```

## Requirements

```python
torch
torchvision
higher                  # Meta-learning library
numpy
matplotlib
seaborn
scikit-learn
pillow
```

## Dataset

This project uses the **MVTec Anomaly Detection Dataset** with 15 object categories:
- bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

**Expected dataset structure:**
```
MVTech AD Dataset/
└── {category}/
    ├── train/good/       # Normal training images
    └── test/
        ├── good/         # Normal test images
        └── {anomaly_type}/ # Various anomaly subtypes
```

> **Note:** Update `BASE_ROOT` in the scripts to point to your dataset location.

## Model Architecture

**EmbeddingNet** - ResNet18 backbone with custom embedding head:
- Pretrained ResNet18 (ImageNet weights)
- Global average pooling → 512-dim features
- Linear(512, 128) → ReLU → LayerNorm → 128-dim embeddings

## Core Algorithm

### Bayesian Inner Loop
1. Compute support embeddings `z_sup`
2. Estimate NIW posterior parameters (`μ`, `Σ`, `dof`)
3. Compute Student-t log-likelihood loss on support set
4. Update model via inner gradient step

### Bayesian Outer Loss
1. Re-compute posterior from adapted support embeddings
2. Score query samples against:
   - **Normal class**: Student-t centered at learned prototype (`μ`, `Σ`, `dof`)
   - **Anomaly class**: Vague Student-t prior (wide covariance, low dof)
3. Compute cross-entropy style loss for anomaly scoring

## Training

```bash
python train_bayesian_protomaml_leave1out.py
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EPOCHS` | 50 | Number of training epochs |
| `EPISODES_PER_EPOCH` | 50 | Episodes per epoch |
| `VAL_EPISODES` | 20 | Validation episodes |
| `INNER_STEPS` | 1 | Inner loop gradient steps |
| `INNER_LR` | 5e-4 | Inner loop learning rate |
| `META_LR` | 1e-4 | Meta (outer) learning rate |
| `K_SHOT` | 5 | Support set size (normal) |
| `Q_N` | 12 | Query normals |
| `Q_A` | 4 | Query anomalies |

### Outputs
- Checkpoints saved every 5 epochs in `runs/bpmaml_leave1out_cpu/checkpoints/`
- Loss curves saved in `runs/bpmaml_leave1out_cpu/plots/`

## Evaluation

```bash
python evaluate_bayesian_protomaml_leave1out.py
```

### Evaluation Protocol
- 300 test episodes per client category
- Metrics computed with standard error estimates
- Anomaly scoring: `log p(x | anomaly) - log p(x | normal)`

### Metrics
- **AUC-ROC**: Area under ROC curve
- **AUPRC**: Area under Precision-Recall curve
- **F1 Score**: Best F1 across thresholds

### Generated Outputs (per category)
- `roc.png` - ROC curve
- `pr.png` - Precision-Recall curve
- `score_hist.png` - Score distribution histogram
- `confusion_matrix.png` - Confusion matrix (95th percentile threshold)
- `tsne.png` - t-SNE embedding visualization

### Global Outputs
- `all_clients_pr_curves.png` - Combined PR curves across categories
- `results.npy` - Serialized metrics dictionary

## Utility Modules

### `utils_niw.py`
- `niw_posterior(z)`: Compute Normal-Inverse-Wishart posterior parameters
- `log_student_t(x, mu, Sigma, dof)`: Multivariate Student-t log-likelihood

**NIW Hyperparameters:**
- `KAPPA0 = 0.01` (prior pseudo-count)
- `NU0_OFFSET = 2` (degrees of freedom offset)
- `LAMBDA0_SCALE = 1.0` (prior scale matrix)

### `utils_data_fedcontrast_leave1out.py`
- `load_client_dataset_leave1out()`: Leave-one-subtype-out data loader
- `transform_train`: Training augmentations (resize, crop, flip, color jitter)
- `transform_eval`: Evaluation transforms (resize, center crop)

### `utils_eval_stats.py`
- `compute_episode_metrics()`: Per-episode AUC, AUPRC, F1
- `aggregate_with_se()`: Mean ± standard error aggregation

### `utils_plotting.py`
- `plot_loss_curve()`: Training/validation loss curves
- `plot_roc_curve()`, `plot_pr_curve()`: ROC and PR visualization
- `plot_score_histogram()`: Normal vs anomaly score distribution

## Key Design Decisions

1. **CPU-friendly**: Designed to run efficiently on CPU (no GPU required)
2. **Higher library**: Uses Facebook's `higher` for differentiable inner loop optimization
3. **Leave-one-out protocol**: Held-out anomaly subtype for unbiased evaluation
4. **Bayesian uncertainty**: NIW posterior provides uncertainty estimates vs. point prototypes
5. **Vague anomaly prior**: Wide Student-t prior for anomalies (no learned anomaly prototype)

## Citation

If you use this code, please cite the relevant works on:
- Model-Agnostic Meta-Learning (MAML)
- Prototypical Networks
- MVTec Anomaly Detection Dataset

## License

[Add your license information here]
