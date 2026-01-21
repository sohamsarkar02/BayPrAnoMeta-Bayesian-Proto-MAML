import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def compute_episode_metrics(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return 0.5, 0.5, 0.0

    auc = roc_auc_score(labels, scores)
    pr  = average_precision_score(labels, scores)

    thresholds = np.linspace(scores.min(), scores.max(), 200)
    f1s = [f1_score(labels, scores >= t, zero_division=0) for t in thresholds]

    return auc, pr, max(f1s)

def aggregate_with_se(values):
    values = np.asarray(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    return mean, se
