"""
utils_data_fedcontrast.py
-------------------------
Federated data utilities for supervised contrastive learning
with anomaly subtypes on MVTec AD.
"""

import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

SEED = 42
random.seed(SEED)

# ---------- Transforms ----------
transform_train = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.1, 0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

transform_eval = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ---------- Helpers ----------
def list_images(path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += list(Path(path).glob(e))
    return [str(f) for f in sorted(files)]

# ---------- Client dataset ----------
def load_client_dataset(client_root):
    """
    client_root/
      train/good/
      test/good/
      test/{anomaly_type}/
    """
    client_root = Path(client_root)

    normal = (
        list_images(client_root / "train" / "good")
        + list_images(client_root / "test" / "good")
    )

    anomaly = {}
    for d in (client_root / "test").iterdir():
        if d.is_dir() and d.name != "good":
            anomaly[d.name] = list_images(d)

    if len(normal) == 0:
        raise ValueError(f"No normal images found in {client_root}")

    return normal, anomaly

# ---------- Supervised contrastive batches ----------
def sample_contrastive_batch(normal, anomaly_dict, batch_size=32):
    """
    Labels:
      0            -> normal
      1..K         -> anomaly subtypes
    """
    images, labels = [], []

    n_norm = batch_size // 2
    images += random.sample(normal, min(len(normal), n_norm))
    labels += [0] * len(images)

    anomaly_types = list(anomaly_dict.keys())
    remaining = batch_size - len(images)

    for i in range(remaining):
        a = random.choice(anomaly_types)
        images.append(random.choice(anomaly_dict[a]))
        labels.append(anomaly_types.index(a) + 1)

    return images, labels

def load_images(paths, transform):
    return [transform(Image.open(p).convert("RGB")) for p in paths]
