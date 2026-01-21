import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ------------------------------
# Transforms (CPU-safe)
# ------------------------------

transform_train = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.1, 0.1, 0.1, 0.1),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_eval = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Utilities (CPU-only)
# ------------------------------

def list_images(path):
    path = Path(path)
    return sorted([str(p) for p in path.glob("*.png")])

def load_images(paths, transform):
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))
        except Exception:
            continue
    return imgs

# ------------------------------
# Leave-One-Anomaly-Subtype-Out
# ------------------------------

def load_client_dataset_leave1out(client_root, seed=42):
    """
    Leave-one-anomaly-subtype-out protocol.

    Training:
      - Normal  : train/good + test/good
      - Anomaly : all anomaly subtypes EXCEPT one

    Testing:
      - Normal  : test/good
      - Anomaly : ONLY the held-out subtype

    Clients with < 2 anomaly subtypes (e.g., toothbrush)
    are skipped by returning None.
    """

    rng = random.Random(seed)
    root = Path(client_root)

    # -------- Normal images --------
    train_good = list_images(root / "train" / "good")
    test_good = list_images(root / "test" / "good")

    if len(train_good) == 0 or len(test_good) == 0:
        return None

    # -------- Anomaly subtypes --------
    anom_dirs = [
        d for d in (root / "test").iterdir()
        if d.is_dir() and d.name != "good"
    ]

    # Skip clients with only one anomaly subtype
    if len(anom_dirs) < 2:
        return None

    rng.shuffle(anom_dirs)

    held_out_dir = anom_dirs[0]
    seen_dirs = anom_dirs[1:]

    # -------- Build anomaly pools --------
    anom_train = []
    for d in seen_dirs:
        anom_train.extend(list_images(d))

    anom_test = list_images(held_out_dir)

    if len(anom_train) == 0 or len(anom_test) == 0:
        return None

    return {
        "train": {
            "normal": train_good + test_good,
            "anomaly": anom_train
        },
        "test": {
            "normal": test_good,
            "anomaly": anom_test
        },
        "held_out_subtype": held_out_dir.name,
        "seen_subtypes": [d.name for d in seen_dirs]
    }
