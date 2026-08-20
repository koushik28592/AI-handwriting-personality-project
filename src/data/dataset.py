import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from config import CLASS_NAMES, IMG_EXTENSIONS, SEED, TRAIN_RATIO, VAL_RATIO


def inspect_dataset(data_dir):
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}. Extract Dataset/ beside src/ first."
        )
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise ValueError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            row = {"path": str(path.resolve()), "class_name": class_name, "valid": False}
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    row.update(width=image.width, height=image.height, mode=image.mode)
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                row["valid"] = True
            except (OSError, UnidentifiedImageError):
                row["error"] = "unreadable image"
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No supported images found in {root}")
    return frame


def make_splits(frame):
    valid = frame[frame["valid"]].copy()
    valid["duplicate_group"] = valid["sha256"]
    valid = valid.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    if valid["class_name"].value_counts().min() < 3:
        raise ValueError("Each class needs at least three unique valid images for splitting.")
    train, remainder = train_test_split(
        valid, test_size=1 - TRAIN_RATIO, stratify=valid["class_name"], random_state=SEED
    )
    val_fraction = VAL_RATIO / (1 - TRAIN_RATIO)
    val, test = train_test_split(
        remainder, test_size=1 - val_fraction, stratify=remainder["class_name"], random_state=SEED
    )
    train = train.assign(split="train")
    val = val.assign(split="validation")
    test = test.assign(split="test")
    return pd.concat([train, val, test], ignore_index=True)