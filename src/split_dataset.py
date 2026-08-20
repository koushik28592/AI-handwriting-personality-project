"""
Optional utility.

The main train.py uses TensorFlow's deterministic 80/20 validation split.
This script is provided if you want physically separated train/val/test folders.

It creates:
split_dataset/
    train/
    val/
    test/

Run:
python src/split_dataset.py --data_dir Dataset
"""

import argparse
import random
import shutil
from pathlib import Path

from config import CLASS_NAMES, IMG_EXTENSIONS, SEED

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--output_dir", default="split_dataset")
    args = parser.parse_args()

    random.seed(SEED)

    source = Path(args.data_dir)
    output = Path(args.output_dir)

    if output.exists():
        shutil.rmtree(output)

    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            (output / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASS_NAMES:
        files = [
            p for p in (source / cls).rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)

        groups = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, paths in groups.items():
            for idx, path in enumerate(paths):
                destination = output / split / cls / f"{idx}_{path.name}"
                shutil.copy2(path, destination)

        print(
            f"{cls}: train={len(groups['train'])}, "
            f"val={len(groups['val'])}, test={len(groups['test'])}"
        )

    print(f"\nCreated: {output}")

if __name__ == "__main__":
    main()
