from pathlib import Path
from collections import Counter
from config import IMG_EXTENSIONS

def inspect_dataset(data_dir):
    data_dir = Path(data_dir)
    counts = Counter()

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        count = sum(
            1 for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        counts[class_dir.name] = count

    total = sum(counts.values())
    print("\nDataset summary")
    print("-" * 40)
    for name, count in counts.items():
        print(f"{name:20s} {count}")
    print("-" * 40)
    print(f"{'Total':20s} {total}\n")

    return counts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    args = parser.parse_args()
    inspect_dataset(args.data_dir)
