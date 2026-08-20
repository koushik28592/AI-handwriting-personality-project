import argparse
from pathlib import Path

from data.dataset import inspect_dataset, make_splits


def main():
    parser = argparse.ArgumentParser(description="Inspect and split handwriting data.")
    parser.add_argument("--data_dir", default="Dataset")
    args = parser.parse_args()
    frame = inspect_dataset(args.data_dir)
    split = make_splits(frame)
    Path("reports").mkdir(exist_ok=True)
    frame.to_csv("reports/dataset_inventory.csv", index=False)
    split.to_csv("reports/dataset_split.csv", index=False)
    print("Image counts including invalid files:")
    print(frame.groupby(["class_name", "valid"]).size().to_string())
    print(f"\nExact duplicate files: {len(frame[frame.valid]) - frame[frame.valid].sha256.nunique()}")
    print("\nLeakage-safe split counts:")
    print(split.groupby(["split", "class_name"]).size().unstack(fill_value=0).to_string())
    print("\nSaved reports/dataset_inventory.csv and reports/dataset_split.csv")


if __name__ == "__main__":
    main()