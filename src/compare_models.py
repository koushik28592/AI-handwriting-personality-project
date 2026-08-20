import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Combine generated neural and classical metric reports.")
    parser.parse_args()
    rows = []
    for path in Path("reports").glob("metrics_*.csv"):
        row = pd.read_csv(path).iloc[0].to_dict()
        row["model"] = path.stem.replace("metrics_", "")
        rows.append(row)
    classical = Path("reports/classical_model_comparison.csv")
    if classical.exists():
        rows.extend(pd.read_csv(classical).to_dict("records"))
    if not rows:
        raise FileNotFoundError("No generated metrics found. Train and evaluate models first.")
    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    result.to_csv("reports/model_comparison.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()