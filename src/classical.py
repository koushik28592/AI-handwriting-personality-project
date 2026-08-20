import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import CLASS_NAMES, REPORT_DIR, SPLIT_PATH
from data.dataset import inspect_dataset, make_splits
from features.opencv_features import extract_features


def main():
    parser = argparse.ArgumentParser(description="Train OpenCV-feature classical baselines.")
    parser.add_argument("--data_dir", default="Dataset")
    args = parser.parse_args()
    split = make_splits(inspect_dataset(args.data_dir))
    Path(REPORT_DIR).mkdir(exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)
    features = {row.path: extract_features(row.path) for row in split.itertuples()}
    labels = {name: i for i, name in enumerate(CLASS_NAMES)}
    train = split[split.split == "train"]
    test = split[split.split == "test"]
    x_train = [features[path] for path in train.path]
    x_test = [features[path] for path in test.path]
    y_train = [labels[name] for name in train.class_name]
    y_test = [labels[name] for name in test.class_name]
    estimators = {
        "svm": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    }
    results = []
    for name, estimator in estimators.items():
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_test)
        results.append({
            "model": name, "accuracy": accuracy_score(y_test, prediction),
            "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
            "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_test, prediction, average="weighted", zero_division=0),
        })
    output = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    output.to_csv(REPORT_DIR / "classical_model_comparison.csv", index=False)
    print(output.to_string(index=False))
    print("Saved reports/classical_model_comparison.csv")


if __name__ == "__main__":
    main()