import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

from config import CLASS_NAMES, REPORT_DIR, SPLIT_PATH
from train import make_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--model", default="resnet50", choices=["cnn", "resnet50", "efficientnetb0"])
    args = parser.parse_args()

    model_path = Path("models") / f"handwriting_personality_{args.model}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py --model {args.model} first."
        )
    if not SPLIT_PATH.exists():
        raise FileNotFoundError("Run src/analyze_dataset.py or src/train.py first to create the split.")
    split = pd.read_csv(SPLIT_PATH)
    _, _, ds = make_datasets(split)
    model = tf.keras.models.load_model(model_path)

    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    print("\nTest metrics")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nClassification Report\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred)

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(REPORT_DIR / f"metrics_{args.model}.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / f"confusion_matrix_{args.model}.png", dpi=180)
    plt.close()

    print(f"Saved reports/metrics_{args.model}.csv and confusion matrix")

if __name__ == "__main__":
    main()
