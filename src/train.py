import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    IMAGE_SIZE, BATCH_SIZE, SEED, CLASS_NAMES,
    MODEL_DIR, MODEL_PATH, CLASS_NAMES_PATH, REPORT_DIR, SPLIT_PATH
)
from data.dataset import inspect_dataset, make_splits
from models.architectures import build_model

def make_datasets(split):
    def load(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
        return tf.cast(image, tf.float32), label
    datasets = {}
    for name in ("train", "validation", "test"):
        part = split[split.split == name]
        paths = part.path.to_numpy()
        labels = part.class_name.map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(load, num_parallel_calls=tf.data.AUTOTUNE)
        datasets[name] = ds.shuffle(len(part), seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) if name == "train" else ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return datasets["train"], datasets["validation"], datasets["test"]


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import f1_score
        y_true = np.concatenate([y.numpy() for _, y in self.validation_data])
        y_pred = np.argmax(self.model.predict(self.validation_data, verbose=0), axis=1)
        score = f1_score(y_true, y_pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = float(score)
        print(f" - val_macro_f1: {score:.4f}")

def plot_history(history1, history2=None):
    history = {}
    for h in [history1, history2]:
        if h is None:
            continue
        for k, v in h.history.items():
            history.setdefault(k, []).extend(v)

    Path("reports").mkdir(exist_ok=True)

    plt.figure()
    plt.plot(history["accuracy"], label="train_accuracy")
    plt.plot(history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/accuracy.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/loss.png", dpi=160)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="Dataset")
    parser.add_argument("--model", choices=["cnn", "resnet50", "efficientnetb0"], default="resnet50")
    parser.add_argument("--epochs_head", type=int, default=8)
    parser.add_argument("--epochs_fine", type=int, default=8)
    parser.add_argument("--fine_tune_layers", type=int, default=30)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    inventory = inspect_dataset(args.data_dir)
    split = make_splits(inventory)
    REPORT_DIR.mkdir(exist_ok=True)
    split.to_csv(SPLIT_PATH, index=False)
    train_ds, val_ds, _ = make_datasets(split)

    model, base = build_model(args.model, len(CLASS_NAMES))
    train_labels = split.loc[split.split == "train", "class_name"].map({c: i for i, c in enumerate(CLASS_NAMES)}).to_numpy()
    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=train_labels)
    class_weight = dict(enumerate(weights))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        MacroF1Callback(val_ds),
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / f"handwriting_personality_{args.model}.keras",
            monitor="val_macro_f1",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            patience=4,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
        ),
    ]

    print("\nStage 1: training classification head...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks, class_weight=class_weight,
    )

    # Fine tune only the last few ResNet layers.
    if base is not None:
        base.trainable = True
        for layer in base.layers[:-args.fine_tune_layers]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nStage 2: fine-tuning ResNet...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_fine,
        callbacks=callbacks, class_weight=class_weight,
    )

    model_path = MODEL_DIR / f"handwriting_personality_{args.model}.keras"
    model.save(model_path)

    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    plot_history(history1, history2)

    print(f"\nSaved model: {model_path}")
    print("Training graphs saved in reports/")

if __name__ == "__main__":
    main()
