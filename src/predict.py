import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import IMAGE_SIZE, MODEL_PATH, CLASS_NAMES_PATH

def load_and_prepare(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="resnet50", choices=["cnn", "resnet50", "efficientnetb0"])
    args = parser.parse_args()

    model_path = Path("models") / f"handwriting_personality_{args.model}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Train the model first using: python src/train.py --model {args.model}")

    model = tf.keras.models.load_model(model_path)

    if Path(CLASS_NAMES_PATH).exists():
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            classes = json.load(f)
    else:
        classes = ["Extrovert", "Introvert", "Optimistic", "Pessimistic", "Stable_Mindset"]

    image = load_and_prepare(args.image)
    probabilities = model.predict(image, verbose=0)[0]

    order = np.argsort(probabilities)[::-1]

    print("\nPERSONALITY PREDICTION")
    print("-" * 40)
    for i in order[:3]:
        print(f"{classes[i]:20s} {probabilities[i] * 100:6.2f}%")

    best = int(order[0])
    print("-" * 40)
    print(f"Prediction: {classes[best]}")
    print(f"Confidence: {probabilities[best] * 100:.2f}%")

if __name__ == "__main__":
    main()
