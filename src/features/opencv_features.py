from pathlib import Path

import cv2
import numpy as np


FEATURE_NAMES = [
    "ink_density", "bbox_width_ratio", "bbox_height_ratio", "bbox_aspect_ratio",
    "contour_count", "mean_contour_area", "estimated_slant", "baseline_variation",
    "horizontal_spacing", "vertical_spacing",
]


def extract_features(image_path):
    image = cv2.imread(str(Path(image_path)), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ink = threshold > 0
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    x, y, width, height = cv2.boundingRect(threshold)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours], dtype=np.float32)
    moments = [cv2.moments(c) for c in contours if cv2.contourArea(c) > 2]
    slopes = []
    for contour in contours:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            angle = ellipse[2]
            slopes.append(abs(float(angle - 90.0)) / 90.0)
    row_counts = np.sum(ink, axis=1)
    occupied_rows = np.flatnonzero(row_counts > 0)
    baseline_variation = float(np.std(occupied_rows)) / 224.0 if len(occupied_rows) else 0.0
    columns = np.flatnonzero(np.sum(ink, axis=0) > 0)
    gaps = np.diff(columns)
    horizontal_spacing = float(np.median(gaps[gaps > 1])) / 224.0 if np.any(gaps > 1) else 0.0
    component_centers = [(m["m10"] / m["m00"], m["m01"] / m["m00"]) for m in moments if m["m00"]]
    vertical_spacing = float(np.std([point[1] for point in component_centers])) / 224.0 if component_centers else 0.0
    values = [
        float(np.mean(ink)), width / 224.0, height / 224.0, width / max(height, 1),
        float(len(contours)), float(np.mean(areas)) / (224.0 * 224.0),
        float(np.mean(slopes)) if slopes else 0.0, baseline_variation,
        horizontal_spacing, vertical_spacing,
    ]
    return np.asarray(values, dtype=np.float32)