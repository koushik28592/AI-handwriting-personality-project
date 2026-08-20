import numpy as np
from PIL import Image, ImageDraw

from features.opencv_features import FEATURE_NAMES, extract_features


def test_opencv_features_are_real_and_finite(tmp_path):
    path = tmp_path / "sample.png"
    image = Image.new("L", (224, 224), 255)
    ImageDraw.Draw(image).line((20, 100, 200, 80), fill=0, width=3)
    image.save(path)
    values = extract_features(path)
    assert len(values) == len(FEATURE_NAMES)
    assert np.isfinite(values).all()
    assert values[0] > 0