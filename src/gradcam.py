import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

try:
    from .config import IMAGE_SIZE
except ImportError:
    from config import IMAGE_SIZE

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, tf.keras.Model):
            nested = find_last_conv_layer(layer)
            if nested:
                return nested
            return layer.name
    return None

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    original = np.array(img)
    resized = img.resize(IMAGE_SIZE)
    arr = np.array(resized, dtype=np.float32)
    return original, np.expand_dims(arr, axis=0)

def make_gradcam(model, image_tensor, class_index=None):
    target_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            target_layer = layer
            break
        if isinstance(layer, tf.keras.Model):
            target_layer = layer
            break
    if target_layer is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    with tf.GradientTape() as tape:
        activations = image_tensor
        feature_maps = None
        for layer in model.layers[1:]:
            activations = layer(activations, training=False)
            if layer is target_layer:
                feature_maps = activations
        predictions = activations
        if feature_maps is None:
            raise ValueError("Could not capture the Grad-CAM activation.")

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        score = predictions[:, class_index]

    gradients = tape.gradient(score, feature_maps)

    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(
        feature_maps * tf.expand_dims(tf.expand_dims(weights, 1), 1),
        axis=-1,
    )

    cam = tf.maximum(cam, 0)
    cam = cam[0].numpy()

    if cam.max() > 0:
        cam /= cam.max()

    return cam, int(class_index.numpy()), predictions[0].numpy()

def overlay_heatmap(original, cam):
    cam = cv2.resize(cam, (original.shape[1], original.shape[0]))
    heat = np.uint8(255 * cam)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 0.55, heat, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
