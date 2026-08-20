import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from config import IMAGE_SIZE


def _head(features, num_classes, name):
    x = layers.GlobalAveragePooling2D(name=f"{name}_pool")(features)
    x = layers.BatchNormalization(name=f"{name}_batch_norm")(x)
    x = layers.Dense(256, activation="relu", name=f"{name}_dense")(x)
    x = layers.Dropout(0.45, name=f"{name}_dropout")(x)
    return layers.Dense(num_classes, activation="softmax", name="personality")(x)


def build_model(model_name, num_classes):
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="handwriting_image")
    augmented = tf.keras.Sequential([
        layers.RandomRotation(0.035), layers.RandomTranslation(0.04, 0.04),
        layers.RandomZoom((-0.05, 0.08)), layers.RandomContrast(0.12),
    ], name="handwriting_augmentation")(inputs)
    if model_name == "cnn":
        x = layers.Rescaling(1.0 / 255)(augmented)
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D()(x)
        outputs = _head(x, num_classes, "cnn")
        return tf.keras.Model(inputs, outputs, name="HandwritingCNN"), None
    if model_name == "efficientnetb0":
        base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
        x = efficientnet_preprocess(augmented)
    elif model_name == "resnet50":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
        x = resnet_preprocess(augmented)
    else:
        raise ValueError("model_name must be cnn, resnet50, or efficientnetb0")
    base.trainable = False
    outputs = _head(base(x, training=False), num_classes, model_name)
    return tf.keras.Model(inputs, outputs, name=f"Handwriting{model_name.title()}"), base


def build_fusion_model(num_features, num_classes):
    """Build an image plus OpenCV-feature classifier for optional experiments."""
    image_input = layers.Input(shape=(*IMAGE_SIZE, 3), name="handwriting_image")
    feature_input = layers.Input(shape=(num_features,), name="opencv_features")
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    image_embedding = layers.GlobalAveragePooling2D()(base(resnet_preprocess(image_input), training=False))
    feature_embedding = layers.BatchNormalization()(feature_input)
    feature_embedding = layers.Dense(64, activation="relu")(feature_embedding)
    merged = layers.Concatenate()([image_embedding, feature_embedding])
    merged = layers.Dropout(0.45)(merged)
    outputs = layers.Dense(num_classes, activation="softmax", name="personality")(merged)
    return tf.keras.Model([image_input, feature_input], outputs, name="HandwritingResNetOpenCVFusion"), base