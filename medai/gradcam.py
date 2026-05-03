from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


def find_last_conv_layer_name(model: tf.keras.Model) -> Optional[str]:
    """Best-effort: returns the last Conv-like layer name in a Keras model."""
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    for layer in reversed(model.layers):
        if isinstance(layer, conv_types):
            return layer.name
    return None


def compute_gradcam(
    img_batch: np.ndarray,
    model: tf.keras.Model,
    *,
    pred_index: Optional[int] = None,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """Computes a Grad-CAM heatmap for a single image batch (shape [1,H,W,3]).

    Returns a normalized heatmap array (Hc, Wc) in [0,1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    if last_conv_layer_name is None:
        return np.zeros((7, 7), dtype=np.float32)

    try:
        grad_model = Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            if pred_index is None:
                pred_index = int(tf.argmax(predictions[0]))
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return np.zeros((7, 7), dtype=np.float32)

        output = conv_outputs[0]
        grads = grads[0]

        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = output @ weights[..., tf.newaxis]
        cam = tf.squeeze(cam)

        cam = tf.maximum(cam, 0)
        denom = tf.reduce_max(cam) + 1e-10
        cam = cam / denom
        return cam.numpy().astype(np.float32)
    except Exception:
        return np.zeros((7, 7), dtype=np.float32)
