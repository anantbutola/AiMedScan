from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def load_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_image_as_rgb(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_model_input(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    resized = cv2.resize(rgb, (size, size))
    # Keep in 0..255 float32; trained model should include its own preprocessing layer.
    return np.expand_dims(resized.astype(np.float32), axis=0)


def downscale_for_display(rgb: np.ndarray, max_side: int = 1024) -> np.ndarray:
    h, w = rgb.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return rgb
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def superimpose_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
    heatmap_u8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    jet = cv2.resize(jet, (rgb.shape[1], rgb.shape[0]))
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    overlay = jet_rgb * alpha + rgb * (1 - alpha)
    return overlay.astype("uint8"), jet_rgb.astype("uint8")


def img_to_b64_png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")
