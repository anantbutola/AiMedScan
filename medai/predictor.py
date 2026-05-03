from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from .gradcam import compute_gradcam


@dataclass
class ModelInfo:
    modality: str
    available: bool
    model_type: str
    classes: List[str]
    task: str
    input_size: int


@dataclass
class ModelConfig:
    modality: str
    model_path: str
    classes_path: str
    detected_type: str
    model_tag: str
    task: str
    positive_label: str
    negative_label: str


@dataclass
class ModelBundle:
    config: ModelConfig
    model: Optional[tf.keras.Model]
    classes: List[str]
    task: str
    positive_label: str
    negative_label: str
    input_size: int
    last_conv_layer_name: str


class MultiPredictor:
    def __init__(self, configs: Optional[List[ModelConfig]] = None) -> None:
        if configs is None:
            configs = [
                ModelConfig(
                    modality="chest",
                    model_path=os.path.join("models", "chest_classifier.keras"),
                    classes_path=os.path.join("models", "chest_classes.json"),
                    detected_type="Chest X-Ray",
                    model_tag="CheXpert-5",
                    task="multilabel",
                    positive_label="",
                    negative_label="No Finding",
                ),
                ModelConfig(
                    modality="brain",
                    model_path=os.path.join("models", "brain_classifier.keras"),
                    classes_path=os.path.join("models", "brain_classes.json"),
                    detected_type="Brain MRI",
                    model_tag="BraTS-IXI",
                    task="binary",
                    positive_label="Tumor",
                    negative_label="Normal",
                ),
                ModelConfig(
                    modality="bone",
                    model_path=os.path.join("models", "bone_classifier.keras"),
                    classes_path=os.path.join("models", "bone_classes.json"),
                    detected_type="Bone X-Ray",
                    model_tag="MURA",
                    task="binary",
                    positive_label="Abnormal",
                    negative_label="Normal",
                ),
                ModelConfig(
                    modality="knee",
                    model_path=os.path.join("models", "knee_classifier.keras"),
                    classes_path=os.path.join("models", "knee_classes.json"),
                    detected_type="Knee X-Ray",
                    model_tag="Knee-KL",
                    task="multiclass",
                    positive_label="",
                    negative_label="0Normal",
                ),
            ]

        self.bundles: Dict[str, ModelBundle] = {}
        for cfg in configs:
            self.bundles[cfg.modality] = self._load_bundle(cfg)

        self.legacy_chest_model = os.path.join("models", "medical_classifier.keras")
        self.legacy_chest_classes = os.path.join("models", "classes.json")
        if (
            self.bundles["chest"].model is None
            and os.path.exists(self.legacy_chest_model)
            and os.path.exists(self.legacy_chest_classes)
        ):
            legacy_cfg = ModelConfig(
                modality="chest",
                model_path=self.legacy_chest_model,
                classes_path=self.legacy_chest_classes,
                detected_type="Chest X-Ray",
                model_tag="CheXpert-5",
                task="multilabel",
                positive_label="",
                negative_label="No Finding",
            )
            self.bundles["chest"] = self._load_bundle(legacy_cfg)

    def _infer_input_size(self, model: tf.keras.Model) -> int:
        shape = model.input_shape
        if isinstance(shape, list):
            shape = shape[0]
        if shape and len(shape) >= 3 and shape[1]:
            return int(shape[1])
        return 224

    def _load_bundle(self, cfg: ModelConfig) -> ModelBundle:
        model: Optional[tf.keras.Model] = None
        classes: List[str] = []
        task = cfg.task
        positive_label = cfg.positive_label
        negative_label = cfg.negative_label

        if os.path.exists(cfg.classes_path):
            try:
                with open(cfg.classes_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                classes = list(meta.get("classes", []))
                task = meta.get("task", task)
                positive_label = meta.get("positive_label", positive_label)
                negative_label = meta.get("negative_label", negative_label)
            except Exception:
                classes = []

        if os.path.exists(cfg.model_path):
            try:
                model = tf.keras.models.load_model(cfg.model_path)
            except Exception:
                model = None

        last_conv_layer_name = ""
        if model is not None:
            for layer in reversed(model.layers):
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv_layer_name = layer.name
                    break

        input_size = self._infer_input_size(model) if model is not None else 224
        return ModelBundle(
            config=cfg,
            model=model,
            classes=classes,
            task=task,
            positive_label=positive_label,
            negative_label=negative_label,
            input_size=input_size,
            last_conv_layer_name=last_conv_layer_name,
        )

    def info(self) -> Dict[str, ModelInfo]:
        info: Dict[str, ModelInfo] = {}
        for modality, bundle in self.bundles.items():
            info[modality] = ModelInfo(
                modality=modality,
                available=bundle.model is not None and bool(bundle.classes),
                model_type=os.path.basename(bundle.config.model_path),
                classes=bundle.classes,
                task=bundle.task,
                input_size=bundle.input_size,
            )
        return info

    def predict_with_heatmap(
        self,
        img_batch: np.ndarray,
        *,
        modality: str,
    ) -> Tuple[Dict[str, Any], np.ndarray, int]:
        modality = (modality or "").lower()
        if modality not in self.bundles:
            payload = {
                "label": "Unsupported modality",
                "confidence": 0.0,
                "error": "unsupported_modality",
                "medical_data": {
                    "severity": "Unknown",
                    "recommendation": "Select chest, brain, bone, or knee.",
                    "detected_type": "Unknown",
                    "model_type": "Unavailable",
                    "is_pathological": False,
                },
            }
            return payload, np.zeros((7, 7), dtype=np.float32), 400

        bundle = self.bundles[modality]
        if bundle.model is None or not bundle.classes:
            payload = {
                "label": "Model unavailable",
                "confidence": 0.0,
                "error": "model_unavailable",
                "medical_data": {
                    "severity": "Unknown",
                    "recommendation": "Train and load the model for this modality.",
                    "detected_type": bundle.config.detected_type,
                    "model_type": f"{bundle.config.model_tag}:{os.path.basename(bundle.config.model_path)}",
                    "is_pathological": False,
                },
            }
            return payload, np.zeros((7, 7), dtype=np.float32), 503

        probs = bundle.model.predict(img_batch, verbose=0)[0]
        probs = np.asarray(probs, dtype=np.float32).flatten()
        pred_index = int(np.argmax(probs)) if probs.size > 1 else 0

        threshold = 0.5
        if bundle.task == "binary":
            if probs.size == 1:
                positive_prob = float(probs[0])
                prob_pairs = [
                    {"class": bundle.negative_label, "prob": float(1.0 - positive_prob)},
                    {"class": bundle.positive_label, "prob": positive_prob},
                ]
            else:
                prob_pairs = [
                    {"class": bundle.classes[i], "prob": float(probs[i])}
                    for i in range(min(len(bundle.classes), len(probs)))
                ]
                positive_prob = next(
                    (p["prob"] for p in prob_pairs if p["class"] == bundle.positive_label),
                    float(np.max(probs)),
                )

            label = bundle.positive_label if positive_prob >= threshold else bundle.negative_label
            confidence = max(positive_prob, 1.0 - positive_prob)
            payload = {
                "label": label,
                "confidence": round(confidence * 100.0, 2),
                "medical_data": {
                    "severity": "Review" if positive_prob >= threshold else "Low",
                    "recommendation": (
                        "Potential abnormality detected; clinical review required."
                        if positive_prob >= threshold
                        else "No abnormality exceeds the threshold; clinical review still advised."
                    ),
                    "detected_type": bundle.config.detected_type,
                    "model_type": f"{bundle.config.model_tag}:{os.path.basename(bundle.config.model_path)}",
                    "is_pathological": positive_prob >= threshold,
                    "threshold": threshold,
                    "positive_findings": [bundle.positive_label] if positive_prob >= threshold else [],
                    "probabilities": prob_pairs,
                },
            }
        elif bundle.task == "multilabel":
            positives = [
                bundle.classes[i]
                for i in range(min(len(bundle.classes), len(probs)))
                if probs[i] >= threshold
            ]
            top_label = bundle.classes[pred_index] if pred_index < len(bundle.classes) else "Unknown"
            max_prob = float(np.max(probs)) if probs.size else 0.0
            display_label = top_label if positives else bundle.negative_label
            display_conf = max_prob if positives else (1.0 - max_prob)

            payload = {
                "label": display_label,
                "confidence": round(display_conf * 100.0, 2),
                "medical_data": {
                    "severity": "Review" if positives else "Low",
                    "recommendation": (
                        "One or more findings exceed the threshold; clinical review required."
                        if positives
                        else "No findings exceed the threshold; clinical review still advised."
                    ),
                    "detected_type": bundle.config.detected_type,
                    "model_type": f"{bundle.config.model_tag}:{os.path.basename(bundle.config.model_path)}",
                    "is_pathological": bool(positives),
                    "threshold": threshold,
                    "positive_findings": positives,
                    "top_finding": top_label,
                    "probabilities": [
                        {"class": bundle.classes[i], "prob": float(probs[i])}
                        for i in range(min(len(bundle.classes), len(probs)))
                    ],
                },
            }
        else:  # multiclass
            top_label = bundle.classes[pred_index] if pred_index < len(bundle.classes) else "Unknown"
            max_prob = float(np.max(probs)) if probs.size else 0.0
            is_pathological = top_label != bundle.negative_label
            payload = {
                "label": top_label,
                "confidence": round(max_prob * 100.0, 2),
                "medical_data": {
                    "severity": "Review" if is_pathological else "Low",
                    "recommendation": (
                        "Abnormal knee grade predicted; clinical review required."
                        if is_pathological
                        else "Normal knee grade predicted; clinical review still advised."
                    ),
                    "detected_type": bundle.config.detected_type,
                    "model_type": f"{bundle.config.model_tag}:{os.path.basename(bundle.config.model_path)}",
                    "is_pathological": is_pathological,
                    "positive_findings": [top_label] if is_pathological else [],
                    "top_finding": top_label,
                    "probabilities": [
                        {"class": bundle.classes[i], "prob": float(probs[i])}
                        for i in range(min(len(bundle.classes), len(probs)))
                    ],
                },
            }

        heatmap = compute_gradcam(
            img_batch,
            bundle.model,
            pred_index=pred_index,
            last_conv_layer_name=bundle.last_conv_layer_name or None,
        )
        return payload, heatmap, 200
