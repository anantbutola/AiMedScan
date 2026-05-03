import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2

CHEXPERT_5 = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def resolve_paths(root: str, paths: List[str]) -> List[str]:
    root = os.path.abspath(root)
    if not paths:
        return []

    sample = paths[0]
    candidate = os.path.join(root, sample)
    if os.path.exists(candidate):
        return [os.path.join(root, p) for p in paths]

    base = os.path.basename(root)
    prefix = f"{base}{os.sep}"
    if sample.startswith(prefix):
        trimmed = [p[len(prefix):] if p.startswith(prefix) else p for p in paths]
        return [os.path.join(root, p) for p in trimmed]

    # Common CheXpert CSV values start with "CheXpert-v1.0-small/" while data_dir
    # may already be that extracted folder or its parent. If so, strip the first segment.
    if os.sep in sample:
        first_seg = sample.split(os.sep, 1)[0]
        alt_sample = sample.split(os.sep, 1)[1]
        alt_candidate = os.path.join(root, alt_sample)
        if first_seg.lower().startswith("chexpert") and os.path.exists(alt_candidate):
            trimmed = [p.split(os.sep, 1)[1] if os.sep in p else p for p in paths]
            return [os.path.join(root, p) for p in trimmed]

    return [os.path.join(root, p) for p in paths]


def load_chexpert_csv(
    csv_path: str,
    data_dir: str,
    labels: List[str],
    uncertain_policy: str,
) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"]

    if "Path" not in df.columns:
        raise ValueError("CheXpert CSV missing Path column.")

    label_df = df[labels].copy()
    label_df = label_df.fillna(0)
    if uncertain_policy == "u-zeros":
        label_df = label_df.replace(-1, 0)
    elif uncertain_policy == "u-ones":
        label_df = label_df.replace(-1, 1)
    else:
        raise ValueError("Unsupported uncertain label policy.")

    paths = resolve_paths(data_dir, df["Path"].astype(str).tolist())
    labels_arr = label_df.values.astype(np.float32)
    return paths, labels_arr


def filter_existing_paths(paths: List[str], labels: np.ndarray, split_name: str) -> Tuple[List[str], np.ndarray]:
    keep_idx = [i for i, p in enumerate(paths) if os.path.exists(p)]
    missing = len(paths) - len(keep_idx)
    if missing:
        print(f"[WARN] Skipping {missing} missing {split_name} image path(s).")
    if not keep_idx:
        raise RuntimeError(f"No existing {split_name} image paths found after path resolution.")
    kept_paths = [paths[i] for i in keep_idx]
    kept_labels = labels[keep_idx]
    return kept_paths, kept_labels


def build_dataset(
    paths: List[str],
    labels: np.ndarray,
    *,
    img_size: int,
    batch_size: int,
    training: bool,
    augment: bool,
    seed: int,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 10000), seed=seed, reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)

        if training and augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.05)

        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int, num_labels: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = mobilenet_v2.preprocess_input(inputs)
    base = mobilenet_v2.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_backbone",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation="sigmoid", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model, learning_rate: float, num_labels: int) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=num_labels),
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CheXpert 5-label classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to CheXpert-v1.0[-small] root directory.")
    parser.add_argument("--out-dir", default="models", help="Output directory for model + classes.json.")
    parser.add_argument("--model-name", default="chest_classifier.keras", help="Output model filename.")
    parser.add_argument("--classes-name", default="chest_classes.json", help="Output classes filename.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--uncertain-policy", choices=["u-zeros", "u-ones"], default="u-zeros")
    parser.add_argument("--max-train", type=int, default=0, help="Limit training rows (0 = full).")
    parser.add_argument("--max-val", type=int, default=0, help="Limit validation rows (0 = full).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation.")
    args = parser.parse_args()

    train_csv = os.path.join(args.data_dir, "train.csv")
    valid_csv = os.path.join(args.data_dir, "valid.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv in {args.data_dir}")
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Missing valid.csv in {args.data_dir}")

    train_paths, train_labels = load_chexpert_csv(
        train_csv,
        args.data_dir,
        CHEXPERT_5,
        args.uncertain_policy,
    )
    val_paths, val_labels = load_chexpert_csv(
        valid_csv,
        args.data_dir,
        CHEXPERT_5,
        args.uncertain_policy,
    )

    if args.max_train > 0:
        train_paths = train_paths[: args.max_train]
        train_labels = train_labels[: args.max_train]
    if args.max_val > 0:
        val_paths = val_paths[: args.max_val]
        val_labels = val_labels[: args.max_val]

    train_paths, train_labels = filter_existing_paths(train_paths, train_labels, "training")
    val_paths, val_labels = filter_existing_paths(val_paths, val_labels, "validation")

    train_ds = build_dataset(
        train_paths,
        train_labels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        training=True,
        augment=not args.no_augment,
        seed=args.seed,
    )
    val_ds = build_dataset(
        val_paths,
        val_labels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        training=False,
        augment=False,
        seed=args.seed,
    )

    model = build_model(args.img_size, len(CHEXPERT_5))
    compile_model(model, args.learning_rate, len(CHEXPERT_5))
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune_epochs > 0:
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        compile_model(model, args.fine_tune_lr, len(CHEXPERT_5))
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, args.model_name)
    model.save(model_path)

    classes_path = os.path.join(args.out_dir, args.classes_name)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": CHEXPERT_5,
                "task": "multilabel",
                "modality": "chest",
                "negative_label": "No Finding",
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved classes to {classes_path}")


if __name__ == "__main__":
    main()
