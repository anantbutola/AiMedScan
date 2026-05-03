import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2


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

    return [os.path.join(root, p) for p in paths]


def load_mura_studies(csv_path: str, data_dir: str) -> List[Tuple[str, int]]:
    df = pd.read_csv(csv_path, header=None)
    paths = df.iloc[:, 0].astype(str).tolist()
    paths = resolve_paths(data_dir, paths)

    studies: List[Tuple[str, int]] = []
    for path in paths:
        label = 1 if "positive" in path.lower() else 0
        studies.append((path, label))
    return studies


def collect_images(studies: List[Tuple[str, int]]) -> Tuple[List[str], List[int]]:
    image_paths: List[str] = []
    labels: List[int] = []
    for study_path, label in studies:
        if not os.path.exists(study_path):
            continue
        for root, _, files in os.walk(study_path):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_paths.append(os.path.join(root, fname))
                    labels.append(label)
    return image_paths, labels


def filter_readable_images(paths: List[str], labels: List[int], split_name: str) -> Tuple[List[str], List[int]]:
    valid_paths: List[str] = []
    valid_labels: List[int] = []
    invalid_paths: List[str] = []

    for path, label in zip(paths, labels):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            invalid_paths.append(path)
            continue
        valid_paths.append(path)
        valid_labels.append(label)

    if invalid_paths:
        preview = "\n".join(invalid_paths[:5])
        print(
            f"[WARN] Skipping {len(invalid_paths)} unreadable {split_name} image(s). "
            f"First few paths:\n{preview}"
        )

    if not valid_paths:
        raise RuntimeError(f"No readable {split_name} images found.")

    return valid_paths, valid_labels


def build_dataset(
    paths: List[str],
    labels: List[int],
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
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)

        if training and augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.05)

        return img, tf.cast(label, tf.float32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int) -> tf.keras.Model:
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
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MURA abnormal/normal classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to MURA-v1.1 root directory.")
    parser.add_argument("--out-dir", default="models", help="Output directory for model + classes.json.")
    parser.add_argument("--model-name", default="bone_classifier.keras", help="Output model filename.")
    parser.add_argument("--classes-name", default="bone_classes.json", help="Output classes filename.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation.")
    parser.add_argument("--max-train", type=int, default=0, help="Limit training images (0 = full).")
    parser.add_argument("--max-val", type=int, default=0, help="Limit validation images (0 = full).")
    args = parser.parse_args()

    train_csv = os.path.join(args.data_dir, "train.csv")
    valid_csv = os.path.join(args.data_dir, "valid.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv in {args.data_dir}")
    if not os.path.exists(valid_csv):
        raise FileNotFoundError(f"Missing valid.csv in {args.data_dir}")

    train_studies = load_mura_studies(train_csv, args.data_dir)
    val_studies = load_mura_studies(valid_csv, args.data_dir)
    train_paths, train_labels = collect_images(train_studies)
    val_paths, val_labels = collect_images(val_studies)

    if not train_paths:
        raise RuntimeError(f"No training images found under {args.data_dir}")
    if not val_paths:
        raise RuntimeError(f"No validation images found under {args.data_dir}")

    if args.max_train > 0:
        train_paths = train_paths[: args.max_train]
        train_labels = train_labels[: args.max_train]
    if args.max_val > 0:
        val_paths = val_paths[: args.max_val]
        val_labels = val_labels[: args.max_val]

    train_paths, train_labels = filter_readable_images(train_paths, train_labels, "training")
    val_paths, val_labels = filter_readable_images(val_paths, val_labels, "validation")

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

    model = build_model(args.img_size)
    compile_model(model, args.learning_rate)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    if args.fine_tune_epochs > 0:
        backbone = model.get_layer("mobilenetv2_backbone")
        backbone.trainable = True
        for layer in backbone.layers[:-20]:
            layer.trainable = False
        compile_model(model, args.fine_tune_lr)
        model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, args.model_name)
    model.save(model_path)

    classes_path = os.path.join(args.out_dir, args.classes_name)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": ["Abnormal"],
                "task": "binary",
                "modality": "bone",
                "positive_label": "Abnormal",
                "negative_label": "Normal",
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved classes to {classes_path}")


if __name__ == "__main__":
    main()
