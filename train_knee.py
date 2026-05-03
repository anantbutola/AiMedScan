import argparse
import json
import os
import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2


def parse_class_token(dirname: str) -> Tuple[int, str] | None:
    token = dirname.strip()
    if not token:
        return None
    first = token[0]
    if not first.isdigit():
        return None
    idx = int(first)
    name = token[1:].strip() or f"Class{idx}"
    return idx, name


def collect_labeled_images(data_dir: str) -> Tuple[List[str], List[int], List[str]]:
    classes: dict[int, str] = {}
    items: List[Tuple[str, int]] = []

    for root, _, files in os.walk(data_dir):
        class_info = parse_class_token(os.path.basename(root))
        if class_info is None:
            continue
        class_idx, class_name = class_info
        classes[class_idx] = class_name
        for fname in files:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                items.append((os.path.join(root, fname), class_idx))

    if not items:
        raise FileNotFoundError(
            f"No labeled knee images found in {data_dir}. "
            "Expected folders like 0Normal, 1Doubtful, 2Mild, 3Moderate, 4Severe."
        )

    if not classes:
        raise FileNotFoundError("No class folders found for knee dataset.")

    class_keys = sorted(classes.keys())
    index_map = {k: i for i, k in enumerate(class_keys)}
    class_names = [f"{k}{classes[k]}" for k in class_keys]
    remapped = [(p, index_map[k]) for p, k in items]
    paths = [p for p, _ in remapped]
    labels = [y for _, y in remapped]
    return paths, labels, class_names


def split_dataset(paths: List[str], labels: List[int], val_split: float, seed: int):
    pairs = list(zip(paths, labels))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    val_count = int(len(pairs) * val_split)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    train_paths = [p for p, _ in train_pairs]
    train_labels = [y for _, y in train_pairs]
    val_paths = [p for p, _ in val_pairs]
    val_labels = [y for _, y in val_pairs]
    return train_paths, train_labels, val_paths, val_labels


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

    def _load(path: tf.Tensor, label: tf.Tensor):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        if training and augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.05)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(img_size: int, num_classes: int) -> tf.keras.Model:
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train knee OA (KL grade) classifier.")
    parser.add_argument("--data-dir", required=True, help="Path to extracted knee dataset root.")
    parser.add_argument("--out-dir", default="models", help="Output directory for model + classes.json.")
    parser.add_argument("--model-name", default="knee_classifier.keras", help="Output model filename.")
    parser.add_argument("--classes-name", default="knee_classes.json", help="Output classes filename.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    paths, labels, class_names = collect_labeled_images(args.data_dir)
    train_paths, train_labels, val_paths, val_labels = split_dataset(paths, labels, args.val_split, args.seed)
    if not train_paths or not val_paths:
        raise RuntimeError("Empty train/val split. Adjust --val-split or provide more images.")

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

    model = build_model(args.img_size, len(class_names))
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
                "classes": class_names,
                "task": "multiclass",
                "modality": "knee",
                "negative_label": "0Normal",
            },
            f,
            indent=2,
        )

    print(f"Samples: train={len(train_paths)}, val={len(val_paths)}")
    print(f"Classes: {class_names}")
    print(f"Saved model to {model_path}")
    print(f"Saved classes to {classes_path}")


if __name__ == "__main__":
    main()
