import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2


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


def find_split_dir(base: str, candidates: list[str]) -> str:
    for name in candidates:
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Missing expected split dirs under {base}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train brain MRI tumor classifier (BraTS + IXI).")
    parser.add_argument("--data-dir", required=True, help="Path to prepared brain dataset root.")
    parser.add_argument("--out-dir", default="models", help="Output directory for model + classes.json.")
    parser.add_argument("--model-name", default="brain_classifier.keras", help="Output model filename.")
    parser.add_argument("--classes-name", default="brain_classes.json", help="Output classes filename.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = find_split_dir(args.data_dir, ["train", "training"])
    val_dir = find_split_dir(args.data_dir, ["val", "valid", "validation"])

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="binary",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="binary",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        seed=args.seed,
        shuffle=False,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

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
                "classes": ["Tumor"],
                "task": "binary",
                "modality": "brain",
                "positive_label": "Tumor",
                "negative_label": "Normal",
            },
            f,
            indent=2,
        )

    print(f"Saved model to {model_path}")
    print(f"Saved classes to {classes_path}")


if __name__ == "__main__":
    main()
