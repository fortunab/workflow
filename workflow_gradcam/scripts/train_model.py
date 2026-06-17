import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from workflow_gradcam.data import resolve_dataset_path, make_generators
from workflow_gradcam.model import build_alzheimer_cnn
from workflow_gradcam.utils import set_seed, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=None, help="Folder containing train/ and test/. If omitted, kagglehub downloads the dataset.")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model-out", default="outputs/models/best_model.keras")
    parser.add_argument("--metrics-out", default="outputs/results/train_metrics.json")
    args = parser.parse_args()

    set_seed(42)

    train_dir, test_dir = resolve_dataset_path(args.dataset_path)
    train_data, test_data = make_generators(train_dir, test_dir, img_size=args.img_size, batch_size=args.batch_size)

    num_classes = len(train_data.class_indices)
    model = build_alzheimer_cnn(img_size=args.img_size, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(args.model_out, monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    loss, acc = model.evaluate(test_data, verbose=1)

    payload = {
        "model_path": args.model_out,
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "class_indices": train_data.class_indices,
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }
    save_json(args.metrics_out, payload)

    print("Saved model:", args.model_out)
    print("Saved metrics:", args.metrics_out)


if __name__ == "__main__":
    main()
