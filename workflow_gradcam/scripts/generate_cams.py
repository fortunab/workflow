import argparse
import glob
import os
import random
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import tensorflow as tf

from workflow_gradcam.data import resolve_dataset_path, make_generators
from workflow_gradcam.cam import (
    plot_single_image_cams,
    plot_all_class_cams_for_one_image,
    plot_one_image_per_class,
)
from workflow_gradcam.utils import set_seed, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="outputs/models/best_model.keras")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, a random test image is selected.")
    parser.add_argument("--mode", default="single", choices=["single", "all-classes", "one-per-class"])
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", default="outputs/figures/gradcam_result.png")
    parser.add_argument("--metadata-out", default="outputs/results/gradcam_metadata.json")
    args = parser.parse_args()

    set_seed(42)

    train_dir, test_dir = resolve_dataset_path(args.dataset_path)
    train_data, _ = make_generators(train_dir, test_dir, img_size=args.img_size, batch_size=16)
    class_names = list(train_data.class_indices.keys())

    model = tf.keras.models.load_model(args.model_path)

    if args.mode == "one-per-class":
        result = plot_one_image_per_class(
            model=model,
            test_dir=test_dir,
            class_names=class_names,
            output_path=args.output,
            img_size=args.img_size,
        )
    else:
        image_path = args.image
        if image_path is None:
            all_images = glob.glob(str(Path(test_dir) / "*" / "*.jpg"))
            if not all_images:
                all_images = glob.glob(str(Path(test_dir) / "*" / "*.png"))
            if not all_images:
                raise FileNotFoundError(f"No test images found under {test_dir}")
            image_path = random.choice(all_images)

        if args.mode == "single":
            result = plot_single_image_cams(
                model=model,
                image_path=image_path,
                class_names=class_names,
                output_path=args.output,
                img_size=args.img_size,
            )
        else:
            result = plot_all_class_cams_for_one_image(
                model=model,
                image_path=image_path,
                class_names=class_names,
                output_path=args.output,
                img_size=args.img_size,
            )

    save_json(args.metadata_out, result)
    print("Saved figure:", args.output)
    print("Saved metadata:", args.metadata_out)


if __name__ == "__main__":
    main()
