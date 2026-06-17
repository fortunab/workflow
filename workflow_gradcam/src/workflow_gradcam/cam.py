import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore


def load_image(path, img_size=224):
    img = tf.keras.utils.load_img(path, target_size=(img_size, img_size))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0), arr


def normalize_cam(cam):
    cam = np.maximum(cam, 0)
    denom = cam.max()
    return cam / (denom + 1e-8) if denom > 0 else cam


def overlay_cam(cam, image, img_size=224, mask_background=True):
    """Overlay CAM heatmap on an RGB image.

    mask_background=True keeps black MRI background black, matching the notebook fix.
    """
    img_uint8 = np.uint8(image * 255)

    cam = cv2.resize(cam, (img_size, img_size))
    cam = np.uint8(255 * cam)

    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_uint8, 0.6, heat, 0.4, 0)

    if not mask_background:
        return blended

    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(blended, blended, mask=mask)


def keras3_linear_modifier(model_instance):
    """Keras 3 compatible replacement for ReplaceToLinear."""
    model_instance.layers[-1].activation = tf.keras.activations.linear


def make_cam_objects(model):
    """Create GradCAM, GradCAM++, and ScoreCAM objects."""
    # Safe warmup avoids lazy-build issues.
    _ = model.predict(np.zeros((1, 224, 224, 3)), verbose=0)

    return {
        "gradcam": Gradcam(model, model_modifier=keras3_linear_modifier, clone=False),
        "gradcam_pp": GradcamPlusPlus(model, model_modifier=keras3_linear_modifier, clone=False),
        "scorecam": Scorecam(model, model_modifier=keras3_linear_modifier, clone=False),
    }


def compute_cams(model, image_batch, class_index):
    score = CategoricalScore(class_index)
    cam_objects = make_cam_objects(model)

    g = normalize_cam(cam_objects["gradcam"](score, image_batch)[0])
    gp = normalize_cam(cam_objects["gradcam_pp"](score, image_batch)[0])
    sc = normalize_cam(cam_objects["scorecam"](score, image_batch)[0])

    return {"gradcam": g, "gradcam_pp": gp, "scorecam": sc}


def plot_single_image_cams(model, image_path, class_names, output_path, img_size=224):
    image_batch, orig = load_image(image_path, img_size=img_size)

    pred = model.predict(image_batch, verbose=0)[0]
    pred_class = int(np.argmax(pred))

    cams = compute_cams(model, image_batch, pred_class)
    overlays = {name: overlay_cam(cam, orig, img_size=img_size) for name, cam in cams.items()}

    true_class = os.path.basename(os.path.dirname(image_path))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(orig)
    plt.title(f"Original\nTrue: {true_class}")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(overlays["gradcam"])
    plt.title(f"GradCAM\nPred: {class_names[pred_class]}")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(overlays["gradcam_pp"])
    plt.title("GradCAM++")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(overlays["scorecam"])
    plt.title("ScoreCAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {
        "image_path": str(image_path),
        "true_class": true_class,
        "predicted_class": class_names[pred_class],
        "probabilities": {name: float(pred[i]) for i, name in enumerate(class_names)},
        "output_path": str(output_path),
    }


def plot_all_class_cams_for_one_image(model, image_path, class_names, output_path, img_size=224):
    image_batch, orig = load_image(image_path, img_size=img_size)
    pred = model.predict(image_batch, verbose=0)[0]
    true_class = os.path.basename(os.path.dirname(image_path))

    cam_objects = make_cam_objects(model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(class_names), 4, figsize=(16, 4 * len(class_names)))

    for idx, cls_name in enumerate(class_names):
        score = CategoricalScore(idx)

        g = normalize_cam(cam_objects["gradcam"](score, image_batch)[0])
        gp = normalize_cam(cam_objects["gradcam_pp"](score, image_batch)[0])
        sc = normalize_cam(cam_objects["scorecam"](score, image_batch)[0])

        overlays = [
            overlay_cam(g, orig, img_size=img_size),
            overlay_cam(gp, orig, img_size=img_size),
            overlay_cam(sc, orig, img_size=img_size),
        ]

        axes[idx, 0].imshow(orig)
        axes[idx, 0].set_title(f"Original\nTrue: {true_class}")
        axes[idx, 0].axis("off")

        titles = [
            f"GradCAM\n{cls_name}\nProb={pred[idx]:.4f}",
            f"GradCAM++\n{cls_name}",
            f"ScoreCAM\n{cls_name}",
        ]

        for col, (overlay, title) in enumerate(zip(overlays, titles), start=1):
            axes[idx, col].imshow(overlay)
            axes[idx, col].set_title(title)
            axes[idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {"image_path": str(image_path), "output_path": str(output_path)}


def plot_one_image_per_class(model, test_dir, class_names, output_path, img_size=224):
    cam_objects = make_cam_objects(model)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(class_names), 4, figsize=(16, 4 * len(class_names)))

    for row_idx, cls_name in enumerate(class_names):
        class_folder = Path(test_dir) / cls_name
        class_images = list(class_folder.glob("*.jpg"))
        if not class_images:
            class_images = list(class_folder.glob("*.png"))
        if not class_images:
            raise FileNotFoundError(f"No images found in {class_folder}")

        img_path = random.choice(class_images)
        image_batch, orig = load_image(img_path, img_size=img_size)
        pred = model.predict(image_batch, verbose=0)[0]
        pred_class = int(np.argmax(pred))

        score = CategoricalScore(row_idx)

        g = normalize_cam(cam_objects["gradcam"](score, image_batch)[0])
        gp = normalize_cam(cam_objects["gradcam_pp"](score, image_batch)[0])
        sc = normalize_cam(cam_objects["scorecam"](score, image_batch)[0])

        overlays = [
            overlay_cam(g, orig, img_size=img_size),
            overlay_cam(gp, orig, img_size=img_size),
            overlay_cam(sc, orig, img_size=img_size),
        ]

        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].set_title(f"Original\nTrue: {cls_name}")
        axes[row_idx, 0].axis("off")

        titles = [
            f"GradCAM\nPred: {class_names[pred_class]}",
            "GradCAM++",
            "ScoreCAM",
        ]

        for col, (overlay, title) in enumerate(zip(overlays, titles), start=1):
            axes[row_idx, col].imshow(overlay)
            axes[row_idx, col].set_title(title)
            axes[row_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return {"output_path": str(output_path)}
