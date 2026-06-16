from dataclasses import dataclass
from pathlib import Path
import yaml
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

@dataclass
class Detection:
    label: str
    confidence: float
    box_xyxy: list

def load_config(path="configs/pipeline.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_detection(image_path, weights, class_name="polyp", conf=0.25, iou=0.50):
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. "
            "Train the model first and copy best.pt to models/yolo/best.pt."
        )

    model = YOLO(str(weights))
    results = model.predict(source=str(image_path), conf=conf, iou=iou, verbose=False)
    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            score = float(box.conf[0].detach().cpu().item())
            detections.append(Detection(label=class_name, confidence=score, box_xyxy=xyxy))

    return detections

def fallback_bbox_segmentation(image_path, detections):
    """
    Reproducible fallback segmentation.
    It creates rectangular masks from detection boxes.
    Replace this function with SAM2.1 + LoRA inference for article-level segmentation.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    masks = []

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        area_ratio = float(mask.sum() / 255.0 / (h * w))
        masks.append({"area_ratio": area_ratio})

    return masks

def build_tokens(detections, masks):
    tokens = []

    if detections:
        avg_conf = sum(d.confidence for d in detections) / len(detections)
        level = "high" if avg_conf >= 0.75 else "medium" if avg_conf >= 0.45 else "low"
        tokens.append(f"<DET count='{len(detections)}' avg_conf='{avg_conf:.3f}' level='{level}'>")

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = [round(v, 2) for v in d.box_xyxy]
            tokens.append(
                f"<BOX id='{i}' label='{d.label}' conf='{d.confidence:.3f}' "
                f"x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' />"
            )
    else:
        tokens.append("<DET count='0' avg_conf='0.000' level='none'>")

    if masks:
        avg_area = sum(m["area_ratio"] for m in masks) / len(masks)
        tokens.append(f"<SEG masks='{len(masks)}' avg_area='{avg_area:.4f}' />")
    else:
        tokens.append("<SEG masks='0' avg_area='0.0000' />")

    if not detections:
        tokens.append("<UNCERTAINTY level='high' reason='no_detection' />")
    elif any(d.confidence < 0.45 for d in detections):
        tokens.append("<UNCERTAINTY level='medium' reason='low_confidence_detection' />")
    else:
        tokens.append("<UNCERTAINTY level='low' reason='consistent_detection' />")

    return tokens

def generate_template_report(detections, masks):
    if not detections:
        return (
            "No polyp-like region was detected. "
            "The result is uncertain and additional clinical review is recommended."
        )

    best = max(detections, key=lambda d: d.confidence)

    area_sentence = ""
    if masks:
        area_sentence = f" Approximate segmented area ratio: {masks[0]['area_ratio']:.4f}."

    return (
        f"A polyp-like region was detected with confidence {best.confidence:.3f}. "
        f"The detected bounding box is {best.box_xyxy}."
        f"{area_sentence} "
        "This output is intended for decision support and must not be used as a standalone diagnosis."
    )

def save_visualization(image_path, detections, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.box_xyxy]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{d.label} {d.confidence:.2f}",
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return str(out_path)

def run_pipeline(image_path, config_path="configs/pipeline.yaml"):
    cfg = load_config(config_path)

    detections = run_detection(
        image_path=image_path,
        weights=cfg["yolo_weights"],
        class_name=cfg.get("class_name", "polyp"),
        conf=cfg.get("confidence_threshold", 0.25),
        iou=cfg.get("iou_threshold", 0.50)
    )

    masks = fallback_bbox_segmentation(image_path, detections)
    tokens = build_tokens(detections, masks)
    report = generate_template_report(detections, masks)

    visualization = save_visualization(
        image_path=image_path,
        detections=detections,
        out_path=Path("outputs") / f"{Path(image_path).stem}_prediction.jpg"
    )

    return {
        "detections": [d.__dict__ for d in detections],
        "tokens": tokens,
        "report": report,
        "visualization": visualization
    }
