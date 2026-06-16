import argparse
import json
from pathlib import Path
import requests

from utils_orchestrator import load_config, save_json
from token_builder import build_tokens
from reasoning_template import generate_report

def call_detection_api(api_url, image_path):
    with open(image_path, "rb") as f:
        response = requests.post(api_url, files={"file": f}, timeout=180)
    response.raise_for_status()
    return response.json()

def fallback_detection():
    return {
        "detections": [
            {"label": "polyp", "confidence": 0.91, "box_xyxy": [51, 42, 248, 219]}
        ]
    }

def fallback_segmentation(detections):
    if not detections:
        return []
    return [{"area_ratio": 0.0732}]

def run_orchestration(image_path=None, config_path="configs/orchestrator.yaml", use_demo=False):
    cfg = load_config(config_path)

    if use_demo or image_path is None:
        det_payload = fallback_detection()
    else:
        det_payload = call_detection_api(cfg["detection_api"], image_path)

    detections = det_payload.get("detections", [])

    # Selective execution rule:
    # if no detection or all detections are below threshold, skip segmentation.
    threshold = float(cfg.get("confidence_threshold", 0.45))
    enable_selective = bool(cfg.get("enable_selective_execution", True))

    if enable_selective and (not detections or max(float(d.get("confidence", 0.0)) for d in detections) < threshold):
        segmentations = []
    else:
        # Here you can call a segmentation API. For reproducibility, a fallback is used.
        segmentations = fallback_segmentation(detections)

    tokens = build_tokens(
        detections=detections,
        segmentations=segmentations,
        low=float(cfg.get("confidence_threshold", 0.45)),
        high=float(cfg.get("high_confidence_threshold", 0.75)),
    )

    # Here you can call Qwen/VLM reasoning. For reproducibility, a template is used.
    report = generate_report(tokens)

    result = {
        "detections": detections,
        "segmentations": segmentations,
        "tokens": tokens,
        "report": report
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--config", default="configs/orchestrator.yaml")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--out", default="outputs/orchestration_result.json")
    args = parser.parse_args()

    result = run_orchestration(args.image, args.config, args.demo)
    save_json(args.out, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
