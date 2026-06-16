import argparse
from utils_orchestrator import confidence_level

def build_tokens(detections=None, segmentations=None, low=0.45, high=0.75):
    detections = detections or []
    segmentations = segmentations or []

    tokens = []

    if detections:
        avg_conf = sum(float(d.get("confidence", 0.0)) for d in detections) / len(detections)
        level = confidence_level(avg_conf, low, high)
        tokens.append(f"<DET count='{len(detections)}' avg_conf='{avg_conf:.3f}' level='{level}'>")

        for i, d in enumerate(detections):
            box = d.get("box_xyxy", [0, 0, 0, 0])
            label = d.get("label", "object")
            conf = float(d.get("confidence", 0.0))
            x1, y1, x2, y2 = [round(float(v), 2) for v in box]
            tokens.append(
                f"<BOX id='{i}' label='{label}' conf='{conf:.3f}' "
                f"x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' />"
            )
    else:
        tokens.append("<DET count='0' avg_conf='0.000' level='none'>")

    if segmentations:
        avg_area = sum(float(s.get("area_ratio", 0.0)) for s in segmentations) / len(segmentations)
        tokens.append(f"<SEG masks='{len(segmentations)}' avg_area='{avg_area:.4f}' />")
    else:
        tokens.append("<SEG masks='0' avg_area='0.0000' />")

    if not detections:
        tokens.append("<UNCERTAINTY level='high' reason='no_detection' />")
    elif any(float(d.get("confidence", 0.0)) < low for d in detections):
        tokens.append("<UNCERTAINTY level='medium' reason='low_confidence_detection' />")
    else:
        tokens.append("<UNCERTAINTY level='low' reason='consistent_detection' />")

    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    detections = [{"label": "polyp", "confidence": 0.91, "box_xyxy": [51, 42, 248, 219]}]
    segmentations = [{"area_ratio": 0.0732}]
    print("\n".join(build_tokens(detections, segmentations)))

if __name__ == "__main__":
    main()
