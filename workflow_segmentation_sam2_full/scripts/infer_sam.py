import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam2Processor, Sam2Model

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_box(box_string):
    values = [float(v.strip()) for v in box_string.split(",")]
    if len(values) != 4:
        raise ValueError("Box must contain four values: x1,y1,x2,y2")
    return values

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--box", required=True, help="Bounding box as x1,y1,x2,y2")
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--out", default="outputs/inference_overlay.png")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = args.model_path or cfg["merged_output_dir"]

    image = Image.open(args.image).convert("RGB")
    bbox = parse_box(args.box)

    processor = Sam2Processor.from_pretrained(model_path)
    model = Sam2Model.from_pretrained(model_path).to(device).eval()

    inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
        pred = torch.sigmoid(outputs.pred_masks.squeeze()).cpu().numpy()
        pred = (pred > float(cfg["threshold"])).astype(np.uint8)

    if pred.shape != (image.height, image.width):
        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred_t, size=(image.height, image.width), mode="nearest").squeeze().numpy().astype(np.uint8)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    show_mask(pred, ax)
    ax.set_title("SAM2.1 + LoRA Prediction")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved:", out_path)

if __name__ == "__main__":
    main()
