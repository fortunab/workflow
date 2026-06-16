import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Sam2Processor, Sam2Model
from tqdm import tqdm

from utils_segmentation import get_bounding_box, calculate_metrics

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def evaluate_model(model, processor, ds, device, threshold):
    model.eval()
    ious, dices = [], []
    with torch.no_grad():
        for item in tqdm(ds, desc="Evaluating model"):
            image = item["image"].convert("RGB")
            gt = np.array(item["mask"].convert("L"))
            gt = (gt > 128).astype(np.uint8)
            bbox = get_bounding_box(gt)

            inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)
            pred = torch.sigmoid(outputs.pred_masks.squeeze()).cpu().numpy()
            pred = (pred > threshold).astype(np.uint8)

            if pred.shape != gt.shape:
                pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred_t, size=gt.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

            iou, dice = calculate_metrics(pred, gt)
            ious.append(iou)
            dices.append(dice)
    return float(np.mean(ious)), float(np.mean(dices))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = load_dataset(cfg["dataset_id"])
    ds = dataset[args.split]
    threshold = float(cfg["threshold"])

    processor_base = Sam2Processor.from_pretrained(cfg["model_id"])
    base_model = Sam2Model.from_pretrained(cfg["model_id"]).to(device)
    base_iou, base_dice = evaluate_model(base_model, processor_base, ds, device, threshold)

    processor_lora = Sam2Processor.from_pretrained(cfg["merged_output_dir"])
    lora_model = Sam2Model.from_pretrained(cfg["merged_output_dir"]).to(device)
    lora_iou, lora_dice = evaluate_model(lora_model, processor_lora, ds, device, threshold)

    print("\nBase SAM2.1")
    print(f"mIoU: {base_iou:.4f}")
    print(f"Dice: {base_dice:.4f}")

    print("\nSAM2.1 + LoRA")
    print(f"mIoU: {lora_iou:.4f}")
    print(f"Dice: {lora_dice:.4f}")

if __name__ == "__main__":
    main()
