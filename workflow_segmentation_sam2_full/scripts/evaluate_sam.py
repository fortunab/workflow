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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    parser.add_argument("--model_path", default=None, help="Defaults to merged_output_dir from config.")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model_path = args.model_path or cfg["merged_output_dir"]

    dataset = load_dataset(cfg["dataset_id"])
    test_ds = dataset[args.split]

    processor = Sam2Processor.from_pretrained(model_path if model_path else cfg["model_id"])
    model = Sam2Model.from_pretrained(model_path).to(device)
    model.eval()

    all_iou, all_dice = [], []

    with torch.no_grad():
        for item in tqdm(test_ds, desc="Evaluating"):
            image = item["image"].convert("RGB")
            gt_mask = np.array(item["mask"].convert("L"))
            gt_mask = (gt_mask > 128).astype(np.uint8)

            bbox = get_bounding_box(gt_mask)

            inputs = processor(images=image, input_boxes=[[bbox]], return_tensors="pt").to(device)
            outputs = model(**inputs, multimask_output=False)

            pred = torch.sigmoid(outputs.pred_masks.squeeze()).detach().cpu().numpy()
            pred = (pred > float(cfg["threshold"])).astype(np.uint8)

            if pred.shape != gt_mask.shape:
                pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
                pred = F.interpolate(pred_t, size=gt_mask.shape, mode="nearest").squeeze().numpy().astype(np.uint8)

            iou, dice = calculate_metrics(pred, gt_mask)
            all_iou.append(iou)
            all_dice.append(dice)

    print("=" * 30)
    print(f"Mean IoU (mIoU): {np.mean(all_iou):.4f}")
    print(f"Mean Dice Score: {np.mean(all_dice):.4f}")
    print("=" * 30)

if __name__ == "__main__":
    main()
