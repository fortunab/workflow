import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import Sam2Processor, Sam2Model
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

from utils_segmentation import get_bounding_box, dice_loss

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def make_collate_fn(processor):
    def collate_fn(batch):
        images, masks, boxes = [], [], []
        for item in batch:
            img = item["image"].convert("RGB")
            mask = np.array(item["mask"].convert("L"))
            mask = (mask > 128).astype(np.uint8)
            box = get_bounding_box(mask)

            images.append(np.array(img))
            masks.append(mask)
            boxes.append([box])

        inputs = processor(images=images, input_boxes=boxes, return_tensors="pt")
        inputs["ground_truth_masks"] = torch.from_numpy(np.array(masks)).unsqueeze(1)
        return inputs
    return collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    dataset = load_dataset(cfg["dataset_id"])
    train_ds = dataset["train"]
    val_ds = dataset["validation"] if "validation" in dataset else dataset["test"]

    processor = Sam2Processor.from_pretrained(cfg["model_id"])
    model = Sam2Model.from_pretrained(cfg["model_id"])

    lora_config = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))

    for epoch in range(int(cfg["epochs"])):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()

            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            gt_masks = batch["ground_truth_masks"].to(device).float()

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )

            pred_masks = outputs.pred_masks.squeeze(2)

            # Resize GT if needed
            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                gt_masks_resized = F.interpolate(gt_masks, size=pred_masks.shape[-2:], mode="nearest")
            else:
                gt_masks_resized = gt_masks

            loss = (
                F.binary_cross_entropy_with_logits(pred_masks, gt_masks_resized)
                + dice_loss(pred_masks, gt_masks_resized)
            )

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1} train loss: {train_loss / max(1, len(train_loader)):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                pixel_values = batch["pixel_values"].to(device)
                input_boxes = batch["input_boxes"].to(device)
                gt_masks = batch["ground_truth_masks"].to(device).float()

                outputs = model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                pred_masks = outputs.pred_masks.squeeze(2)

                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    gt_masks = F.interpolate(gt_masks, size=pred_masks.shape[-2:], mode="nearest")

                loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks) + dice_loss(pred_masks, gt_masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} val loss: {val_loss / max(1, len(val_loader)):.4f}")

    out_dir = Path(cfg["lora_output_dir"])
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("LoRA adapter saved to:", out_dir)

if __name__ == "__main__":
    main()
