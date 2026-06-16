import argparse
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def convert_to_yolo(size, box):
    width, height = size
    dw = 1.0 / width
    dh = 1.0 / height

    # box format: [xmin, ymin, xmax, ymax]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]

    return x * dw, y * dh, w * dw, h * dh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="halyusuf/PolypGen2.0")
    parser.add_argument("--out", default="yolo_dataset")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.out)
    for split in ["train", "val"]:
        (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    full_ds = load_dataset(args.dataset, split="train")
    split_ds = full_ds.train_test_split(test_size=args.test_size, seed=args.seed)
    split_map = {"train": split_ds["train"], "val": split_ds["test"]}

    for split, ds in split_map.items():
        for i, item in enumerate(tqdm(ds, desc=f"Processing {split}")):
            img = item["image"].convert("RGB")
            img_filename = f"{split}_{i}.jpg"
            img_path = data_dir / split / "images" / img_filename
            img.save(img_path)

            width = item.get("width", img.width)
            height = item.get("height", img.height)

            objects = item.get("objects", {})
            bboxes = objects.get("bbox", [])

            label_path = data_dir / split / "labels" / f"{split}_{i}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for box in bboxes:
                    if len(box) != 4:
                        continue

                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        continue

                    yolo_box = convert_to_yolo((width, height), box)
                    yolo_box = [max(0.0, min(1.0, float(v))) for v in yolo_box]
                    f.write(f"0 {' '.join([f'{v:.6f}' for v in yolo_box])}\n")

    yaml_content = {
        "path": str(data_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "polyp"}
    }

    with open("data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print("Done.")
    print("YOLO dataset:", data_dir.absolute())
    print("Config file: data.yaml")

if __name__ == "__main__":
    main()
