import os
import json
import pandas as pd

os.makedirs("results", exist_ok=True)

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

config_file = ROOT / "configs" / "table_segmentation.json"

with open(config_file, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for setting, methods in data.items():
    for method, metrics in methods.items():
        rows.append({
            "Train/Test": setting,
            "Method": method,
            "mIoU": metrics["mIoU"],
            "Dice": metrics["Dice"],
            "mAP": metrics["mAP"]
        })

df = pd.DataFrame(rows)

df.to_csv(
    "results/table_segmentation.csv",
    index=False
)

print("\nSegmentation Performance Table")
print(df.to_string(index=False))