import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "cross_dataset_transfer.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for model, metrics in data.items():

    rows.append({
        "Model": model,
        "mIoU": metrics["mIoU"],
        "Dice": metrics["Dice"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/cross_dataset_transfer.csv",
    index=False
)

print("\nCross-Dataset Transfer Evaluation")
print(df.to_string(index=False))

models = df["Model"]

miou = df["mIoU"]
dice = df["Dice"]

x = np.arange(len(models))
width = 0.32

plt.figure(figsize=(8,5))

bars1 = plt.bar(
    x - width/2,
    miou,
    width,
    label="mIoU"
)

bars2 = plt.bar(
    x + width/2,
    dice,
    width,
    label="Dice"
)

for bars in [bars1, bars2]:

    for bar in bars:

        h = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.003,
            f"{h:.2f}",
            ha="center",
            fontsize=8,
            rotation=90
        )

plt.xticks(
    x,
    models,
    rotation=20,
    ha="right"
)

plt.ylabel("Score")
plt.ylim(0.60, 0.90)

plt.legend()

plt.title(
    "Cross-Dataset Transfer Performance"
)

plt.tight_layout()

plt.savefig(
    "figures/cross_dataset_transfer.png",
    dpi=300
)

plt.show()