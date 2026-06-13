import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "perception_analysis.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for model, metrics in data.items():

    rows.append({
        "Model": model,
        "IoU": metrics["IoU"],
        "Dice": metrics["Dice"],
        "mAP@0.5": metrics["mAP@0.5"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/perception_analysis.csv",
    index=False
)

print("\nPerception Analysis")
print(df.to_string(index=False))

models = df["Model"]

iou = df["IoU"]
dice = df["Dice"]
map50 = df["mAP@0.5"]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(9, 5))

bars1 = plt.bar(
    x - width,
    iou,
    width,
    label="IoU"
)

bars2 = plt.bar(
    x,
    dice,
    width,
    label="Dice"
)

bars3 = plt.bar(
    x + width,
    map50,
    width,
    label="mAP@0.5"
)

for bars in [bars1, bars2, bars3]:

    for bar in bars:

        h = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.002,
            f"{h:.2f}",
            ha="center",
            fontsize=8,
            rotation=90
        )

plt.xticks(
    x,
    models,
    rotation=25,
    ha="right"
)

plt.ylabel("Score")
plt.ylim(0.72, 0.93)

plt.legend()

plt.title(
    "Overall Perception Performance"
)

plt.tight_layout()

plt.savefig(
    "figures/perception_analysis.png",
    dpi=300
)

plt.show()