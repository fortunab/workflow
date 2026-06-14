import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

CONFIG = ROOT / "configs" / "segmentation_ablation.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame({
    "Method": data.keys(),
    "Trainable parameters (M)": data.values()
})

df.to_csv(ROOT / "results" / "segmentation_ablation.csv", index=False)

print("\nSegmentation ablation")
print(df.to_string(index=False))

plt.figure(figsize=(7, 4))

x = range(len(df))

plt.plot(
    x,
    df["Trainable parameters (M)"],
    marker="o"
)

plt.fill_between(
    x,
    df["Trainable parameters (M)"],
    alpha=0.25
)

for i, value in enumerate(df["Trainable parameters (M)"]):
    plt.text(
        i,
        value + 1,
        str(value),
        ha="center",
        fontsize=8
    )

plt.xticks(x, df["Method"])
plt.ylabel("Trainable parameters (M)")
plt.title("Segmentation Ablation")
plt.tight_layout()
plt.savefig(ROOT / "figures" / "segmentation_ablation.png", dpi=300)
plt.show()