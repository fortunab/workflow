import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

CONFIG = ROOT / "configs" / "uncertainty_behavior.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame({
    "Subset": data.keys(),
    "End-to-end accuracy": data.values()
})

df.to_csv(ROOT / "results" / "uncertainty_behavior.csv", index=False)

print("\nUncertainty-aware behavior")
print(df.to_string(index=False))

plt.figure(figsize=(7, 4))

bars = plt.bar(df["Subset"], df["End-to-end accuracy"])

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.02,
        f"{h:.2f}",
        ha="center",
        fontsize=8
    )

plt.ylabel("Accuracy (end-to-end)")
plt.ylim(0, 1.0)
plt.title("Uncertainty-aware Behavior")
plt.tight_layout()
plt.savefig(ROOT / "figures" / "uncertainty_behavior.png", dpi=300)
plt.show()