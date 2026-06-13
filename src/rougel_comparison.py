import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "figure8_rougel_comparison.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    rougel = json.load(f)

df = pd.DataFrame({
    "Model": rougel.keys(),
    "ROUGE-L": rougel.values()
})

df.to_csv(
    "results/figure8_rougel_comparison.csv",
    index=False
)

print("\nFigure 8 - ROUGE-L comparison")
print(df.to_string(index=False))

plt.figure(figsize=(8,5))

bars = plt.bar(
    df["Model"],
    df["ROUGE-L"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        fontsize=9
    )

plt.ylabel("ROUGE-L")
plt.ylim(0, 0.8)

plt.xticks(
    rotation=15,
    ha="right"
)

plt.title(
    "ROUGE-L Comparison Across Models"
)

plt.tight_layout()

plt.savefig(
    "figures/figure8_rougel_comparison.png",
    dpi=300
)

plt.show()