import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "figure9_bleu_smoothed.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    bleu = json.load(f)


df = pd.DataFrame({
    "Model": bleu.keys(),
    "BLEU (smoothed)": bleu.values()
})

df.to_csv("results/figure9_bleu_smoothed.csv", index=False)

print("\nFigure 9 - Smoothed BLEU comparison")
print(df.to_string(index=False))

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["Model"],
    df["BLEU (smoothed)"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        fontsize=9
    )

plt.ylabel("BLEU (smoothed)")
plt.ylim(0, 0.7)
plt.xticks(rotation=15, ha="right")
plt.title("Smoothed BLEU Comparison")

plt.tight_layout()
plt.savefig("figures/figure9_bleu_smoothed.png", dpi=300)
plt.show()