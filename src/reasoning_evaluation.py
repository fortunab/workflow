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

CONFIG = ROOT / "configs" / "perception_analysis.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for model, metrics in data.items():

    rows.append({
        "Model": model,
        "BLEU-4": metrics["BLEU-4"],
        "ROUGE-L": metrics["ROUGE-L"],
        "METEOR": metrics["METEOR"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/reasoning_evaluation.csv",
    index=False
)

print("\nReasoning Evaluation")
print(df.to_string(index=False))

models = df["Model"]

bleu = df["BLEU-4"]
rouge = df["ROUGE-L"]
meteor = df["METEOR"]

x = np.arange(len(models))
width = 0.24

plt.figure(figsize=(9,5))

bars1 = plt.bar(
    x - width,
    bleu,
    width,
    label="BLEU-4"
)

bars2 = plt.bar(
    x,
    rouge,
    width,
    label="ROUGE-L"
)

bars3 = plt.bar(
    x + width,
    meteor,
    width,
    label="METEOR"
)

for bars in [bars1, bars2, bars3]:

    for bar in bars:

        h = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.005,
            f"{h:.2f}",
            ha="center",
            fontsize=8,
            rotation=90
        )

plt.xticks(
    x,
    models,
    rotation=15,
    ha="right"
)

plt.ylabel("Score")
plt.ylim(0, 0.9)

plt.legend()

plt.title(
    "VQA and Reasoning Evaluation"
)

plt.tight_layout()

plt.savefig(
    "figures/reasoning_evaluation.png",
    dpi=300
)

plt.show()