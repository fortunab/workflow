import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

CONFIG = ROOT / "configs" / "table4_vqa_large.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    evaluation = json.load(f)

rows = []

for metric, values in evaluation.items():
    rows.append({
        "Metric": metric,
        "Base model": values["Base model"],
        "Fine-tuned qwen3.5-0.8B-LoRA": values["Fine-tuned qwen3.5-0.8B-LoRA"],
        "Transformer VQA": values["Transformer VQA"],
        "MedFuseNet": values["MedFuseNet"]
    })

df = pd.DataFrame(rows)

df.to_csv("results/table4_vqa_large.csv", index=False)

print("\nTable 4 - Evaluation on 10000 samples (500 steps)")
print(df.to_string(index=False))

plot_df = df.set_index("Metric")

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 5)
)

ax.set_ylabel("Score")
ax.set_xlabel("Metric")
ax.set_title("Evaluation on 10000 Samples (500 Steps)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=2)

plt.tight_layout()
plt.savefig("figures/table4_vqa_large.png", dpi=300)
plt.show()