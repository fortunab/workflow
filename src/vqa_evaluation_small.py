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

CONFIG = ROOT / "configs" / "table_llm_evaluation.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    evaluation = json.load(f)

rows = []

for metric, values in evaluation.items():

    rows.append({
        "Metric": metric,
        "Base model": values["Base model"],
        "Fine-tuned qwen3.5-0.8B-LoRA": values["Fine-tuned qwen3.5-0.8B-LoRA"],
        "MMBERT": values["MMBERT"],
        "MedFuseNet": values["MedFuseNet"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/table3_llm_evaluation.csv",
    index=False
)

print("\nTable 3 - LLM Evaluation")
print(df.to_string(index=False))

# grouped bar chart

plot_df = df.set_index("Metric")

ax = plot_df.plot(
    kind="bar",
    figsize=(10,5)
)

ax.set_ylabel("Score")
ax.set_xlabel("Metric")
ax.set_title("Evaluation on 1000 Samples")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        fontsize=7,
        rotation=90,
        padding=2
    )

plt.tight_layout()

plt.savefig(
    "figures/table3_llm_evaluation.png",
    dpi=300
)

plt.show()