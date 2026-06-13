import os
import json
import pandas as pd

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "pipeline_metrics.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for use_case, metrics in data.items():

    for metric_name, values in metrics.items():

        rows.append({
            "Use case": use_case,
            "Metric": metric_name,
            "Baselines": values["Baselines"],
            "Ensemble": values["Ensemble"],
            "W/o tokens": values["W/o tokens"],
            "Tokens": values["Tokens"]
        })

df = pd.DataFrame(rows)

df.to_csv(
    "results/pipeline_metrics.csv",
    index=False
)

print("\nPipeline Metrics Comparison")
print(df.to_string(index=False))