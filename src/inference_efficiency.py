import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "table_inference_efficiency.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for method, metrics in data.items():

    rows.append({
        "Method": method,
        "Latency (ms)": metrics["Latency (ms)"],
        "GPU mem (MB)": metrics["GPU mem (MB)"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/table2_inference_efficiency.csv",
    index=False
)

print("\nTable 2 - Inference Efficiency")
print(df.to_string(index=False))

# Latency figure

plt.figure(figsize=(8,4))

bars = plt.bar(
    df["Method"],
    df["Latency (ms)"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 1,
        f"{h:.0f}",
        ha="center"
    )

plt.ylabel("Latency (ms)")
plt.xticks(rotation=20, ha="right")
plt.title("Inference Latency")
plt.tight_layout()

plt.savefig(
    "figures/inference_latency.png",
    dpi=300
)

plt.show()

# GPU memory figure

plt.figure(figsize=(8,4))

bars = plt.bar(
    df["Method"],
    df["GPU mem (MB)"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 50,
        f"{int(h)}",
        ha="center"
    )

plt.ylabel("GPU memory (MB)")
plt.xticks(rotation=20, ha="right")
plt.title("Peak GPU Memory Usage")
plt.tight_layout()

plt.savefig(
    "figures/gpu_memory_usage.png",
    dpi=300
)

plt.show()