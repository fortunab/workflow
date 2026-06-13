import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "latency_efficiency.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for variant, metrics in data.items():

    rows.append({
        "Variant": variant,
        "Latency (ms)": metrics["Latency (ms)"],
        "GPU load (%)": metrics["GPU load (%)"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/latency_efficiency.csv",
    index=False
)

print("\nLatency and GPU Efficiency")
print(df.to_string(index=False))

# Latency figure

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["Variant"],
    df["Latency (ms)"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 5,
        f"{int(h)}",
        ha="center"
    )

plt.ylabel("Latency (ms)")
plt.xticks(rotation=20, ha="right")
plt.title("Single-Image Latency")
plt.tight_layout()

plt.savefig(
    "figures/latency_comparison.png",
    dpi=300
)

plt.show()


# GPU load figure

plt.figure(figsize=(8, 5))

bars = plt.bar(
    df["Variant"],
    df["GPU load (%)"]
)

for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h + 1,
        f"{int(h)}%",
        ha="center"
    )

plt.ylabel("GPU load (%)")
plt.ylim(0, 110)

plt.xticks(rotation=20, ha="right")
plt.title("GPU Utilization")

plt.tight_layout()

plt.savefig(
    "figures/gpu_load_comparison.png",
    dpi=300
)

plt.show()