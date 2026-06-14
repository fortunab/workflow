import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

CONFIG = ROOT / "configs" / "pipeline_stage_latency.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

latency_df = pd.DataFrame({
    "Stage": list(data["latency"].keys()),
    "Latency (ms)": list(data["latency"].values())
})

gpu_df = pd.DataFrame({
    "Stage": list(data["gpu_load"].keys()),
    "GPU load (%)": list(data["gpu_load"].values())
})

latency_df.to_csv(
    ROOT / "results" / "pipeline_stage_latency.csv",
    index=False
)

gpu_df.to_csv(
    ROOT / "results" / "pipeline_stage_gpu.csv",
    index=False
)

print("\nPipeline stage latency")
print(latency_df.to_string(index=False))

print("\nPipeline stage GPU load")
print(gpu_df.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 6)
)

bars = ax1.bar(
    latency_df["Stage"],
    latency_df["Latency (ms)"]
)

for bar in bars:

    h = bar.get_height()

    ax1.text(
        bar.get_x() + bar.get_width()/2,
        h + 5,
        f"{int(h)}",
        ha="center",
        fontsize=8
    )

ax1.set_ylabel("Latency (ms)")
ax1.set_ylim(0, 500)
ax1.set_title("Per-stage latency")

x = range(len(gpu_df))

ax2.plot(
    x,
    gpu_df["GPU load (%)"],
    marker="o"
)

ax2.fill_between(
    x,
    gpu_df["GPU load (%)"],
    alpha=0.25
)

for i, value in enumerate(gpu_df["GPU load (%)"]):

    ax2.text(
        i,
        value + 2,
        str(value),
        ha="center",
        fontsize=8
    )

ax2.set_xticks(list(x))
ax2.set_xticklabels(gpu_df["Stage"])

ax2.set_ylabel("Relative GPU load (%)")
ax2.set_ylim(0, 100)

plt.tight_layout()

plt.savefig(
    ROOT / "figures" / "pipeline_stage_latency.png",
    dpi=300
)

plt.show()
