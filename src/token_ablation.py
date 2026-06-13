import os
import json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "token_ablation.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)


# --------------------------
# Export CSV
# --------------------------

rows = []

for metric in ["accuracy", "rougel"]:

    for variant, value in data[metric].items():

        rows.append({
            "Metric": metric,
            "Variant": variant,
            "Value": value
        })

for variant, value in data["latency"].items():

    rows.append({
        "Metric": "latency",
        "Variant": variant,
        "Value": value
    })

df = pd.DataFrame(rows)

df.to_csv(
    "results/token_ablation.csv",
    index=False
)

print("\nToken Ablation")
print(df.to_string(index=False))

# --------------------------
# Figure
# --------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9, 4)
)

# --------------------------
# Accuracy + ROUGE-L
# --------------------------

ax = axes[0]

labels = ["End-to-end Acc.", "ROUGE-L"]

without_tokens = [
    data["accuracy"]["Pipeline w/o tokens"],
    data["rougel"]["Pipeline w/o tokens"]
]

with_tokens = [
    data["accuracy"]["Pipeline with tokens"],
    data["rougel"]["Pipeline with tokens"]
]

x = range(len(labels))
width = 0.35

bars1 = ax.bar(
    [i - width/2 for i in x],
    without_tokens,
    width,
    label="Pipeline w/o tokens"
)

bars2 = ax.bar(
    [i + width/2 for i in x],
    with_tokens,
    width,
    label="Pipeline with tokens"
)

for b in list(bars1) + list(bars2):

    h = b.get_height()

    ax.text(
        b.get_x() + b.get_width()/2,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        fontsize=8,
        rotation=90
    )

ax.set_xticks(list(x))
ax.set_xticklabels(labels)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)

ax.legend(fontsize=8)

ax.set_title("(a) Accuracy and ROUGE-L")

# --------------------------
# Latency
# --------------------------

ax = axes[1]

latency_labels = list(data["latency"].keys())
latency_values = list(data["latency"].values())

bars = ax.bar(
    latency_labels,
    latency_values
)

for b in bars:

    h = b.get_height()

    ax.text(
        b.get_x() + b.get_width()/2,
        h - 25,
        f"{int(h)}",
        ha="center",
        fontsize=9
    )

ax.set_ylabel("Latency (ms)")
ax.set_ylim(0, 600)

ax.set_title("(b) Latency")

plt.tight_layout()

plt.savefig(
    "figures/token_ablation.png",
    dpi=300
)

plt.show()