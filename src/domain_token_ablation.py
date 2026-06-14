import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.makedirs(ROOT / "results", exist_ok=True)
os.makedirs(ROOT / "figures", exist_ok=True)

CONFIG = ROOT / "configs" / "domain_token_ablation.json"

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

# -------------------------------------
# Export CSV
# -------------------------------------

rows = []

for domain, values in data["accuracy"].items():

    rows.append({
        "Domain": domain,
        "Metric": "Accuracy",
        "Without tokens": values["Without tokens"],
        "With tokens": values["With tokens"]
    })

for domain, values in data["rougel"].items():

    rows.append({
        "Domain": domain,
        "Metric": "ROUGE-L",
        "Without tokens": values["Without tokens"],
        "With tokens": values["With tokens"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    ROOT / "results" / "domain_token_ablation.csv",
    index=False
)

print("\nDomain Token Ablation")
print(df.to_string(index=False))

# -------------------------------------
# Figure
# -------------------------------------

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(8, 6)
)

domains = list(data["accuracy"].keys())
x = np.arange(len(domains))
width = 0.18

# -------------------------------------
# Accuracy panel
# -------------------------------------

acc_without = [
    data["accuracy"][d]["Without tokens"]
    for d in domains
]

acc_with = [
    data["accuracy"][d]["With tokens"]
    for d in domains
]

bars1 = ax1.bar(
    x - width/2,
    acc_without,
    width,
    label="Pipeline w/o tokens"
)

bars2 = ax1.bar(
    x + width/2,
    acc_with,
    width,
    label="Pipeline with tokens"
)

for bars in [bars1, bars2]:

    for bar in bars:

        h = bar.get_height()

        ax1.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.005,
            f"{h:.2f}",
            ha="center",
            fontsize=8
        )

ax1.set_ylabel("End-to-end accuracy")
ax1.set_ylim(0.6, 1.0)
ax1.set_xticks(x)
ax1.set_xticklabels(domains)
ax1.legend()

# -------------------------------------
# ROUGE-L panel
# -------------------------------------

rouge_without = [
    data["rougel"][d]["Without tokens"]
    for d in domains
]

rouge_with = [
    data["rougel"][d]["With tokens"]
    for d in domains
]

bars1 = ax2.bar(
    x - width/2,
    rouge_without,
    width,
    label="Pipeline w/o tokens"
)

bars2 = ax2.bar(
    x + width/2,
    rouge_with,
    width,
    label="Pipeline with tokens"
)

for bars in [bars1, bars2]:

    for bar in bars:

        h = bar.get_height()

        ax2.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.005,
            f"{h:.2f}",
            ha="center",
            fontsize=8
        )

ax2.set_ylabel("ROUGE-L")
ax2.set_ylim(0.4, 0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(domains)
ax2.legend()

plt.tight_layout()

plt.savefig(
    ROOT / "figures" / "domain_token_ablation.png",
    dpi=300
)

plt.show()