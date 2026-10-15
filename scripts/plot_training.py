"""Plot recorded measurements. Run with: uv run --with matplotlib --no-project python scripts/plot_training.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path("reports/tinystories-core")
rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
evaluations = [row for row in rows if row["event"] == "validation"]
training = [row for row in rows if row["event"] == "train"]
summary = json.loads((directory / "summary.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
for axis, key, label in zip(axes, ["step", "elapsed_seconds"], ["Optimizer steps", "Elapsed seconds"]):
    axis.plot([row[key] for row in training], [row["loss"] for row in training],
              color="#94a3b8", linewidth=1, alpha=0.7, label="Training batch loss (window mean)")
    for split, name, color in (("train", "Train: fixed 32,768 targets", "#2463a6"),
                               ("valid", "Validation: fixed 32,768 targets", "#d26132")):
        axis.plot([row[key] for row in evaluations], [row[split]["loss"] for row in evaluations],
                  marker="o", markersize=3, linewidth=2, color=color, label=name)
    axis.scatter([summary[key]], [summary["valid"]["loss"]], color="#9b3820", marker="*", s=110,
                 label="Final: all 202,721 validation targets", zorder=5)
    axis.set(xlabel=label, ylabel="Cross-entropy (nats / token)", ylim=(2, 8.7))
    axis.grid(alpha=0.18)
    axis.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("TinyStories subset · 3.34M parameters · Apple M5 / MPS", fontsize=14, fontweight="bold")
for extension in ("png", "svg"):
    fig.savefig(directory / f"loss-curve.{extension}", dpi=180)
