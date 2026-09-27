"""Export small animations from the saved training history and one saved run."""
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
BG = "#f5f3ec"


def frame_image(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=70, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE)


def save_training():
    data = json.loads((ROOT / "results/metrics.json").read_text())
    runs = [r for r in data["runs"] if r["kind"] == "connectome"]
    frames = []
    for epoch in range(1, data["epochs"] + 1):
        fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
        ax.set_facecolor(BG)
        for run in runs:
            history = run["history"][:epoch]
            ax.plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
                    alpha=.42, linewidth=1.2)
            ax.plot([h["epoch"] for h in history], [h["validation_loss"] for h in history],
                    linewidth=1.5, label=f"seed {run['seed']} validation")
        ax.set(xlim=(1, data["epochs"]), ylim=(0, .6), xlabel="epoch",
               ylabel="binary cross-entropy", title=f"Imitation training, epoch {epoch}")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
        frames.append(frame_image(fig))
    frames[0].save(ROOT / "figures/training-progress.gif", save_all=True,
                   append_images=frames[1:], duration=110, loop=0, optimize=True)


def save_traffic():
    traces = json.loads((ROOT / "results/traces.json").read_text())
    steps = traces["standard"]["connectome"]["steps"]
    frames = []
    for stop in range(0, len(steps), 8):
        current = steps[:stop + 1]
        t = np.array([step["t"] / 60 for step in current])
        q = np.array([step["q"] for step in current])
        fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
        ax.set_facecolor(BG)
        for i, label in enumerate(("9th & Q", "9th & P", "9th & O")):
            ax.plot(t, q[:, i, :].sum(axis=1), label=label, linewidth=1.7)
        ymax = max(20, float(np.max(q.sum(axis=(1, 2))) * 1.12))
        ax.set(xlim=(0, 60), ylim=(0, ymax), xlabel="minutes",
               ylabel="vehicles queued", title=f"Example connectome-controlled run, t = {t[-1]:.1f} min")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, ncol=3)
        frames.append(frame_image(fig))
    frames[0].save(ROOT / "figures/traffic-playback.gif", save_all=True,
                   append_images=frames[1:], duration=100, loop=0, optimize=True)


if __name__ == "__main__":
    save_training()
    save_traffic()
    print("Wrote training-progress.gif and traffic-playback.gif")
