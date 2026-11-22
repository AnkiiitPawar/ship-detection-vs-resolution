"""Chart the detector's score (mAP50) against camera ground resolution, from the notebook's results.csv.

Usage:
    python plot_results.py results/results.csv results/

Writes score_vs_resolution.png (light background) and score_vs_resolution_dark.png (dark background).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BOAT_M = 30   # a 30 m fishing boat, to show how many pixels long a small ship is at each resolution

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "series": "#3987e5"},
}


def metres(label):
    """'1.5 m' -> 1.5"""
    return float(str(label).replace("m", "").strip())


def draw(results, out_path, theme):
    c = THEMES[theme]
    x = list(range(len(results)))
    y = results["mAP50"].tolist()
    gsd = [metres(r) for r in results["resolution"]]

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])

    ax.plot(x, y, color=c["series"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(x, y, s=81, color=c["series"], edgecolors=c["surface"], linewidths=2, zorder=4)

    for i in (0, len(x) - 1):   # label the two ends only; the table carries every value
        ax.annotate(f"{y[i]:.2f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 11),
                    ha="center", va="bottom", fontsize=11, color=c["ink"], fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlim(-0.4, len(x) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}\n{BOAT_M / g:g} px boat" for r, g in zip(results["resolution"], gsd)])
    ax.grid(axis="y", color=c["grid"], linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(axis="both", length=0, colors=c["muted"], labelsize=10, pad=6)
    ax.set_xlabel("Camera ground resolution (metres per pixel) and length of a 30 m boat in pixels",
                  color=c["ink2"], fontsize=10, labelpad=10)

    fig.text(0.06, 0.95, "Ship-detection score (mAP50) vs camera resolution", fontsize=14,
             color=c["ink"], fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.885, "YOLO11n trained on 1.5 m Airbus tiles, tested on the same held-out tiles made coarser",
             fontsize=10, color=c["ink2"], ha="left", va="top")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.8, bottom=0.2)
    fig.savefig(out_path, facecolor=c["surface"])
    plt.close(fig)


if __name__ == "__main__":
    results = pd.read_csv(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(results, out_dir / "score_vs_resolution.png", "light")
    draw(results, out_dir / "score_vs_resolution_dark.png", "dark")
    print("Charts written to", out_dir)
