"""Central publication plotting style for IEEE-style single/double-column figures."""
from pathlib import Path
import matplotlib.pyplot as plt

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Tinos", "Liberation Serif", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

def apply_style():
    plt.rcParams.update(RC)

def box_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in", width=0.8)

def save_publication_figure(fig, stem, dpi=600):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
