"""Generate a deterministic synthetic figure to validate the publication plotting pipeline."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style, box_axes, save_publication_figure

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_outputs"

def main():
    rng = np.random.default_rng(20260929)
    x = np.arange(1, 11)
    y = 50 + 0.35*x + rng.normal(0, 0.12, x.size)
    apply_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    ax.plot(x, y, marker="o", label="Synthetic validation")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("Goodput (Mbit/s)")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.18)
    box_axes(ax)
    fig.tight_layout()
    save_publication_figure(fig, OUT / "example_twc_figure")
    plt.close(fig)
    np.savetxt(OUT / "synthetic_validation.csv", np.column_stack([x, y]), delimiter=",",
               header="evaluation_index,goodput_mbps", comments="")
    print("Created example_outputs/example_twc_figure.pdf")
    print("Created example_outputs/example_twc_figure.png (600 dpi)")
    print("Created example_outputs/synthetic_validation.csv")

if __name__ == "__main__":
    main()
