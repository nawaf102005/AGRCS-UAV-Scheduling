"""Small automated checks for the publication plotting style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_style import apply_style, box_axes

def main():
    apply_style()
    fig, ax = plt.subplots()
    box_axes(ax)
    assert all(spine.get_visible() for spine in ax.spines.values())
    assert all(abs(spine.get_linewidth() - 0.8) < 1e-12 for spine in ax.spines.values())
    assert matplotlib.rcParams["font.family"][0] == "serif"
    assert matplotlib.rcParams["axes.labelsize"] == 9.0
    assert matplotlib.rcParams["xtick.labelsize"] == 8.0
    assert matplotlib.rcParams["ytick.labelsize"] == 8.0
    assert matplotlib.rcParams["legend.fontsize"] == 8.0
    plt.close(fig)
    print("Figure-style checks: PASS")

if __name__ == "__main__":
    main()
