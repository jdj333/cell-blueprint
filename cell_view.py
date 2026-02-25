"""
cell_view.py
------------
View: Contains all code for rendering the cell blueprint and visual output.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from typing import Dict

# Import organelle size data from model
from cell_model import ORGANELLE_SIZES

# (Rendering class and helpers will be moved here from blueprint-output.py)

# Example stub for the view interface:

# Colorblind-friendly palette
PALETTE = {
    "nucleus": "#377eb8",
    "mitochondria": "#e41a1c",
    "lysosome": "#4daf4a",
    "peroxisome": "#984ea3",
    "golgi": "#ff7f00",
    "er": "#a65628",
    "ribosome": "#999999",
    "vesicle": "#f781bf",
    "boundary": "#333333",
}

def render_cell_blueprint(state: Dict[str, float], outpath: str = "cell_blueprint.png", dpi: int = 300, figsize: tuple = (8,8)):
    """
    Render a to-scale, visually enhanced cell blueprint using accurate organelle data.
    """
    import numpy as np
    from matplotlib.patches import Ellipse, Arc
    from cell_model import ORGANELLE_SIZES, ORGANELLE_COUNTS

    # --- Setup
    cell_r = ORGANELLE_SIZES["cell_diameter_um"] / 2.0
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    pad = 2.0
    ax.set_xlim(-cell_r - pad, cell_r + pad)
    ax.set_ylim(-cell_r - pad, cell_r + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis('off')

    # --- Grid
    major = 1.0
    minor = 0.5
    ax.set_xticks(np.arange(-cell_r - pad, cell_r + pad, major))
    ax.set_yticks(np.arange(-cell_r - pad, cell_r + pad, major))
    ax.set_xticks(np.arange(-cell_r - pad, cell_r + pad, minor), minor=True)
    ax.set_yticks(np.arange(-cell_r - pad, cell_r + pad, minor), minor=True)
    ax.grid(which="major", color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.3, alpha=0.6, zorder=0)

    # --- Cell boundary (with drop shadow)
    shadow = Circle((0.3, -0.3), radius=cell_r, fill=True, color="#bbbbbb", alpha=0.18, zorder=1)
    ax.add_patch(shadow)
    cell = Circle((0, 0), radius=cell_r, fill=False, linewidth=2.5, edgecolor=PALETTE["boundary"], zorder=2)
    ax.add_patch(cell)

    # --- Nucleus (double membrane, nucleolus, shadow, envelope)
    nuc_r = ORGANELLE_SIZES["nucleus_diameter_um"] / 2.0
    # Shadow
    ax.add_patch(Circle((0.5, 0.5), nuc_r, color="#bbbbdd", alpha=0.18, zorder=3))
    # Main nucleus
    ax.add_patch(Circle((0, 0), nuc_r, fill=True, color=PALETTE["nucleus"], alpha=0.25, zorder=4))
    ax.add_patch(Circle((0, 0), nuc_r, fill=False, linewidth=2, edgecolor=PALETTE["nucleus"], zorder=5))
    # Nuclear envelope (thicker outline)
    ax.add_patch(Circle((0, 0), nuc_r+0.12, fill=False, linewidth=2.5, edgecolor="#6a8ad7", alpha=0.7, zorder=5))
    ax.text(nuc_r + 0.2, 0.1, "Nucleus", fontsize=10, color=PALETTE["nucleus"], zorder=10)
    ax.text(nuc_r + 0.2, -0.5, "Nuclear envelope", fontsize=8, color="#6a8ad7", zorder=10)
    # Nucleolus (ellipse inside nucleus)
    ax.add_patch(Ellipse((0.3, -0.2), width=0.7, height=0.45, angle=12, edgecolor="#2a2a5a", facecolor="#e6e6fa", alpha=0.85, linewidth=0.7, zorder=6))
    ax.text(0.7, -0.4, "Nucleolus", fontsize=8, color="#2a2a5a", zorder=10)

    # --- Golgi (stacked cisternae)
    golgi_x = nuc_r + 1.2
    golgi_y = -0.5
    n_cisternae = 6
    for i in range(n_cisternae):
        y_offset = golgi_y + i * 0.3
        width = ORGANELLE_SIZES["golgi_region_width_um"] - i * 0.3
        height = 0.22 + 0.04 * i
        x0 = golgi_x + 0.15 * i
        arc = Arc((x0 + width / 2, y_offset), width, height, theta1=0, theta2=180,
                  edgecolor=PALETTE["golgi"], lw=1.2, alpha=0.9, zorder=6)
        ax.add_patch(arc)
    ax.text(golgi_x + 0.1, golgi_y + n_cisternae * 0.3 + 0.2, "Golgi", fontsize=9, color=PALETTE["golgi"], zorder=10)

    # --- ER (network, rough and smooth)
    er_density = state.get("ER_capacity", 0.5)
    segments = int(60 + 120 * er_density)
    for _ in range(segments):
        if np.random.rand() < 0.5:
            r1 = np.random.uniform(nuc_r + 0.4, nuc_r + 1.2)
            r2 = np.random.uniform(nuc_r + 0.4, cell_r - 0.7)
        else:
            r1 = np.random.uniform(nuc_r + 0.4, cell_r - 0.7)
            r2 = np.random.uniform(nuc_r + 0.4, cell_r - 0.7)
        a1 = np.random.uniform(0, 2 * np.pi)
        a2 = a1 + np.random.uniform(-0.6, 0.6)
        x1, y1 = r1 * np.cos(a1), r1 * np.sin(a1)
        x2, y2 = r2 * np.cos(a2), r2 * np.sin(a2)
        ax.plot([x1, x2], [y1, y2], color=PALETTE["er"], linewidth=0.5, alpha=0.5, zorder=4)
        # Rough ER: ribosome dots
        if np.random.rand() < 0.18:
            n_ribo = np.random.randint(2, 6)
            for i in range(n_ribo):
                frac = np.random.uniform(0.1, 0.9)
                rx = x1 + frac * (x2 - x1)
                ry = y1 + frac * (y2 - y1)
                ax.scatter([rx], [ry], s=3, c=PALETTE["ribosome"], alpha=0.7, linewidths=0, zorder=5)
    ax.text(-cell_r + 0.6, cell_r - 1.1, "ER", fontsize=8, color=PALETTE["er"], zorder=10)

    # --- Mitochondria (rods, ovals, branched, cristae, transparency)
    mito_count = ORGANELLE_COUNTS["mitochondria"]
    L = ORGANELLE_SIZES["mito_length_um"]
    mito_types = ["rod", "oval", "branched"]
    cristae_drawn = False
    for i in range(mito_count):
        if i < mito_count // 3:
            r = np.random.uniform(nuc_r + 0.5, nuc_r + 2.0)
        else:
            r = np.random.uniform(nuc_r + 2.0, cell_r - 0.8)
        t = 2 * np.pi * np.random.rand()
        x, y = r * np.cos(t), r * np.sin(t)
        mtype = np.random.choice(mito_types)
        angle = np.random.uniform(0, np.pi)
        if mtype == "rod":
            dx = (L / 2) * np.cos(angle)
            dy = (L / 2) * np.sin(angle)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=PALETTE["mitochondria"], linewidth=1.1, alpha=0.45, zorder=7)
            # Draw cristae as arcs inside a few rods
            if not cristae_drawn and np.random.rand() < 0.02:
                for j in range(3):
                    frac = np.random.uniform(0.2, 0.8)
                    cx = x - dx + frac * 2 * dx
                    cy = y - dy + frac * 2 * dy
                    ax.add_patch(Arc((cx, cy), 0.3, 0.12, angle=np.degrees(angle), theta1=0, theta2=180, color="#b22222", lw=1, alpha=0.7, zorder=8))
                ax.text(x + 0.2, y, "Cristae", fontsize=7, color="#b22222", zorder=10)
                cristae_drawn = True
        elif mtype == "oval":
            width = L * np.random.uniform(0.5, 1.0)
            height = ORGANELLE_SIZES["mito_width_um"] * np.random.uniform(0.7, 1.2)
            e = Ellipse((x, y), width, height, angle=np.degrees(angle), edgecolor=PALETTE["mitochondria"], facecolor='none', linewidth=1.1, alpha=0.45, zorder=7)
            ax.add_patch(e)
        elif mtype == "branched":
            dx = (L / 2) * np.cos(angle)
            dy = (L / 2) * np.sin(angle)
            ax.plot([x, x + dx], [y, y + dy], color=PALETTE["mitochondria"], linewidth=0.9, alpha=0.45, zorder=7)
            ax.plot([x, x - dx], [y, y - dy], color=PALETTE["mitochondria"], linewidth=0.9, alpha=0.45, zorder=7)
            angle2 = angle + np.pi / 3
            dx2 = (L / 2) * np.cos(angle2)
            dy2 = (L / 2) * np.sin(angle2)
            ax.plot([x, x + dx2], [y, y + dy2], color=PALETTE["mitochondria"], linewidth=0.9, alpha=0.45, zorder=7)
    ax.text(-cell_r + 0.6, -0.2, "Mitochondria", fontsize=8, color=PALETTE["mitochondria"], zorder=10)

    # --- Lysosomes & Peroxisomes (different color, peripheral)
    for label, count, color, diam, ylab in [
        ("Lysosome", ORGANELLE_COUNTS["lysosomes"], PALETTE["lysosome"], ORGANELLE_SIZES["lysosome_diameter_um"], "Lysosomes"),
        ("Peroxisome", ORGANELLE_COUNTS["peroxisomes"], PALETTE["peroxisome"], ORGANELLE_SIZES["peroxisome_diameter_um"], "Peroxisomes")]:
        r = diam / 2.0
        placed_label = False
        for _ in range(count):
            if label == "Lysosome":
                rpos = np.random.uniform(0.6 * nuc_r + 0.4 * cell_r, cell_r - 0.8)
            else:
                rpos = np.random.uniform(nuc_r + 0.2, cell_r - 0.8)
            t = 2 * np.pi * np.random.rand()
            x, y = rpos * np.cos(t), rpos * np.sin(t)
            c = Circle((x, y), r, fill=True, facecolor=color, edgecolor=color, alpha=0.32, linewidth=0.7, zorder=8)
            ax.add_patch(c)
            if not placed_label and np.random.rand() < 0.01:
                ax.text(x + r + 0.2, y, ylab, fontsize=7, color=color, alpha=0.9, zorder=10)
                placed_label = True

    # --- Ribosomes (dots on ER and cytosol)
    ribo_count = ORGANELLE_COUNTS["ribosomes"] // 10000  # Downsample for speed
    xs, ys = [], []
    for _ in range(ribo_count):
        x, y = np.random.uniform(-cell_r + 1, cell_r - 1), np.random.uniform(-cell_r + 1, cell_r - 1)
        if (x**2 + y**2) > (nuc_r + 0.5)**2 and (x**2 + y**2) < (cell_r - 1.2)**2:
            xs.append(x)
            ys.append(y)
    ax.scatter(xs, ys, s=1, c=PALETTE["ribosome"], alpha=0.25, linewidths=0, zorder=6)
    ax.text(0.2, cell_r - 1.1, "Ribosomes", fontsize=7, color=PALETTE["ribosome"], zorder=10)

    # --- Legend
    legend_x = -cell_r - 1.5
    legend_y = -cell_r - 1.5
    legend_lines = [
        (PALETTE["boundary"], "Cell boundary"),
        (PALETTE["nucleus"], "Nucleus"),
        (PALETTE["mitochondria"], "Mitochondria"),
        (PALETTE["lysosome"], "Lysosome"),
        (PALETTE["peroxisome"], "Peroxisome"),
        (PALETTE["golgi"], "Golgi"),
        (PALETTE["er"], "ER"),
        (PALETTE["ribosome"], "Ribosome"),
    ]
    for i, (color, label) in enumerate(legend_lines):
        ax.plot([legend_x, legend_x + 0.7], [legend_y - i * 0.5, legend_y - i * 0.5], color=color, lw=5, alpha=0.8, zorder=20)
        ax.text(legend_x + 0.8, legend_y - i * 0.5, label, fontsize=6, color="black", va="center", zorder=20)

    # --- Scale bar (5 µm)
    bar_x = cell_r - 6.5
    bar_y = -cell_r - 1.5
    bar_len = 5.0
    ax.plot([bar_x, bar_x + bar_len], [bar_y, bar_y], color="black", lw=2, zorder=20)
    ax.text(bar_x + bar_len / 2, bar_y - 0.3, "5 µm", fontsize=7, color="black", ha="center", zorder=20)

    # --- Title and caption
    ax.set_title("Organelle Control Blueprint (to-scale schematic)", fontsize=14, color="black", pad=16)
    fig.text(0.5, 0.02, "Simulated mammalian cell, literature-based organelle sizes/counts", ha="center", fontsize=10, color="#333333")

    # --- Save
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
