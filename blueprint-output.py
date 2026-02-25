"""
blueprint-output.py
-------------------
Renders a to-scale (micrometer) 2D "cell blueprint" from the Organelle Control Model.

Design goals:
- White background + light-gray grid (µm scale)
- Blue for cellular structures (lines/outlines)
- Orange for input arrows
- Purple for processes (flow arrows/links)
- Red for outputs (readouts)

Scale anchors (typical values):
- Typical animal cell diameter: ~10–20 µm  (Alberts NCBI; Li 2015)  :contentReference[oaicite:7]{index=7}
- Nucleus diameter: ~5–6 µm (NIGMS; nucleus wiki)                  :contentReference[oaicite:8]{index=8}
- Mitochondrion length: ~1–2 µm (NIGMS; EBSCO summary)             :contentReference[oaicite:9]{index=9}
- Lysosome diameter: ~0.1–1.2 µm (lysosome wiki; NIGMS)            :contentReference[oaicite:10]{index=10}
- Peroxisome diameter: ~0.1–1 µm (Smith 2013 PMC; SciDirect topic) :contentReference[oaicite:11]{index=11}
- Golgi cisternae diameter: ~0.7–1.1 µm (Klumperman 2011 PMC)      :contentReference[oaicite:12]{index=12}
- ER tubule diameter ~60–100 nm (BioNumbers; Terasaki 2018)        :contentReference[oaicite:13]{index=13}

Note:
ER/ribosomes are below light-microscope scale. To stay truthful to scale,
we render ER as thin network lines (not as thick tubes) and ribosomes as tiny dots.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch


# -----------------------------
# Optional: Load your model module
# -----------------------------

from cell_model import build_ocm
from cell_controller import run_simulation, scenario_baseline
from cell_view import render_cell_blueprint


# -----------------------------
# Drawing configuration
# -----------------------------

@dataclass
class Style:
    bg: str = "white"
    grid: str = "#e6e6e6"     # light gray
    structure: str = "#1f4fbf"  # blue-ish
    inputs: str = "#ff8c00"     # orange
    outputs: str = "#cc0000"    # red
    process: str = "#6a0dad"    # purple

    label_size: int = 7
    tiny_label_size: int = 6


# -----------------------------
# Scale model (µm)
# -----------------------------

@dataclass
class CellScale:
    # Choose a "typical animal cell" diameter; we default to 20 µm
    # (Within the common 10–20 µm range.) :contentReference[oaicite:14]{index=14}
    cell_diameter_um: float = 20.0

    # nucleus ~5–6 µm diameter :contentReference[oaicite:15]{index=15}
    nucleus_diameter_um: float = 5.5

    # mitochondria ~1–2 µm long :contentReference[oaicite:16]{index=16}
    mito_length_um: float = 1.6
    mito_width_um: float = 0.6  # cross-section commonly sub-micron

    # lysosome 0.1–1.2 µm :contentReference[oaicite:17]{index=17}
    lysosome_diameter_um: float = 0.5

    # peroxisome 0.1–1 µm :contentReference[oaicite:18]{index=18}
    peroxisome_diameter_um: float = 0.45

    # Golgi: cisternae ~0.7–1.1 µm but Golgi region is larger; we draw a 3 µm wide region
    # :contentReference[oaicite:19]{index=19}
    golgi_region_width_um: float = 3.0
    golgi_region_height_um: float = 2.0


# -----------------------------
# Utilities
# -----------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def rand_point_in_circle(r: float) -> Tuple[float, float]:
    # uniform distribution in circle
    t = 2 * math.pi * random.random()
    u = random.random() + random.random()
    rr = 2 - u if u > 1 else u
    return (rr * r * math.cos(t), rr * r * math.sin(t))

def inside_cell(x: float, y: float, cell_r: float, margin: float = 0.0) -> bool:
    return (x * x + y * y) <= (cell_r - margin) ** 2


# -----------------------------
# Blueprint renderer
# -----------------------------

class CellBlueprintRenderer:
    def __init__(self, style: Style, scale: CellScale):
        self.style = style
        self.scale = scale

    def render(
        self,
        state: Dict[str, float],
        outpath: str = "cell_blueprint.png",
        seed: int = 7
    ) -> None:
        """
        state: dict of model node values (0..1). Key nodes used:
          Inputs: IN_* (optional)
          Organelles: Mito_capacity, Lysosome_capacity, Autophagy_flux, Proteasome_capacity, ER_capacity, Ribosome_capacity, Peroxisome_capacity
          Outputs: OUT_* (optional)

        outpath: where to save PNG
        """
        random.seed(seed)

        # --- Pull state values with safe defaults
        mito = clamp(state.get("Mito_capacity", 0.5))
        lyso = clamp(state.get("Lysosome_capacity", 0.5))
        auto = clamp(state.get("Autophagy_flux", 0.45))
        prot = clamp(state.get("Proteasome_capacity", 0.5))
        er   = clamp(state.get("ER_capacity", 0.5))
        ribo = clamp(state.get("Ribosome_capacity", 0.5))
        per  = clamp(state.get("Peroxisome_capacity", 0.45))

        out_lipo = clamp(state.get("OUT_lipofuscin_pressure", 0.4))
        out_atp  = clamp(state.get("OUT_ATP_capacity", 0.5))
        out_prot = clamp(state.get("OUT_proteostasis", 0.5))
        out_infl = clamp(state.get("OUT_inflammatory_pressure", 0.3))

        # --- Create figure in µm coordinates
        cell_r = self.scale.cell_diameter_um / 2.0
        fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
        fig.patch.set_facecolor(self.style.bg)
        ax.set_facecolor(self.style.bg)

        # Axes limits: a little padding around the cell
        pad = 2.0
        ax.set_xlim(-cell_r - pad, cell_r + pad)
        ax.set_ylim(-cell_r - pad, cell_r + pad)
        ax.set_aspect("equal", adjustable="box")

        # --- Grid: 1 µm major grid, 0.5 µm minor grid (light)
        major = 1.0
        minor = 0.5
        ax.set_xticks([x for x in self._frange(-cell_r - pad, cell_r + pad, major)])
        ax.set_yticks([y for y in self._frange(-cell_r - pad, cell_r + pad, major)])
        ax.set_xticks([x for x in self._frange(-cell_r - pad, cell_r + pad, minor)], minor=True)
        ax.set_yticks([y for y in self._frange(-cell_r - pad, cell_r + pad, minor)], minor=True)
        ax.grid(which="major", color=self.style.grid, linewidth=0.6)
        ax.grid(which="minor", color=self.style.grid, linewidth=0.3, alpha=0.6)

        # Hide tick labels; this is a diagram, not a plot
        ax.tick_params(labelbottom=False, labelleft=False, length=0)

        # --- Draw cell boundary (blue)
        cell = Circle((0, 0), radius=cell_r, fill=False, linewidth=2.0, edgecolor=self.style.structure)
        ax.add_patch(cell)
        ax.text(-cell_r, cell_r + 0.8, f"Cell (~{self.scale.cell_diameter_um:.0f} µm diameter)",
                fontsize=self.style.label_size, color=self.style.structure)

        # --- Draw nucleus (double membrane + nucleolus)
        nuc_r = self.scale.nucleus_diameter_um / 2.0
        # Outer membrane
        nucleus_outer = Circle((0, 0), radius=nuc_r, fill=False, linewidth=1.6, edgecolor=self.style.structure)
        ax.add_patch(nucleus_outer)
        # Inner membrane
        nucleus_inner = Circle((0, 0), radius=nuc_r - 0.18, fill=False, linewidth=0.9, edgecolor="#3a6cc7", alpha=0.7)
        ax.add_patch(nucleus_inner)
        # Nucleolus
        from matplotlib.patches import Ellipse
        nucleolus = Ellipse((0.7, 0.5), width=1.1, height=0.7, angle=18, edgecolor="#2a2a5a", facecolor="#bfc6e0", alpha=0.7, linewidth=0.7)
        ax.add_patch(nucleolus)
        ax.text(nuc_r + 0.2, 0.1, "Nucleus", fontsize=self.style.label_size, color=self.style.structure)

        # --- Golgi region (stacked cisternae near nucleus)
        golgi_x = nuc_r + 1.2
        golgi_y = -0.5
        n_cisternae = 5
        for i in range(n_cisternae):
            y_offset = golgi_y + i * 0.3
            width = self.scale.golgi_region_width_um - i * 0.3
            height = 0.22 + 0.04 * i
            x0 = golgi_x + 0.15 * i
            # Draw as a curved arc (half ellipse)
            from matplotlib.patches import Arc
            arc = Arc((x0 + width / 2, y_offset), width, height, theta1=0, theta2=180,
                  edgecolor=self.style.structure, lw=1.1, alpha=0.9)
            ax.add_patch(arc)
        ax.text(golgi_x + 0.1, golgi_y + n_cisternae * 0.3 + 0.2,
            "Golgi (cisternae)", fontsize=self.style.label_size, color=self.style.structure)

        # --- ER network (thin blue lines) around nucleus (not to tube diameter; tubes are ~60–100 nm)
        # We represent topology instead of thickness.
        self._draw_er_network(ax, center=(0, 0), inner_r=nuc_r + 0.4, outer_r=cell_r - 0.7, density=er)

        # --- Mitochondria (blue "capsules") count scales with mito capacity
        mito_count = int(6 + 18 * mito)  # rough visual scaling
        self._draw_mitochondria(ax, cell_r, nuc_r, count=mito_count, thickness=0.8 + 1.6 * mito)

        # --- Lysosomes (blue small circles) count scales with lysosome capacity
        lys_count = int(8 + 28 * lyso)
        self._draw_vesicles(ax, cell_r, nuc_r, count=lys_count,
                            diameter=self.scale.lysosome_diameter_um,
                            label="Lysosomes", label_once=True)

        # --- Peroxisomes (blue small circles) count scales with peroxisome capacity
        per_count = int(4 + 16 * per)
        self._draw_vesicles(ax, cell_r, nuc_r, count=per_count,
                            diameter=self.scale.peroxisome_diameter_um,
                            label="Peroxisomes", label_once=True)

        # --- Ribosomes (tiny blue dots) density scales with ribosome capacity
        ribo_count = int(40 + 160 * ribo)
        self._draw_ribosomes(ax, cell_r, nuc_r, count=ribo_count)

        # --- Process flows (purple)
        # Autophagy: draw arrows from cytosol -> lysosome region to communicate flux
        self._draw_process_arrows(ax, cell_r, strength=auto, label="Autophagy flux", y=-cell_r + 2.0)

        # Proteostasis: arrow between proteasome capacity + lysosome (conceptual)
        self._draw_process_arrows(ax, cell_r, strength=prot, label="Proteostasis", y=-cell_r + 1.1)

        # --- Inputs (orange) as arrows entering cell boundary
        self._draw_inputs(ax, cell_r, state)

        # --- Outputs (red) as a small readout box
        self._draw_outputs_box(ax, cell_r, out_atp, out_prot, out_lipo, out_infl)


        ax.set_title("Organelle Control Blueprint (to-scale schematic)", fontsize=10, color="black", pad=10)

        # --- Legend
        legend_x = -cell_r - 1.5
        legend_y = -cell_r - 1.5
        legend_lines = [
            (self.style.structure, "Cell boundary, organelles"),
            ("#2255a5", "Lysosome"),
            ("#5fa8d3", "Peroxisome"),
            ("#222244", "Ribosome (free/ER)"),
            (self.style.inputs, "Inputs"),
            (self.style.process, "Process arrows"),
            (self.style.outputs, "Outputs box")
        ]
        for i, (color, label) in enumerate(legend_lines):
            ax.plot([legend_x, legend_x + 0.7], [legend_y - i * 0.5, legend_y - i * 0.5], color=color, lw=3)
            ax.text(legend_x + 0.8, legend_y - i * 0.5, label, fontsize=6, color="black", va="center")

        # --- Scale bar (5 µm)
        bar_x = cell_r - 6.5
        bar_y = -cell_r - 1.5
        bar_len = 5.0
        ax.plot([bar_x, bar_x + bar_len], [bar_y, bar_y], color="black", lw=2)
        ax.text(bar_x + bar_len / 2, bar_y - 0.3, "5 µm", fontsize=7, color="black", ha="center")

        # Save
        fig.savefig(outpath, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved blueprint to: {os.path.abspath(outpath)}")

    # -----------------------------
    # Internal drawing helpers
    # -----------------------------

    def _frange(self, start: float, stop: float, step: float):
        x = start
        while x <= stop + 1e-9:
            yield x
            x += step

    def _draw_er_network(self, ax, center: Tuple[float, float], inner_r: float, outer_r: float, density: float):
        """
        ER is nanometer-scale tubules; densify near nucleus, add rough ER with ribosome dots.
        """
        segments = int(30 + 90 * density)
        cx, cy = center
        # Draw denser network near nucleus
        for _ in range(segments):
            if random.random() < 0.5:
                # Near nucleus
                r1 = random.uniform(inner_r, inner_r + 1.2)
                r2 = random.uniform(inner_r, inner_r + 1.2)
            else:
                r1 = random.uniform(inner_r, outer_r)
                r2 = random.uniform(inner_r, outer_r)
            a1 = random.uniform(0, 2 * math.pi)
            a2 = a1 + random.uniform(-0.6, 0.6)
            x1, y1 = cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)
            x2, y2 = cx + r2 * math.cos(a2), cy + r2 * math.sin(a2)
            ax.plot([x1, x2], [y1, y2], color=self.style.structure, linewidth=0.35, alpha=0.65)
            # Add rough ER: ribosome dots on some lines
            if random.random() < 0.18:
                n_ribo = random.randint(2, 6)
                for i in range(n_ribo):
                    frac = random.uniform(0.1, 0.9)
                    rx = x1 + frac * (x2 - x1)
                    ry = y1 + frac * (y2 - y1)
                    ax.scatter([rx], [ry], s=2, c="#222244", alpha=0.7, linewidths=0)
        ax.text(-outer_r + 0.6, outer_r - 0.6, "ER (network)", fontsize=self.style.label_size,
                color=self.style.structure, alpha=0.85)

    def _draw_mitochondria(self, ax, cell_r: float, nuc_r: float, count: int, thickness: float):
        """
        Draw mitochondria as rods, ovals, and branched forms, clustered near nucleus and scattered in cytosol.
        """
        L = self.scale.mito_length_um
        mito_types = ["rod", "oval", "branched"]
        for i in range(count):
            # Cluster 1/3 near nucleus, rest in cytosol
            if i < count // 3:
                # Near nucleus, but outside it
                r = random.uniform(nuc_r + 0.5, nuc_r + 2.0)
            else:
                r = random.uniform(nuc_r + 2.0, cell_r - 0.8)
            t = 2 * math.pi * random.random()
            x, y = r * math.cos(t), r * math.sin(t)
            if not inside_cell(x, y, cell_r, margin=0.8):
                continue
            mtype = random.choice(mito_types)
            angle = random.uniform(0, math.pi)
            if mtype == "rod":
                dx = (L / 2) * math.cos(angle)
                dy = (L / 2) * math.sin(angle)
                ax.plot([x - dx, x + dx], [y - dy, y + dy],
                        color=self.style.structure, linewidth=0.6 * (thickness / 2.0),
                        solid_capstyle="round", alpha=0.9)
            elif mtype == "oval":
                from matplotlib.patches import Ellipse
                width = L * random.uniform(0.5, 1.0)
                height = self.scale.mito_width_um * random.uniform(0.7, 1.2)
                e = Ellipse((x, y), width, height, angle=math.degrees(angle),
                            edgecolor=self.style.structure, facecolor='none',
                            linewidth=0.6 * (thickness / 2.0), alpha=0.9)
                ax.add_patch(e)
            elif mtype == "branched":
                # Y-shaped: center + two arms
                dx = (L / 2) * math.cos(angle)
                dy = (L / 2) * math.sin(angle)
                ax.plot([x, x + dx], [y, y + dy], color=self.style.structure, linewidth=0.5 * (thickness / 2.0), alpha=0.9)
                ax.plot([x, x - dx], [y, y - dy], color=self.style.structure, linewidth=0.5 * (thickness / 2.0), alpha=0.9)
                # Third arm
                angle2 = angle + math.pi / 3
                dx2 = (L / 2) * math.cos(angle2)
                dy2 = (L / 2) * math.sin(angle2)
                ax.plot([x, x + dx2], [y, y + dy2], color=self.style.structure, linewidth=0.5 * (thickness / 2.0), alpha=0.9)
        ax.text(-cell_r + 0.6, -0.2, "Mitochondria", fontsize=self.style.label_size,
                color=self.style.structure, alpha=0.9)

    def _draw_vesicles(self, ax, cell_r: float, nuc_r: float, count: int, diameter: float,
                       label: str, label_once: bool):
        r = diameter / 2.0
        placed_label = False
        # Color by vesicle type
        color = self.style.structure
        if "Lysosome" in label:
            color = "#2255a5"  # darker blue
        elif "Peroxisome" in label:
            color = "#5fa8d3"  # lighter blue
        for _ in range(count):
            # Lysosomes: more peripheral, Peroxisomes: random
            if "Lysosome" in label:
                # Place in outer 40% of cytosol
                rpos = random.uniform(0.6 * nuc_r + 0.4 * cell_r, cell_r - 0.8)
            else:
                rpos = random.uniform(nuc_r + 0.2, cell_r - 0.8)
            t = 2 * math.pi * random.random()
            x, y = rpos * math.cos(t), rpos * math.sin(t)
            if not inside_cell(x, y, cell_r, margin=0.8) or inside_cell(x, y, nuc_r, margin=0.2):
                continue
            c = Circle((x, y), r, fill=False, linewidth=0.8, edgecolor=color, alpha=0.9)
            ax.add_patch(c)
            if label_once and not placed_label:
                ax.text(x + r + 0.2, y, label, fontsize=self.style.label_size,
                        color=color, alpha=0.9)
                placed_label = True

    def _draw_ribosomes(self, ax, cell_r: float, nuc_r: float, count: int):
        """
        Ribosomes: dense on rough ER, sparser in cytosol.
        """
        xs, ys = [], []
        # Cytosolic ribosomes (30% of count)
        for _ in range(int(count * 0.3)):
            x, y = rand_point_in_circle(cell_r - 0.8)
            if inside_cell(x, y, nuc_r, margin=0.1):
                continue
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=1, c="#222244", alpha=0.35, linewidths=0)
        # Rough ER ribosomes are now drawn in _draw_er_network
        ax.text(0.2, cell_r - 1.1, "Ribosomes (dots)", fontsize=self.style.tiny_label_size,
                color=self.style.structure, alpha=0.8)

    def _draw_process_arrows(self, ax, cell_r: float, strength: float, label: str, y: float):
        """
        Draws purple arrows along the bottom as "process capacity indicators".
        Stronger process -> thicker and more arrows.
        """
        n = int(1 + 4 * strength)
        x0 = -cell_r + 1.0
        for i in range(n):
            arrow = FancyArrowPatch((x0 + i * 2.0, y), (x0 + i * 2.0 + 1.6, y),
                                    arrowstyle="->", mutation_scale=10,
                                    linewidth=0.8 + 1.6 * strength,
                                    color=self.style.process, alpha=0.85)
            ax.add_patch(arrow)
        ax.text(x0, y + 0.35, label, fontsize=self.style.label_size, color=self.style.process)

    def _draw_inputs(self, ax, cell_r: float, state: Dict[str, float]):
        """
        Inputs enter from the left/top edges as orange arrows.
        We look for IN_* keys.
        """
        # choose a consistent set of input keys (if present)
        keys = [
            "IN_fasting", "IN_endurance_exercise", "IN_resistance_training",
            "IN_protein_aa", "IN_caloric_excess", "IN_inflammation",
            "IN_iron_load", "IN_NAD_availability", "IN_sleep_quality"
        ]
        present = [(k, clamp(state.get(k, 0.0))) for k in keys]

        # place arrows on left side; thickness indicates magnitude
        base_x = -cell_r - 1.2
        for idx, (k, v) in enumerate(present):
            if v <= 0.02:
                continue
            y = cell_r - 1.5 - idx * 1.1
            arr = FancyArrowPatch((base_x, y), (-cell_r + 0.2, y),
                                  arrowstyle="->", mutation_scale=10,
                                  linewidth=0.8 + 2.2 * v,
                                  color=self.style.inputs, alpha=0.9)
            ax.add_patch(arr)
            ax.text(base_x, y + 0.25, f"{k.replace('IN_', '')}: {v:.2f}",
                    fontsize=self.style.tiny_label_size, color=self.style.inputs)

    def _draw_outputs_box(self, ax, cell_r: float, out_atp: float, out_prot: float, out_lipo: float, out_infl: float):
        """
        Outputs displayed as a red readout box on right side.
        """
        box_w, box_h = 6.0, 3.4
        x0 = cell_r - box_w - 0.6
        y0 = cell_r - box_h - 0.6
        rect = Rectangle((x0, y0), box_w, box_h, fill=False, linewidth=1.2, edgecolor=self.style.outputs)
        ax.add_patch(rect)
        ax.text(x0 + 0.2, y0 + box_h - 0.5, "Outputs", fontsize=self.style.label_size, color=self.style.outputs)

        lines = [
            f"ATP capacity: {out_atp:.2f}",
            f"Proteostasis: {out_prot:.2f}",
            f"Lipofuscin pressure: {out_lipo:.2f}",
            f"Inflamm pressure: {out_infl:.2f}",
        ]
        for i, line in enumerate(lines):
            ax.text(x0 + 0.2, y0 + box_h - 1.1 - i * 0.65,
                    line, fontsize=self.style.tiny_label_size, color=self.style.outputs)


# -----------------------------
# Main entry
# -----------------------------


def main():
    # Build model and run scenario
    model = build_ocm()
    schedule = scenario_baseline()
    history = run_simulation(model=model, schedule=schedule, steps=120)
    state = {k: v[-1] for k, v in history.items() if len(v) > 0}
    # Update state with the first input dict from the schedule
    state.update(scenario_baseline()[0])
    render_cell_blueprint(state=state, outpath="cell_blueprint.png")


if __name__ == "__main__":
    main()