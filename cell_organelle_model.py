"""
Organelle Control Model (OCM)
=============================

This is a systems-level model of how major cellular "master regulators" influence
organelle biogenesis / capacity and downstream outputs (e.g., ATP capacity, proteostasis),
including a simple proxy for lipofuscin formation pressure.

IMPORTANT DISCLAIMER (read this once):
- This is NOT a clinically validated model.
- It is a structured, mechanistic "hypothesis engine" that encodes current high-level
  biological relationships (activation / inhibition, saturation, feedback).
- The math is intentionally simple but explicit: a discrete-time dynamical system with
  saturating (sigmoid-like) interactions and decay.

Use it to:
- Organize understanding
- Compare interventions qualitatively
- Identify leverage points and feedback loops
- Extend the model with better parameters as you learn more
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional
import math


# ---------------------------
# Math helpers (simple + stable)
# ---------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Keep values in a safe range; many biological variables are normalized 0..1 here."""
    return max(lo, min(hi, x))

def logistic(x: float, k: float = 6.0, x0: float = 0.0) -> float:
    """
    Smooth saturating nonlinearity.
    - x: input
    - k: slope (bigger -> more switch-like)
    - x0: midpoint
    Output is 0..1
    """
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def signed_effect(x: float, weight: float) -> float:
    """
    Convert a normalized variable x in [0,1] into a signed influence using a weight.
    Positive weight -> activating effect
    Negative weight -> inhibitory effect
    We map x to centered [-1, +1] then scale by weight.
    """
    centered = (x - 0.5) * 2.0  # 0->-1, 0.5->0, 1->+1
    return weight * centered


# ---------------------------
# Model objects
# ---------------------------

@dataclass
class Edge:
    """
    An edge is a directional influence: source -> target.

    Example (layman): AMPK pushes PGC-1α upward (activation),
    while mTORC1 pushes TFEB downward (inhibition).

    weight:
      + values mean "activates"
      - values mean "inhibits"
      magnitude means strength
    """
    source: str
    target: str
    weight: float

@dataclass
class Node:
    """
    A node is a state variable in the system.

    value:
      normalized 0..1 (think: "low" to "high")

    decay:
      how strongly it relaxes back toward baseline each step.
      This prevents runaway and represents turnover / regulation.

    baseline:
      where it tends to sit if inputs stop changing.
    """
    name: str
    value: float = 0.5
    baseline: float = 0.5
    decay: float = 0.08  # 0.0=sticky, 1.0=snaps to baseline immediately
    # Optional: allow custom update behavior
    update_rule: Optional[Callable[['Model', str], float]] = None


@dataclass
class Model:
    """
    The main model.
    """
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, name: str, value: float = 0.5, baseline: float = 0.5, decay: float = 0.08,
                 update_rule: Optional[Callable[['Model', str], float]] = None) -> None:
        self.nodes[name] = Node(name=name, value=value, baseline=baseline, decay=decay, update_rule=update_rule)

    def add_edge(self, source: str, target: str, weight: float) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Edge references unknown node: {source}->{target}")
        self.edges.append(Edge(source=source, target=target, weight=weight))

    def get(self, name: str) -> float:
        return self.nodes[name].value

    def set(self, name: str, value: float) -> None:
        self.nodes[name].value = clamp(value)

    def step(self, external_inputs: Dict[str, float], dt: float = 1.0) -> None:
        """
        Advance the system by one discrete time step.

        external_inputs:
          dictionary of input node values that you want to clamp/set each step.
          (Think: this is your planned intervention schedule.)

        dt:
          time step size (arbitrary units). Keep dt=1 for simplicity.
        """
        # 1) Apply external inputs (force input nodes)
        for k, v in external_inputs.items():
            if k not in self.nodes:
                raise ValueError(f"Unknown input node: {k}")
            self.nodes[k].value = clamp(v)

        # 2) Compute net influences for each target node
        net_influence: Dict[str, float] = {name: 0.0 for name in self.nodes.keys()}

        for e in self.edges:
            src_val = self.nodes[e.source].value
            net_influence[e.target] += signed_effect(src_val, e.weight)

        # 3) Update nodes (synchronously)
        new_values: Dict[str, float] = {}

        for name, node in self.nodes.items():
            # Inputs are typically controlled externally; we keep them as set.
            if name.startswith("IN_"):
                new_values[name] = node.value
                continue

            # If this node has a custom update rule, use it.
            if node.update_rule is not None:
                proposed = node.update_rule(self, name)
                # still apply decay toward baseline for stability
                proposed = (1 - node.decay) * proposed + node.decay * node.baseline
                new_values[name] = clamp(proposed)
                continue

            # Default: baseline relaxation + influence -> passed through logistic
            # The logistic keeps values bounded and captures saturation.
            influence = net_influence[name]

            # Convert influence to a 0..1 target using logistic around 0
            drive = logistic(influence, k=3.5, x0=0.0)

            # Blend current toward drive, plus decay toward baseline
            # Interpret "drive" as the "desired level" given current regulators.
            blended = (1 - node.decay) * (0.65 * node.value + 0.35 * drive) + node.decay * node.baseline
            new_values[name] = clamp(blended)

        # 4) Commit updates
        for name, v in new_values.items():
            self.nodes[name].value = v


# ---------------------------
# Build the Organelle Control Model (OCM)
# ---------------------------

def build_ocm() -> Model:
    """
    Create a model with:
    - Inputs (IN_*)
    - Master regulators
    - Organelle capacity variables
    - Outputs (including lipofuscin pressure proxy)
    """
    m = Model()

    # ---- Inputs you can manipulate (0..1)
    # Interpret 0 as "none/low", 1 as "high"
    m.add_node("IN_fasting", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_endurance_exercise", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_resistance_training", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_protein_aa", value=0.5, baseline=0.5, decay=0.0)
    m.add_node("IN_caloric_excess", value=0.2, baseline=0.2, decay=0.0)
    m.add_node("IN_inflammation", value=0.2, baseline=0.2, decay=0.0)
    m.add_node("IN_iron_load", value=0.3, baseline=0.3, decay=0.0)
    m.add_node("IN_NAD_availability", value=0.5, baseline=0.5, decay=0.0)
    m.add_node("IN_sleep_quality", value=0.6, baseline=0.6, decay=0.0)

    # ---- Master regulators (normalized 0..1)
    # Baselines are "middle-ish"; decay represents regulatory homeostasis.
    m.add_node("AMPK", baseline=0.5, decay=0.10)
    m.add_node("mTORC1", baseline=0.5, decay=0.10)
    m.add_node("SIRT1", baseline=0.5, decay=0.10)
    m.add_node("SIRT3", baseline=0.5, decay=0.10)
    m.add_node("PGC1a", baseline=0.5, decay=0.10)  # PGC-1α
    m.add_node("TFEB", baseline=0.5, decay=0.10)
    m.add_node("UPR", baseline=0.4, decay=0.10)    # Adaptive UPR level (too high is bad; modeled simply)
    m.add_node("Nrf1", baseline=0.5, decay=0.10)
    m.add_node("NFkB", baseline=0.3, decay=0.10)
    m.add_node("ROS", baseline=0.4, decay=0.08)

    # ---- Organelle capacity / biogenesis states (0..1)
    # These represent "capacity / functional abundance" rather than exact counts.
    m.add_node("Mito_capacity", baseline=0.5, decay=0.06)
    m.add_node("Lysosome_capacity", baseline=0.5, decay=0.06)
    m.add_node("Autophagy_flux", baseline=0.45, decay=0.07)
    m.add_node("Proteasome_capacity", baseline=0.5, decay=0.06)
    m.add_node("ER_capacity", baseline=0.5, decay=0.06)
    m.add_node("Ribosome_capacity", baseline=0.5, decay=0.06)
    m.add_node("Peroxisome_capacity", baseline=0.45, decay=0.06)

    # ---- Outputs / composite metrics (computed via custom update rules)
    def out_ATP(model: Model, _: str) -> float:
        # Layman: ATP capacity goes up when mitochondria are good and ROS is controlled.
        mito = model.get("Mito_capacity")
        ros = model.get("ROS")
        return clamp(0.75 * mito + 0.25 * (1.0 - ros))

    def out_proteostasis(model: Model, _: str) -> float:
        # Layman: "protein cleanliness" depends on both proteasome and lysosome/autophagy.
        prot = model.get("Proteasome_capacity")
        lyso = model.get("Lysosome_capacity")
        auto = model.get("Autophagy_flux")
        return clamp(0.40 * prot + 0.30 * lyso + 0.30 * auto)

    def out_lipofuscin_pressure(model: Model, _: str) -> float:
        """
        Lipofuscin pressure proxy (0..1):
        - increases with ROS and iron load (more oxidative chemistry)
        - increases when lysosome capacity/autophagy are low (less clearance)
        - increases with inflammation (partly through oxidative stress)
        This is NOT lipofuscin amount; it's "formation pressure".
        """
        ros = model.get("ROS")
        iron = model.get("IN_iron_load")
        infl = model.get("NFkB")
        lyso = model.get("Lysosome_capacity")
        auto = model.get("Autophagy_flux")
        clearance = 0.55 * lyso + 0.45 * auto
        pressure = 0.50 * ros + 0.25 * iron + 0.25 * infl
        return clamp(pressure * (1.0 - 0.8 * clearance))

    def out_inflammatory_pressure(model: Model, _: str) -> float:
        # Layman: how "inflamed" the signaling environment is.
        return clamp(0.70 * model.get("NFkB") + 0.30 * model.get("ROS"))

    m.add_node("OUT_ATP_capacity", baseline=0.5, decay=0.20, update_rule=out_ATP)
    m.add_node("OUT_proteostasis", baseline=0.5, decay=0.20, update_rule=out_proteostasis)
    m.add_node("OUT_lipofuscin_pressure", baseline=0.4, decay=0.20, update_rule=out_lipofuscin_pressure)
    m.add_node("OUT_inflammatory_pressure", baseline=0.3, decay=0.20, update_rule=out_inflammatory_pressure)

    # ---------------------------
    # Wiring: Inputs -> Regulators
    # ---------------------------

    # Fasting / endurance exercise push AMPK up
    m.add_edge("IN_fasting", "AMPK", +1.4)
    m.add_edge("IN_endurance_exercise", "AMPK", +1.2)

    # Caloric excess / amino acids push mTORC1 up
    m.add_edge("IN_caloric_excess", "mTORC1", +1.3)
    m.add_edge("IN_protein_aa", "mTORC1", +1.1)

    # Resistance training tends to push growth signaling (simplified)
    m.add_edge("IN_resistance_training", "mTORC1", +0.9)

    # NAD availability supports sirtuin activity
    m.add_edge("IN_NAD_availability", "SIRT1", +1.2)
    m.add_edge("IN_NAD_availability", "SIRT3", +1.2)

    # Inflammation drives NF-kB; sleep helps restrain it (simplified)
    m.add_edge("IN_inflammation", "NFkB", +1.6)
    m.add_edge("IN_sleep_quality", "NFkB", -0.9)

    # Iron load increases ROS pressure; inflammation also increases ROS
    m.add_edge("IN_iron_load", "ROS", +1.1)
    m.add_edge("NFkB", "ROS", +0.8)

    # AMPK generally restrains mTORC1
    m.add_edge("AMPK", "mTORC1", -1.1)

    # ---------------------------
    # Wiring: Regulator -> Regulator
    # ---------------------------

    # AMPK + SIRT1 promote PGC-1α (mitochondrial biogenesis program)
    m.add_edge("AMPK", "PGC1a", +1.1)
    m.add_edge("SIRT1", "PGC1a", +0.9)

    # mTORC1 suppresses TFEB nuclear activity (lysosome/autophagy genes)
    m.add_edge("mTORC1", "TFEB", -1.2)
    # AMPK supports autophagy/TFEB-ish programs (simplified)
    m.add_edge("AMPK", "TFEB", +0.6)

    # SIRT3 reduces ROS by improving mitochondrial antioxidant/efficiency (simplified)
    m.add_edge("SIRT3", "ROS", -0.9)

    # UPR rises with ER load + stress; here we tie it to inflammation and growth signaling
    m.add_edge("NFkB", "UPR", +0.6)
    m.add_edge("mTORC1", "UPR", +0.5)

    # Nrf1 "bounce-back" tends to rise when proteostasis is challenged; we proxy with ROS+UPR
    m.add_edge("ROS", "Nrf1", +0.6)
    m.add_edge("UPR", "Nrf1", +0.4)

    # ---------------------------
    # Wiring: Regulators -> Organelles (biogenesis/capacity)
    # ---------------------------

    # Mitochondria: PGC-1α program increases capacity; AMPK helps quality control.
    m.add_edge("PGC1a", "Mito_capacity", +1.4)
    m.add_edge("AMPK", "Mito_capacity", +0.4)
    # Chronic high ROS degrades mitochondrial function
    m.add_edge("ROS", "Mito_capacity", -1.0)

    # Lysosome capacity: TFEB is the main driver; inflammation/ROS can impair
    m.add_edge("TFEB", "Lysosome_capacity", +1.5)
    m.add_edge("ROS", "Lysosome_capacity", -0.6)
    m.add_edge("NFkB", "Lysosome_capacity", -0.4)

    # Autophagy flux: rises when TFEB is high and mTORC1 is low; AMPK helps
    m.add_edge("TFEB", "Autophagy_flux", +1.1)
    m.add_edge("mTORC1", "Autophagy_flux", -1.2)
    m.add_edge("AMPK", "Autophagy_flux", +0.8)

    # Proteasome capacity: Nrf1 increases; ROS overload can impair proteins faster than clearance
    m.add_edge("Nrf1", "Proteasome_capacity", +1.3)
    m.add_edge("ROS", "Proteasome_capacity", -0.5)

    # ER capacity: increased by growth signals and adaptive UPR (but chronic UPR is complex)
    m.add_edge("mTORC1", "ER_capacity", +0.9)
    m.add_edge("UPR", "ER_capacity", +0.6)

    # Ribosome biogenesis: strongly tied to mTORC1 (growth program)
    m.add_edge("mTORC1", "Ribosome_capacity", +1.4)
    m.add_edge("AMPK", "Ribosome_capacity", -0.6)  # energy stress suppresses growth

    # Peroxisomes: tied to lipid metabolism; we proxy with mTORC1 (growth/metabolism) and ROS (demand)
    m.add_edge("mTORC1", "Peroxisome_capacity", +0.4)
    m.add_edge("ROS", "Peroxisome_capacity", +0.3)  # more ROS -> more need for redox organelles (simplified)
    m.add_edge("NFkB", "Peroxisome_capacity", -0.3)  # chronic inflammation can impair metabolism

    return m


# ---------------------------
# Simulation utilities
# ---------------------------

def run_simulation(model: Model,
                   schedule: List[Dict[str, float]],
                   steps: int,
                   dt: float = 1.0,
                   track: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """
    Run the model forward for 'steps' time steps.
    schedule: list of external_inputs dicts; if shorter than steps, last entry repeats.
    track: list of node names to record; if None, record key outputs + organelles.
    """
    if track is None:
        track = [
            # Master regulators
            "AMPK", "mTORC1", "SIRT1", "SIRT3", "PGC1a", "TFEB", "NFkB", "ROS",
            # Organelles
            "Mito_capacity", "Lysosome_capacity", "Autophagy_flux",
            "Proteasome_capacity", "ER_capacity", "Ribosome_capacity", "Peroxisome_capacity",
            # Outputs
            "OUT_ATP_capacity", "OUT_proteostasis", "OUT_lipofuscin_pressure", "OUT_inflammatory_pressure"
        ]

    history: Dict[str, List[float]] = {k: [] for k in track}

    for t in range(steps):
        inputs = schedule[min(t, len(schedule) - 1)]
        model.step(inputs, dt=dt)
        # Update outputs (their update_rule uses current values)
        # (They are updated in step() already, but tracking is after stepping.)
        for k in track:
            history[k].append(model.get(k))

    return history


def pretty_last(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Return last value of each tracked variable."""
    return {k: v[-1] for k, v in history.items() if v}


# ---------------------------
# Example usage / scenarios
# ---------------------------

def scenario_baseline() -> Dict[str, float]:
    return {
        "IN_fasting": 0.0,
        "IN_endurance_exercise": 0.0,
        "IN_resistance_training": 0.0,
        "IN_protein_aa": 0.5,
        "IN_caloric_excess": 0.2,
        "IN_inflammation": 0.2,
        "IN_iron_load": 0.3,
        "IN_NAD_availability": 0.5,
        "IN_sleep_quality": 0.6,
    }

def scenario_repair_mode() -> Dict[str, float]:
    """
    Layman: "repair mode" = more AMPK/TFEB/autophagy support.
    Example: fasting + endurance + good sleep + lower caloric excess.
    """
    return {
        "IN_fasting": 0.7,
        "IN_endurance_exercise": 0.6,
        "IN_resistance_training": 0.1,
        "IN_protein_aa": 0.45,
        "IN_caloric_excess": 0.05,
        "IN_inflammation": 0.15,
        "IN_iron_load": 0.3,
        "IN_NAD_availability": 0.6,
        "IN_sleep_quality": 0.75,
    }

def scenario_growth_mode() -> Dict[str, float]:
    """
    Layman: "growth mode" = more mTORC1/ribosome/ER support.
    Example: resistance training + protein + caloric surplus.
    """
    return {
        "IN_fasting": 0.0,
        "IN_endurance_exercise": 0.2,
        "IN_resistance_training": 0.8,
        "IN_protein_aa": 0.8,
        "IN_caloric_excess": 0.6,
        "IN_inflammation": 0.2,
        "IN_iron_load": 0.3,
        "IN_NAD_availability": 0.5,
        "IN_sleep_quality": 0.65,
    }


if __name__ == "__main__":
    model = build_ocm()

    steps = 120  # "days" or arbitrary time units

    # Compare three scenarios
    baseline_hist = run_simulation(model=build_ocm(),
                                   schedule=[scenario_baseline()],
                                   steps=steps)

    repair_hist = run_simulation(model=build_ocm(),
                                 schedule=[scenario_repair_mode()],
                                 steps=steps)

    growth_hist = run_simulation(model=build_ocm(),
                                 schedule=[scenario_growth_mode()],
                                 steps=steps)

    print("\n=== LAST VALUES (Baseline) ===")
    for k, v in pretty_last(baseline_hist).items():
        if k.startswith("OUT_"):
            print(f"{k:28s} {v:.3f}")

    print("\n=== LAST VALUES (Repair Mode) ===")
    for k, v in pretty_last(repair_hist).items():
        if k.startswith("OUT_"):
            print(f"{k:28s} {v:.3f}")

    print("\n=== LAST VALUES (Growth Mode) ===")
    for k, v in pretty_last(growth_hist).items():
        if k.startswith("OUT_"):
            print(f"{k:28s} {v:.3f}")

    # If you want: print a few internal states too
    print("\n(Example internal state check: TFEB, Lysosome_capacity, Autophagy_flux)")
    print(f"Baseline: TFEB={baseline_hist['TFEB'][-1]:.3f}, Lys={baseline_hist['Lysosome_capacity'][-1]:.3f}, Auto={baseline_hist['Autophagy_flux'][-1]:.3f}")
    print(f"Repair:   TFEB={repair_hist['TFEB'][-1]:.3f}, Lys={repair_hist['Lysosome_capacity'][-1]:.3f}, Auto={repair_hist['Autophagy_flux'][-1]:.3f}")
    print(f"Growth:   TFEB={growth_hist['TFEB'][-1]:.3f}, Lys={growth_hist['Lysosome_capacity'][-1]:.3f}, Auto={growth_hist['Autophagy_flux'][-1]:.3f}")