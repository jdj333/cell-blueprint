# Clamp helper function
def clamp(x: float, minval: float = 0.0, maxval: float = 1.0) -> float:
    return max(minval, min(maxval, x))

def build_ocm():

    m = Model()
    # Add all input nodes
    m.add_node("IN_fasting", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_endurance_exercise", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_resistance_training", value=0.0, baseline=0.0, decay=0.0)
    m.add_node("IN_protein_aa", value=0.5, baseline=0.5, decay=0.0)
    m.add_node("IN_caloric_excess", value=0.2, baseline=0.2, decay=0.0)
    m.add_node("IN_inflammation", value=0.2, baseline=0.2, decay=0.0)
    m.add_node("IN_iron_load", value=0.3, baseline=0.3, decay=0.0)
    m.add_node("IN_NAD_availability", value=0.5, baseline=0.5, decay=0.0)
    m.add_node("IN_sleep_quality", value=0.6, baseline=0.6, decay=0.0)

    # Add all master regulator nodes
    m.add_node("AMPK", baseline=0.5, decay=0.10)
    m.add_node("mTORC1", baseline=0.5, decay=0.10)
    m.add_node("SIRT1", baseline=0.5, decay=0.10)
    m.add_node("SIRT3", baseline=0.5, decay=0.10)
    m.add_node("PGC1a", baseline=0.5, decay=0.10)
    m.add_node("TFEB", baseline=0.5, decay=0.10)
    m.add_node("UPR", baseline=0.4, decay=0.10)
    m.add_node("Nrf1", baseline=0.5, decay=0.10)
    m.add_node("NFkB", baseline=0.3, decay=0.10)
    m.add_node("ROS", baseline=0.4, decay=0.08)

    # Add all organelle capacity nodes
    m.add_node("Mito_capacity", baseline=0.5, decay=0.06)
    m.add_node("Lysosome_capacity", baseline=0.5, decay=0.06)
    m.add_node("Autophagy_flux", baseline=0.45, decay=0.07)
    m.add_node("Proteasome_capacity", baseline=0.5, decay=0.06)
    m.add_node("ER_capacity", baseline=0.5, decay=0.06)
    m.add_node("Ribosome_capacity", baseline=0.5, decay=0.06)
    m.add_node("Peroxisome_capacity", baseline=0.45, decay=0.06)

    # ---- Outputs / composite metrics (computed via custom update rules)
    def out_ATP(model: Model, _: str) -> float:
        mito = model.get("Mito_capacity")
        ros = model.get("ROS")
        return clamp(0.75 * mito + 0.25 * (1.0 - ros))

    def out_proteostasis(model: Model, _: str) -> float:
        prot = model.get("Proteasome_capacity")
        lyso = model.get("Lysosome_capacity")
        auto = model.get("Autophagy_flux")
        return clamp(0.40 * prot + 0.30 * lyso + 0.30 * auto)

    def out_lipofuscin_pressure(model: Model, _: str) -> float:
        ros = model.get("ROS")
        iron = model.get("IN_iron_load")
        infl = model.get("NFkB")
        lyso = model.get("Lysosome_capacity")
        auto = model.get("Autophagy_flux")
        clearance = 0.55 * lyso + 0.45 * auto
        pressure = 0.50 * ros + 0.25 * iron + 0.25 * infl
        return clamp(pressure * (1.0 - 0.8 * clearance))

    def out_inflammatory_pressure(model: Model, _: str) -> float:
        return clamp(0.70 * model.get("NFkB") + 0.30 * model.get("ROS"))

    m.add_node("OUT_ATP_capacity", baseline=0.5, decay=0.20, update_rule=out_ATP)
    m.add_node("OUT_proteostasis", baseline=0.5, decay=0.20, update_rule=out_proteostasis)
    m.add_node("OUT_lipofuscin_pressure", baseline=0.4, decay=0.20, update_rule=out_lipofuscin_pressure)
    m.add_node("OUT_inflammatory_pressure", baseline=0.3, decay=0.20, update_rule=out_inflammatory_pressure)

    # Wiring: Inputs -> Regulators
    m.add_edge("IN_fasting", "AMPK", +1.4)
    m.add_edge("IN_endurance_exercise", "AMPK", +1.2)
    m.add_edge("IN_caloric_excess", "mTORC1", +1.3)
    m.add_edge("IN_protein_aa", "mTORC1", +1.1)
    m.add_edge("IN_resistance_training", "mTORC1", +0.9)
    m.add_edge("IN_NAD_availability", "SIRT1", +1.2)
    m.add_edge("IN_NAD_availability", "SIRT3", +1.2)
    m.add_edge("IN_inflammation", "NFkB", +1.6)
    m.add_edge("IN_sleep_quality", "NFkB", -0.9)
    m.add_edge("IN_iron_load", "ROS", +1.1)
    m.add_edge("NFkB", "ROS", +0.8)
    m.add_edge("AMPK", "mTORC1", -1.1)

    # Regulator -> Regulator
    m.add_edge("AMPK", "PGC1a", +1.1)
    m.add_edge("SIRT1", "PGC1a", +0.9)
    m.add_edge("mTORC1", "TFEB", -1.2)
    m.add_edge("AMPK", "TFEB", +0.6)
    m.add_edge("SIRT3", "ROS", -0.9)
    m.add_edge("NFkB", "UPR", +0.6)
    m.add_edge("mTORC1", "UPR", +0.5)
    m.add_edge("ROS", "Nrf1", +0.6)
    m.add_edge("UPR", "Nrf1", +0.4)

    # Regulators -> Organelles
    m.add_edge("PGC1a", "Mito_capacity", +1.4)
    m.add_edge("AMPK", "Mito_capacity", +0.4)
    m.add_edge("ROS", "Mito_capacity", -1.0)
    m.add_edge("TFEB", "Lysosome_capacity", +1.5)
    m.add_edge("ROS", "Lysosome_capacity", -0.6)
    m.add_edge("NFkB", "Lysosome_capacity", -0.4)
    m.add_edge("TFEB", "Autophagy_flux", +1.1)
    m.add_edge("mTORC1", "Autophagy_flux", -1.2)
    m.add_edge("AMPK", "Autophagy_flux", +0.8)
    m.add_edge("Nrf1", "Proteasome_capacity", +1.3)
    m.add_edge("ROS", "Proteasome_capacity", -0.5)
    m.add_edge("mTORC1", "ER_capacity", +0.9)
    m.add_edge("UPR", "ER_capacity", +0.6)
    m.add_edge("mTORC1", "Ribosome_capacity", +1.4)
    m.add_edge("AMPK", "Ribosome_capacity", -0.6)
    m.add_edge("mTORC1", "Peroxisome_capacity", +0.4)
    m.add_edge("ROS", "Peroxisome_capacity", +0.3)
    m.add_edge("NFkB", "Peroxisome_capacity", -0.3)

    return m
"""
cell_model.py
-------------
Model: Contains all data classes, organelle properties, and OCM logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional
import math


# ---------------------------
# Build the Organelle Control Model (OCM)
# ---------------------------

def logistic(x: float, k: float = 6.0, x0: float = 0.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

def signed_effect(x: float, weight: float) -> float:
    centered = (x - 0.5) * 2.0
    return weight * centered

@dataclass
class Edge:
    source: str
    target: str
    weight: float

@dataclass
class Node:
    name: str
    value: float = 0.5
    baseline: float = 0.5
    decay: float = 0.08
    update_rule: Optional[Callable[["Model", str], float]] = None

@dataclass
class Model:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, name: str, value: float = 0.5, baseline: float = 0.5, decay: float = 0.08,
                 update_rule: Optional[Callable[["Model", str], float]] = None) -> None:
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
        for k, v in external_inputs.items():
            if k not in self.nodes:
                raise ValueError(f"Unknown input node: {k}")
            self.nodes[k].value = clamp(v)
        net_influence: Dict[str, float] = {name: 0.0 for name in self.nodes.keys()}
        for e in self.edges:
            src_val = self.nodes[e.source].value
            net_influence[e.target] += signed_effect(src_val, e.weight)
        new_values: Dict[str, float] = {}
        for name, node in self.nodes.items():
            if name.startswith("IN_"):
                new_values[name] = node.value
                continue
            if node.update_rule is not None:
                proposed = node.update_rule(self, name)
                proposed = (1 - node.decay) * proposed + node.decay * node.baseline
                new_values[name] = clamp(proposed)
                continue
            influence = net_influence[name]
            drive = logistic(influence, k=3.5, x0=0.0)
            blended = (1 - node.decay) * (0.65 * node.value + 0.35 * drive) + node.decay * node.baseline
            new_values[name] = clamp(blended)
        for name, v in new_values.items():
            self.nodes[name].value = v

# Organelle size and count data (literature-based, typical mammalian cell)
# Sources: Alberts et al., BioNumbers, NCBI Bookshelf
ORGANELLE_SIZES = {
    "cell_diameter_um": 20.0,           # Typical animal cell diameter
    "nucleus_diameter_um": 7.0,         # 5–10 µm
    "mito_length_um": 1.5,              # 1–2 µm
    "mito_width_um": 0.7,               # 0.5–1 µm
    "lysosome_diameter_um": 0.5,        # 0.1–1.2 µm
    "peroxisome_diameter_um": 0.3,      # 0.1–1 µm
    "golgi_region_width_um": 3.5,       # 2–5 µm
    "golgi_region_height_um": 2.0,      # region
    "er_tubule_diameter_um": 0.08,      # 60–100 nm
    "ribosome_diameter_um": 0.025,      # 20–30 nm
    "vesicle_diameter_um": 0.1,         # 50–200 nm
}

ORGANELLE_COUNTS = {
    "nucleus": 1,
    "mitochondria": 500,        # 100–2000
    "lysosomes": 200,           # 50–1000
    "peroxisomes": 200,         # 100–400
    "golgi": 1,
    "ribosomes": 2000000,       # 1–10 million
    "vesicles": 500,            # variable
}
