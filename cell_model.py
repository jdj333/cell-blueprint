"""
cell_model.py
-------------
Model: Contains all data classes, organelle properties, and OCM logic.
"""


# ---------------------------
# Build the Organelle Control Model (OCM)
# ---------------------------

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

# Organelle size and property data (for view)
ORGANELLE_SIZES = {
    "cell_diameter_um": 20.0,
    "nucleus_diameter_um": 5.5,
    "mito_length_um": 1.6,
    "mito_width_um": 0.6,
    "lysosome_diameter_um": 0.5,
    "peroxisome_diameter_um": 0.45,
    "golgi_region_width_um": 3.0,
    "golgi_region_height_um": 2.0,
}
