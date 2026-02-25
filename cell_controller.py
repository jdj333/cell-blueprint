"""
cell_controller.py
-----------------
Controller: Handles simulation, scenario selection, and mapping process logic to the model.
"""

from typing import Dict, List, Optional
from cell_model import Model

def run_simulation(model: Model,
                   schedule: List[Dict[str, float]],
                   steps: int,
                   dt: float = 1.0,
                   track: Optional[List[str]] = None) -> Dict[str, List[float]]:
    if track is None:
        track = [
            "AMPK", "mTORC1", "SIRT1", "SIRT3", "PGC1a", "TFEB", "NFkB", "ROS",
            "Mito_capacity", "Lysosome_capacity", "Autophagy_flux",
            "Proteasome_capacity", "ER_capacity", "Ribosome_capacity", "Peroxisome_capacity",
            "OUT_ATP_capacity", "OUT_proteostasis", "OUT_lipofuscin_pressure", "OUT_inflammatory_pressure"
        ]
    history: Dict[str, List[float]] = {k: [] for k in track}
    for t in range(steps):
        inputs = schedule[min(t, len(schedule) - 1)]
        model.step(inputs, dt=dt)
        for k in track:
            history[k].append(model.get(k))
    return history

def pretty_last(history: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: v[-1] for k, v in history.items() if v}

# Example scenarios (can be extended)
