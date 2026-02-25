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
        # Ensure inputs is a dict, not a list
        if isinstance(inputs, list) and len(inputs) == 1 and isinstance(inputs[0], dict):
            # Unwrap if schedule is [[dict], [dict], ...] instead of [dict, dict, ...]
            inputs = inputs[0]
        if not isinstance(inputs, dict):
            raise TypeError(f"Each schedule entry must be a dict, got {type(inputs)} at step {t}.")
        model.step(inputs, dt=dt)
        for k in track:
            history[k].append(model.get(k))
    return history

def pretty_last(history: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: v[-1] for k, v in history.items() if v}


# Example scenario: baseline (all inputs at baseline)
def scenario_baseline(steps: int = 100) -> list:
    # All inputs at their default baseline values
    base_inputs = {
        "IN_fasting": 0.0,
        "IN_endurance_exercise": 0.0,
        "IN_resistance_training": 0.0,
        "IN_protein_aa": 0.5,
        "IN_caloric_excess": 0.2,
        "IN_inflammation": 0.2,
        "IN_iron_load": 0.3,
        "IN_NAD_availability": 0.5,
        "IN_sleep_quality": 0.6
    }
    schedule = [base_inputs.copy() for _ in range(steps)]
    return schedule
