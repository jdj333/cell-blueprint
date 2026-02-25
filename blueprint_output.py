"""
blueprint_output.py (robust version)
-----------------------------------
Entry point: Runs a scenario and produces the visualization, adapting to model changes.
"""

from cell_model import build_ocm
from cell_controller import run_simulation, scenario_baseline, pretty_last
from cell_view import render_cell_blueprint


def main():
    # 1. Build the model
    model = build_ocm()

    # 2. Get the default scenario schedule (steps inferred from scenario)
    schedule = scenario_baseline()
    steps = len(schedule)

    # 3. Run the simulation (track all model nodes dynamically)
    # Try to get all node names from the model if possible
    try:
        track = list(model.nodes.keys())
    except AttributeError:
        # Fallback: use default track list from controller
        track = None

    history = run_simulation(model, schedule, steps, track=track)

    # 4. Get the final state (last value for each tracked variable)
    state = pretty_last(history)

    # 5. Render the cell blueprint
    render_cell_blueprint(state)
    print("Cell blueprint rendered to 'cell_blueprint.png'.")


if __name__ == "__main__":
    main()
