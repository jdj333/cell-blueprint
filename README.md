# cell-blueprint

## Overview

This project models and visualizes a eukaryotic cell using a systems-level Organelle Control Model (OCM). The codebase is organized using the Model-View-Controller (MVC) pattern:

- **Model** (`cell_model.py`): Defines organelle data, sizes, and the OCM logic.
- **Controller** (`cell_controller.py`): Handles simulation, scenarios, and process logic.
- **View** (`cell_view.py`): Renders the cell blueprint as a to-scale 2D schematic.

## How to Run

1. **Install requirements** (if not already):
	```bash
	pip install matplotlib
	```

2. **Run the blueprint visualization:**
	```bash
	python blueprint-output.py
	```
	This will generate a `cell_blueprint.png` image in the current directory, visualizing the cell state for the default scenario.

3. **Modify scenarios:**
	- To try different scenarios, edit the scenario selection in `blueprint-output.py` or use the controller to define your own.

## File Structure

- `cell_model.py` — Model: organelle data, OCM logic
- `cell_controller.py` — Controller: simulation, scenarios
- `cell_view.py` — View: rendering logic
- `blueprint-output.py` — Entry point: runs a scenario and produces the visualization

## Example Output

The output is a to-scale PNG schematic of a cell, showing organelles, processes, and outputs based on the model state.