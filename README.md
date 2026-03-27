# federated-ueba: A Flower / PyTorch app

## Prepare the data
- Due to the way the feature extraction script works, a Linux environment is required.
- Download the CERT dataset and move the "r4.2" folder under the "dataset" folder.
- Run the "feature_extraction.py" script.
- Run the "insider_detection.py" script.
- Run the "temporal.py" script.

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

## Run with the Simulation Engine

Run `flwr run .` command to run a local simulation.
