# federated-ueba

## Prepare the data
- Due to the way the feature extraction script works, a Linux environment is required.
- Download the CERT dataset and move the "r4.2" folder under the "dataset" folder.
- Run the "feature_extraction.py" script.
- Run the "temporal.py" script to create the temporal data used in the project.

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

## Run the project

To run the federated learning simulation and insider detection:

```bash
python federated_insider_detection.py
```

To train the centralized model (required for comparison):

```bash
python train_centralized.py
```

To clear previous results:

```bash
python clear_results.py
```

## Comment on the results
Run the "compare_results.py" script to analyze the results.
