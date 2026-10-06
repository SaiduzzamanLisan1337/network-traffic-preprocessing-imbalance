# Data

Do not commit proprietary, sensitive, or very large IDS datasets to this repository.

Place a compatible CSV at `data/raw/ids.csv` and set `data.target_column` in `configs/experiment.yaml`.

For CICIDS2017, combine/clean the flow CSV files according to the official dataset documentation and map the attack label column to the configured target. The pipeline keeps numeric traffic features and removes rows containing NaN/Inf values or exact duplicates.

The repository also supports a deterministic synthetic demo dataset when the configured CSV is absent. This is intended for smoke tests and reproducibility of the code path, not for research conclusions.
