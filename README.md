# Evaluating Preprocessing and Class-Imbalance Strategies for Network Intrusion Detection

[![CI](https://github.com/YOUR_USERNAME/network-traffic-preprocessing-imbalance/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/network-traffic-preprocessing-imbalance/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/YOUR_USERNAME/network-traffic-preprocessing-imbalance/actions)
[![License](https://img.shields.io/github/license/YOUR_USERNAME/network-traffic-preprocessing-imbalance)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A reproducible research framework for evaluating how preprocessing and class-imbalance strategies affect minority-attack detection in network intrusion detection systems (IDS). It compares standardization, feature selection, class weighting, SMOTE, and combined strategies using stratified cross-validation and imbalance-aware metrics.

## Table of Contents

- [Project Overview](#project-overview)
- [Research Objective](#research-objective)
- [Research Questions](#research-questions)
- [Key Features](#key-features)
- [Experimental Pipeline](#experimental-pipeline)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Using CICIDS2017](#using-cicids2017)
- [Configuration](#configuration)
- [Experimental Strategies](#experimental-strategies)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Statistical Analysis](#statistical-analysis)
- [Results](#results)
- [Output Files](#output-files)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Reproducibility](#reproducibility)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Research Scope and Limitations](#research-scope-and-limitations)
- [Future Research Directions](#future-research-directions)

## Project Overview

Network intrusion datasets are frequently highly imbalanced. Benign traffic and common attacks can dominate the dataset while important attack classes are rare. This can produce misleadingly high accuracy even when a classifier detects few minority attacks.

This project systematically compares:

- Baseline learning
- Feature standardization
- Feature selection
- Class weighting
- SMOTE oversampling
- Combined preprocessing and imbalance handling

The framework emphasizes minority-class recall and F1-score while also reporting macro-F1, balanced accuracy, PR-AUC, precision, accuracy, and variability across cross-validation folds.

## Research Objective

The primary objective is to determine whether common preprocessing and class-imbalance strategies produce meaningful improvements in minority-attack detection and whether those improvements remain consistent across validation folds.

## Research Questions

### RQ1 — Standardization
Does feature standardization improve classification performance for network intrusion detection?

### RQ2 — Feature Selection
Does reducing the feature space improve minority-attack detection while maintaining overall performance?

### RQ3 — Class Weighting
Does cost-sensitive learning improve recall for underrepresented attack classes?

### RQ4 — SMOTE
Does synthetic minority oversampling improve minority-class detection?

### RQ5 — Combined Strategy
Does combining preprocessing and imbalance-handling techniques outperform individual strategies?

### RQ6 — Evaluation
Do improvements in minority recall and F1 justify possible reductions in precision or overall accuracy?

## Key Features

### Machine Learning
- Logistic Regression
- Random Forest
- Extensible model architecture

### Preprocessing
- Missing-value handling
- Standardization
- Feature selection
- Leakage-safe transformation pipelines

### Imbalance Handling
- Baseline
- Class weighting
- SMOTE
- Combined strategy

### Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Balanced accuracy
- PR-AUC
- Minority precision
- Minority recall
- Minority F1
- Mean and standard deviation across folds
- Training time

### Reproducibility
- Stratified 5-fold cross-validation
- Fixed random seeds
- YAML configuration
- Statistical comparisons
- Automated tables and figures
- Unit tests
- GitHub Actions CI

## Experimental Pipeline

```text
Network Traffic Dataset
          |
          v
   Data Validation
          |
          v
   Preprocessing
          |
          v
 Stratified 5-Fold CV
          |
    +-----+-----+
    |           |
    v           v
Preprocessing  Imbalance
 Strategies   Strategies
    |           |
    +-----+-----+
          |
          v
Logistic Regression + Random Forest
          |
          v
 Metrics + Statistical Analysis
          |
          v
    Tables + Figures
```

## Project Architecture

```text
network-traffic-preprocessing-imbalance/
├── .github/
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/
├── configs/
│   └── experiment.yaml
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── reports/
│   ├── methodology.md
│   └── final_report.md
├── results/
│   ├── figures/
│   └── tables/
├── scripts/
│   ├── preprocess.py
│   └── run_experiments.py
├── src/network_imbalance/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── experiment.py
│   ├── feature_selection.py
│   ├── imbalance.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── statistics.py
│   └── visualization.py
├── tests/
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Dataset

The primary target dataset is **CICIDS2017** from the Canadian Institute for Cybersecurity.

The project is intended to work with the CICIDS2017 CSV files, including traffic associated with benign activity, DDoS, DoS, PortScan, Web attacks, Infiltration, Botnet, and brute-force activity.

Do not commit the large raw dataset files to GitHub. Keep them locally under:

```text
data/raw/
```

Example:

```text
data/raw/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

## Prerequisites

- Python 3.10+
- Git
- pip
- Python virtual-environment support
- Adequate RAM for the selected dataset

Check your environment:

```bash
python3 --version
git --version
```

## Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/network-traffic-preprocessing-imbalance.git
cd network-traffic-preprocessing-imbalance
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate it

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run tests

```bash
pytest -q
```

## Quick Start

The repository includes a deterministic demonstration workflow so the project can be tested without downloading the full dataset.

From the project root:

```bash
source .venv/bin/activate
python -m network_imbalance.experiment
```

A successful run ends with:

```text
Experiment complete.
```

The experiment runs two models across the configured preprocessing and imbalance strategies using stratified cross-validation and produces tables and figures under `results/`.

## Using CICIDS2017

Place the downloaded CICIDS2017 CSV files under:

```text
data/raw/
```

The experiment configuration can point to a prepared combined dataset:

```yaml
data:
  path: data/raw/ids.csv
  target_column: Label

experiment:
  random_state: 42
  cv_folds: 5
```

For a full dataset experiment, first inspect:

- Dataset dimensions
- Feature names
- Missing values
- Duplicate records
- Infinite values
- Non-numeric fields
- Label distribution
- Class frequencies

For very large files, use a controlled preprocessing/sampling strategy rather than loading everything into memory at once.

## Configuration

Experiment settings are defined in:

```text
configs/experiment.yaml
```

Keeping settings in YAML makes experiments easier to reproduce and compare without editing source code.

## Experimental Strategies

### 1. Baseline

Reference experiment without explicit class-imbalance correction.

### 2. Standardization

Scales numerical features before model training.

### 3. Feature Selection

Selects a subset of informative features to investigate the effect of dimensionality reduction.

### 4. Class Weighting

Makes minority-class errors more costly during model training.

### 5. SMOTE

Generates synthetic minority samples within training folds.

### 6. Combined

Evaluates multiple preprocessing and imbalance-handling operations together.

## Models

### Logistic Regression

A relatively interpretable linear baseline for studying the effect of preprocessing and class imbalance.

### Random Forest

A nonlinear ensemble model capable of learning complex relationships in network-traffic features.

The architecture is modular so additional models can be added later.

## Evaluation Metrics

Accuracy is reported but is not treated as the primary metric because it can be misleading under severe class imbalance.

The project emphasizes:

### Minority-Class Recall

```text
Recall = TP / (TP + FN)
```

Measures how many minority attacks are successfully detected.

### Minority-Class F1

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

Balances minority precision and recall.

### Macro-F1

Gives each class equal importance.

### Balanced Accuracy

Averages recall across classes and is more informative than raw accuracy for imbalanced datasets.

### PR-AUC

Measures the precision-recall trade-off and is especially useful for rare positive/minority classes.

Additional metrics include precision, recall, F1, accuracy, standard deviations, and training time.

## Statistical Analysis

The project compares strategies using paired cross-validation results to determine whether performance differences are consistent across folds.

The statistical module is:

```text
src/network_imbalance/statistics.py
```

Statistical tests should be interpreted with effect sizes, confidence intervals, dataset characteristics, and the assumptions of the selected test. Statistical significance alone does not establish practical significance.

## Results

The demonstration experiment can show patterns such as:

```text
model                 strategy          minority_recall
--------------------------------------------------------
logistic_regression   baseline               0.265
logistic_regression   smote                  0.704
logistic_regression   class_weighted         0.717

random_forest         baseline               0.101
random_forest         smote                  0.451
random_forest         combined               0.469
```

These values illustrate why overall accuracy should not be the only evaluation criterion for IDS research.

**These are demonstration results, not final CICIDS2017 results.**

## Output Files

Results are organized under:

```text
results/
├── figures/
└── tables/
```

Typical outputs include:

- Fold-level metrics
- Aggregated metrics
- Minority-class metrics
- Statistical comparisons
- Metric comparison plots
- Confusion matrices

## Data Leakage Prevention

Preprocessing and feature-selection steps should be learned from training folds whenever applicable.

SMOTE must be applied **only to the training portion of each cross-validation fold**.

Correct:

```text
Dataset
  |
  v
Stratified Split
  |
  +------------------+
  |                  |
  v                  v
Train Fold       Validation Fold
  |
  v
SMOTE
  |
  v
Model
  |
  +------------------+
           |
           v
       Evaluation
```

This prevents synthetic samples or learned transformations from leaking into validation data.

## Reproducibility

The project uses:

- Fixed random seeds
- Stratified cross-validation
- YAML configuration
- Leakage-safe preprocessing
- Version-controlled source code
- Automated tests
- Documented methodology

For publication-quality experiments, record the dataset version, preprocessing procedure, Python/package versions, random seed, cross-validation method, model parameters, and experiment date.

## Testing

Run:

```bash
pytest -q
```

or:

```bash
pytest -v
```

Tests cover important pipeline components including configuration, data processing, evaluation, feature selection, imbalance handling, and model construction.

## Continuous Integration

GitHub Actions runs automated checks on pushes and pull requests.

Workflow file:

```text
.github/workflows/ci.yml
```

The CI pipeline helps detect broken code before changes are merged.

## Contributing

Contributions are welcome.

Read:

```text
CONTRIBUTING.md
```

Typical workflow:

```bash
git checkout -b feature/your-feature
pytest -q
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature
```

Then open a Pull Request.

Please ensure tests pass, new functionality has appropriate tests, documentation is updated, and raw datasets or sensitive information are not committed.

## Citation

If this project contributes to your research, use the metadata in:

```text
CITATION.cff
```

GitHub can use this file to expose citation information for the repository.

## License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

## Acknowledgments

This project builds upon publicly available intrusion-detection research datasets and the Python scientific-computing ecosystem, including:

- NumPy
- pandas
- scikit-learn
- imbalanced-learn
- SciPy
- Matplotlib
- PyYAML
- pytest

Special acknowledgment is given to the Canadian Institute for Cybersecurity for the CICIDS2017 dataset.

## Research Scope and Limitations

This repository is a research and experimental framework, not a production-ready intrusion-prevention system.

Results can vary with dataset version, preprocessing choices, feature engineering, class distribution, model hyperparameters, hardware, cross-validation design, random seed, and attack taxonomy.

High benchmark performance should not automatically be interpreted as real-world IDS effectiveness.

## Future Research Directions

Potential extensions include:

- Cross-dataset evaluation
- Temporal validation
- Anomaly-based detection
- Explainable intrusion detection
- Robustness and distribution-shift analysis
- Deep learning
- Online/real-time detection
- Domain adaptation
- Evaluation on real network traffic under appropriate privacy controls
- Japanese network-environment evaluation

## Project Status

**Status: Active Research / Experimental**

The current version provides a reproducible framework for evaluating preprocessing and class-imbalance strategies in network intrusion detection.

## Author

**Md. Saiduzzaman**

Research interests include:

- Intrusion Detection Systems
- Anomaly Detection
- Machine Learning
- Network Security
- Imbalanced Learning
- Cybersecurity
- Reliable Machine-Learning Evaluation

---

<p align="center">
  <b>Network Security × Machine Learning × Reproducible Research</b>
</p>
