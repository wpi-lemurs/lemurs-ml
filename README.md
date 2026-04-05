# Lemurs Machine Learning Repository

This repository contains the data extraction, processing, feature engineering, and modeling workflows used for LEMURS analysis. It is designed to support reproducible experiments on passive behavioral data (screentime, health metrics, audio metadata) and survey-derived labels (PHQ-9, suicide risk, self-harm risk, and sleep risk).

## What This Repo Does

- Connects to PostgreSQL and extracts raw study data
- Cleans and standardizes passive data into analysis-ready tables
- Merges time-windowed passive features with survey labels
- Trains and evaluates baseline ML models (e.g., Logistic Regression, Random Forest)
- Supports PyTorch MLP experimentation on the same processed feature sets
- Produces confusion matrices and visualizations for model comparison

## Repository Structure

```text
lemurs-ml/
  README.md
  requirements.txt
  requirements-gpu.txt
  data/                         # Generated outputs, plots, merged datasets, cache artifacts
  src/
    config.py                   # Shared config/constants/path utilities
    database_service.py         # PostgreSQL extraction and DB utilities
    analysis/                   # Analysis scripts and exploratory outputs
    categorization/             # Daily question and app categorization logic
    data_processing/            # Cleaning, aggregation, merging, feature engineering
    modeling/                   # Classical ML and PyTorch training/evaluation scripts
    pipeline/                   # Reusable sklearn-style pipeline + transformers
    synthetic_data_generation/  # Synthetic data creation utilities (not used)
    testing/                    # Project tests
    visualization/              # Plotting and visualization helpers
```

## Core Workflow (End-to-End)

1. Extract source tables from the database (`database_service.py`)
2. Process passive streams into hourly/daily/weekly aggregates (`src/data_processing/`)
3. Build supervised datasets by merging with nearest survey targets
4. Train models across different lookback windows
5. Evaluate via confusion matrices and summary metrics (accuracy, F1, etc.)
6. Iterate on feature engineering (subwindows, app categories, null handling)

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If running GPU-enabled PyTorch workflows, use:

```bash
pip install -r requirements-gpu.txt
```

## Typical Usage Patterns

### 1) Data Extraction and Processing

- Use modules in `src/database_service.py` and `src/data_processing/` to pull and clean raw tables.
- Most scripts are designed for batch/offline updates when CSV/DB snapshots are refreshed.

### 2) Modeling with Existing Pipelines

- Use `src/pipeline/model_pipeline.py` and `src/pipeline/transformers.py` for reusable training flows.
- Example scripts in `src/pipeline/` demonstrate end-to-end usage and window sweeps.

### 3) Classical ML Experiments

- Use `src/modeling/model_screentime_time_windows.py` for lookback-window experiments.
- Compare model behavior across targets (risk labels or PHQ-9 derived labels).

### 4) PyTorch Experiments

- Use the PyTorch modeling scripts in `src/modeling/` to run MLP experiments with the same engineered features.
- Keep preprocessing logic aligned with `src/pipeline/` and `src/data_processing/` for fair comparisons.

## Data and Output Conventions

- `data/` stores generated artifacts such as:
  - confusion matrix images
  - sweep plots
  - merged modeling datasets
  - optional app-category cache files
- Scripts are generally built for repeatable regeneration when source data changes.

## Testing

Run tests from the project root:

```bash
pytest
```

You can also run a specific test file:

```bash
pytest src/testing/<test_file>.py
```

## Notes for Contributors

- Prefer adding new preprocessing logic in `src/data_processing/` and exposing reusable pieces in `src/pipeline/transformers.py`.
- Keep modeling scripts thin; place shared logic in helper modules.
- Preserve consistent naming for targets and label mappings across scripts.
- Avoid duplicating DB extraction logic; centralize in `database_service.py` or shared utilities.
