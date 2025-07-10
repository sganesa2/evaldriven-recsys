# evaldriven-recsys

**evaldriven-recsys** is an evaluation-driven, LLM-based recommendation system designed to facilitate research and experimentation with retrieval, ranking, and evaluation pipelines for product recommendations. The system leverages advanced language models for semantic understanding, named entity recognition, and product evaluation, and supports modular experimentation with different retrieval and scoring strategies.

## Features

- Modular recommendation engine with pluggable retrieval and ranking models
- Support for semantic and keyword-based query interpretation
- Evaluation pipelines for benchmarking recommendation quality (e.g., nDCG)
- Utilities for preprocessing and analyzing test and output datasets
- Jupyter notebook for creating evaluation dataframes

## Getting Started

Follow these steps to set up and run the project:

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd evaldriven-recsys
```

### 2. Set Up Python Environment

- Ensure you have Python 3.11 or higher installed.
- (Optional) Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the required dependencies using `pip`:

```sh
pip install -r requirements.txt
```
Or, if using `pyproject.toml`:

```sh
pip install .
```

For development dependencies (recommended):

```sh
pip install -e .[dev]
```

### 4. Prepare Data

- Place your test and output JSON files in the appropriate directory (see usage in `src/create_evaluation_dataframe.ipynb`).
- Update file paths as needed in your scripts or notebooks.

### 5. Running the Recommendation Engine

You can run the main recommendation engine script:

```sh
python src/rec_engine_run.py
```

This will:
- Initialize the Weave experiment environment
- Load test data using `preprocess_test_json`
- Run the recommendation engine pipeline

### 6. Evaluation and Analysis

- Use the Jupyter notebook `src/create_evaluation_dataframe.ipynb` to generate evaluation dataframes and compute metrics like nDCG.
- Launch Jupyter:

```sh
jupyter notebook src/create_evaluation_dataframe.ipynb
```

### 7. Running Tests

To run the test suite:

```sh
pytest tests/
```

## Project Structure

- `src/` – Main source code
  - `rec_engine_run.py` – Script to run the recommendation engine
  - `models/` – Model implementations (retrievers, encoders, evaluation models)
  - `create_evaluation_dataframe.ipynb` – Notebook for evaluation and analysis
  - `ignore_file.py`, `retriever_evalpipe_run.py`, etc. – Additional scripts/utilities
- `tests/` – Unit and integration tests
- `pyproject.toml` – Project and dependency configuration

## Notes

- Some scripts require specific input file formats (see docstrings or code comments).
- The project uses [Weave](https://wandb.github.io/weave/) for experiment tracking.
- For LLM-based modules, ensure you have access to required APIs or models.

---

For more details, refer to the code and comments in each script.