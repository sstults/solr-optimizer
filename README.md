# Solr Optimizer

A framework for optimizing Solr queries through systematic experimentation, following a modular, agent-based architecture.

## Overview

Solr Optimizer provides a set of tools and components for tuning and optimizing Solr search queries based on relevance judgments and user-defined metrics. The framework allows for:

- Running iterative experiments with different query configurations
- Evaluating results using standard IR metrics (NDCG, precision, recall, etc.)
- Comparing iterations and understanding why rankings changed
- Tracking experiment history and configurations

## Architecture

The framework is built around a modular agent-based architecture:

- **Experiment Manager**: Orchestrates the workflow between agents
- **Query Tuning Agent**: Generates and modifies query configurations
- **Solr Execution Agent**: Interfaces with SolrCloud to run queries
- **Metrics & Evaluation Agent**: Calculates relevance metrics
- **Logging & Tracking Agent**: Records experiment history
- **Comparison & Analysis Agent**: Analyzes differences between iterations

## Installation

### Requirements

- Python 3.9+
- Apache SolrCloud instance

### Setup

```bash
# Clone the repository
git clone https://github.com/sstults/solr-optimizer.git
cd solr-optimizer

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Set up a new experiment
python -m solr_optimizer setup --corpus my_collection --queries queries.txt --judgments judgments.txt

# Run an iteration
python -m solr_optimizer run --experiment-id exp-12345 --params query_params.json

# Compare iterations
python -m solr_optimizer compare --experiment-id exp-12345 --iter1 iter-1 --iter2 iter-2

# List iterations
python -m solr_optimizer list --experiment-id exp-12345
```

### Python API

```python
from solr_optimizer.core.default_experiment_manager import DefaultExperimentManager
from solr_optimizer.models.experiment_config import ExperimentConfig

# Initialize agents (implementation-specific)
# ...

# Initialize experiment manager
manager = DefaultExperimentManager(
    query_tuning_agent=query_agent,
    solr_execution_agent=solr_agent,
    metrics_agent=metrics_agent,
    logging_agent=logging_agent,
    comparison_agent=comparison_agent
)

# Set up an experiment
config = ExperimentConfig(
    corpus="my_collection",
    queries=["search query 1", "search query 2"],
    judgments={
        "search query 1": {"doc1": 3, "doc2": 1, "doc3": 0},
        "search query 2": {"doc4": 2, "doc5": 3, "doc6": 1}
    },
    primary_metric="ndcg",
    metric_depth=10
)

experiment_id = manager.setup_experiment(config)
```

## Development

### Running Tests

```bash
pytest
```

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

```bash
# Format code
black solr_optimizer tests
isort solr_optimizer tests

# Run linting
flake8 solr_optimizer tests
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
