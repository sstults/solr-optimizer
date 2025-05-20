# Contributing to Solr Optimizer

Thank you for your interest in contributing to the Solr Optimizer project! This document provides guidelines and instructions for contributing.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sstults/solr-optimizer.git
   cd solr-optimizer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   pip install -e ".[dev]"  # Includes development dependencies
   ```

## Coding Standards

We follow these guidelines for code quality and consistency:

### Python Coding Standards

- **Python Version**: Code should be compatible with Python 3.9+
- **PEP 8**: Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- **Line Length**: Maximum line length of 88 characters (as per Black formatter)
- **Indentation**: 4 spaces (no tabs)
- **String Quotes**: Prefer double quotes for docstrings, single quotes for regular strings
- **Imports**: Group and sort imports according to isort configuration (stdlib, third-party, local)

### Tooling

- **Black**: We use [Black](https://github.com/psf/black) for code formatting
- **isort**: We use [isort](https://github.com/PyCQA/isort) for sorting imports
- **flake8**: We use [flake8](https://github.com/PyCQA/flake8) for linting
- **mypy**: We use [mypy](https://github.com/python/mypy) for static type checking

Run these tools before submitting a pull request:
```bash
# Format code
black solr_optimizer tests
isort solr_optimizer tests

# Check code quality
flake8 solr_optimizer tests
mypy solr_optimizer
```

## Documentation Guidelines

### Docstrings

- All modules, classes, methods, and functions should include docstrings
- Follow [Google's Python docstring style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints in function/method signatures (leveraging Python's typing module)
- Document parameters, return values, and exceptions

Example:
```python
def calculate_metric(self, metric_name: str, results: List[str], 
                    judgments: Dict[str, int], depth: int) -> float:
    """
    Calculate a single relevance metric for a query.
    
    Args:
        metric_name: Name of the metric to calculate (e.g., 'ndcg', 'precision')
        results: List of document IDs in result order
        judgments: Dictionary of document ID to relevance judgment
        depth: Depth at which to calculate the metric (e.g., 10 for NDCG@10)
        
    Returns:
        The calculated metric value
        
    Raises:
        ValueError: If the metric_name is not supported or parameters are invalid
    """
    # Implementation
```

### Comments

- Use comments sparingly, preferring self-documenting code
- Comments should explain "why" rather than "what" (the code should be clear enough to show what it's doing)
- Keep comments up-to-date with code changes

## Testing

- All new features should include unit tests
- All bug fixes should include regression tests
- Aim for high test coverage, especially for critical components
- Use pytest for test discovery and execution

Running tests:
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=solr_optimizer
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass and code quality tools show no issues
4. Update documentation as necessary
5. Submit a pull request with a clear description of the changes

## Versioning

We follow [Semantic Versioning](https://semver.org/). When proposing version changes:

- **MAJOR** version for incompatible API changes
- **MINOR** version for added functionality in a backwards compatible manner
- **PATCH** version for backwards compatible bug fixes

## Code Review

All submissions require review. We use GitHub pull requests for this purpose:

1. Submit your PR
2. Maintainers will review your code
3. Address any suggested changes
4. Once approved, a maintainer will merge your changes

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
