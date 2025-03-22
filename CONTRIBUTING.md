# Contributing to Renrag ColBERT

Thank you for considering contributing to Renrag ColBERT! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Implement your changes
4. Add tests for your changes
5. Run the test suite to ensure all tests pass
6. Submit a pull request

## Development Setup

1. Clone your fork of the repository
2. Install the development dependencies:
   ```
   pip install -e ".[dev]"
   ```
3. Install pre-commit hooks:
   ```
   pre-commit install
   ```

## Testing

Run the test suite with:

```
pytest
```

## Coding Standards

- Follow [PEP 8](https://pep8.org/) style guide
- Write descriptive docstrings in Google style format
- Include type hints for function arguments and return values
- Write tests for new features and bug fixes

## Pull Request Process

1. Update the README.md with details of changes if appropriate
2. Update the example.py if relevant
3. The PR should work for Python 3.7+
4. The PR will be merged once it passes all checks and is approved by a maintainer

## Release Process

1. Update version number in setup.py
2. Update CHANGELOG.md
3. Create a new GitHub release with the version number
4. Publish the package to PyPI

Thank you for your contributions!