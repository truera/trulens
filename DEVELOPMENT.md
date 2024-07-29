# Development Guide

## Clone the repository

```bash
git clone git@github.com:truera/trulens.git
cd trulens
```

## (Optional) Install PyEnv for environment management

Optionally install a Python runtime manager like PyEnv. This helps install and switch across multiple python versions which can be useful for local testing.

```bash
curl https://pyenv.run | bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
pyenv install 3.11  # python >= 3.9 supported
pyenv local 3.11
```

## Install Poetry

TruLens uses Poetry for dependency management and packaging.

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Install `trulens` package into your directory, along with relevant dependencies.

```bash
poetry install
```

## Install pre-commit hooks

TruLens uses pre-commit hooks for running simple syntax and style checks before committing to the repository.

```bash
pre-commit install
```

## Helpful commands

### Formatting

Runs ruff formatter

```bash
make format
```

### Linting

Runs ruff linter

```bash
make lint
```

### Run tests

```bash
# Runs unit tests with the current environment
make test-unit
```

Tests can be run in two environments: `required` and `optional`.
The `required` environment installs only the required dependencies, while `optional` environment installs all optional dependencies (e.g LlamaIndex, OpenAI, etc).

```bash
# Installs only required dependencies and runs unit tests
make test-unit-required
```

```bash
# Installs optional dependencies and runs unit tests
make test-unit-optional
```

### Get Coverage Report

Tests come in two flavors: required and optional

```bash
make coverage
```

### Generate Poetry Lockfiles

Generate the poetry lockfile for all subdirectories

```bash
make poetry-lock
```

### Build all packages

Generate the poetry lockfile for all subdirectories

```bash
make build
```
