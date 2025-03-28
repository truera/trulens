# Development Guide

## Dev dependencies

### Node.js

TruLens uses Node.js for building react components for the dashboard. Install Node.js with the following command:
test

See this page for instructions on installing Node.js: [Node.js](https://nodejs.org/en/download/)


### Install homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install make

```bash
brew install make
echo 'PATH="$HOMEBREW_PREFIX/opt/make/libexec/gnubin:$PATH"' >> ~/.zshrc
```

## Clone the repository

```bash
git clone git@github.com:truera/trulens.git
cd trulens
```

## Install Git LFS

Git LFS is used avoid tracking larger files directly in the repository.

```bash
brew install git-lfs
git lfs install && git lfs pull
```

## (Optional) Install PyEnv for environment management

Optionally install a Python runtime manager like PyEnv. This helps install and switch across multiple python versions which can be useful for local testing.

```bash
curl https://pyenv.run | bash
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv
pyenv install 3.11  # python 3.11 recommended, python >= 3.9 supported
pyenv local 3.11  # set the local python version
```

For more information on PyEnv, see the [pyenv repository](https://github.com/pyenv/pyenv).

## Install Poetry

TruLens uses Poetry for dependency management and packaging. Install Poetry with the following command:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

You may need to add the Poetry binary to your `PATH` by adding the following line to your shell profile (e.g. `~/.bashrc`, `~/.zshrc`):

```bash
export PATH=$PATH:$HOME/.local/bin
```

## Install the TruLens project

Install `trulens` into your environment by running the following command:

```bash
poetry install
```

This will install dependencies specified in `poetry.lock`, which is built from `pyproject.toml`.

To synchronize the exact environment specified by `poetry.lock` use the `--sync` flag. In addition to installing relevant dependencies, `--sync` will remove any packages not specified in `poetry.lock`.

```bash
poetry install --sync
```

These commands install the `trulens` package and all its dependencies in editable mode, so changes to the code are immediately reflected in the environment.

For more information on Poetry, see [poetry docs](https://python-poetry.org/docs/).

## Install pre-commit hooks

TruLens uses pre-commit hooks for running simple syntax and style checks before committing to the repository. Install the hooks with the following command:

```bash
pre-commit install
```

For more information on pre-commit, see [pre-commit.com](https://pre-commit.com/).

## Install ggshield

TruLens developers use ggshield to scan for secrets locally in addition to gitguardian in CLI. Install and authenticate to ggshield with the following commands:

```bash
brew install gitguardian/tap/ggshield
ggshield auth login
```

Then, ggshield can be run with the following command from trulens root directory to scan the full repository:

```bash
ggshield secret scan repo ./
```

It can also be run with smaller scope, such as only for docs with the following as included in `make docs-upload`

```bash
ggshield secret scan repo ./docs/
```

## Helpful commands

### Formatting

Runs [ruff formatter](https://docs.astral.sh/ruff/formatter/) to format all python and notebook files in the repository.

```bash
make format
```

### Linting

Runs [ruff linter](https://docs.astral.sh/ruff/linter/) to check for style issues in the codebase.

```bash
make lint
```

### Run tests

```bash
# Runs tests from tests/unit with the current environment
make test-unit
```

Tests can also be run in two predetermined environments: `required` and `optional`.
The `required` environment installs only the required dependencies, while `optional` environment installs all optional dependencies (e.g LlamaIndex, OpenAI, etc).

```bash
# Installs only required dependencies and runs basic unit tests
make test-unit-basic
```

```bash
# Installs optional dependencies and runs unit tests
make test-unit-all
```

To install a environment matching the dependencies required for a specific test, use the following commands:

```bash
make env-required  # installs only required dependencies

make env-optional  # installs optional dependencies
```

### Get Coverage Report

Uses the `pytest-cov` plugin to generate a coverage report (`coverage.xml` & `htmlcov/index.html`)

```bash
make coverage
```

### Update Poetry Locks

Recreates lockfiles for all packages. This runs `poetry lock` in the root directory and in each package.

```bash
make lock
```

### Update package version

To update the version of a specific package:

```bash
# If updating version of a specific package
cd src/[path-to-package]
poetry version [major | minor | patch]
```

This can also be done manually by editing the `pyproject.toml` file in the respective directory.

### Build all packages

Builds `trulens` and all packages to `dist/*`

```bash
make build
```

### Upload packages to PyPI

To upload all packages to PyPI, run the following command with the `TOKEN` environment variable set to your PyPI token.

```bash
TOKEN=... make upload-all
```

To upload a specific package, run the following command with the `TOKEN` environment variable set to your PyPI token. The package name should exclude the `trulens` prefix.

```bash
# Uploads trulens-providers-openai
TOKEN=... make upload-trulens-providers-openai
```

### Deploy documentation locally

To deploy the documentation locally, run the following command:

```bash
make docs-serve
```
