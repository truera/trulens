# Make targets useful for developing TruLens.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .

SHELL := /bin/bash
REPO_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
POETRY_DIRS := $(shell find . -not -path "./dist/*" -maxdepth 4 -name "*poetry.lock" -exec dirname {} \;)

# Create the poetry env for building website, docs, formatting, etc.
.env/create:
	poetry install --sync

# Lock the poetry dependencies for all the subprojects.
poetry-lock-all: $(POETRY_DIRS)
	for dir in $(POETRY_DIRS); do \
		poetry lock -C $$dir; \
	done


# Run the ruff linter.
lint: .env/create
	poetry run ruff check --fix

# Run the ruff formatter.
format: .env/create
	poetry run ruff format

precommit-hooks: .env/create
	poetry run pre-commit install

run-precommit: precommit-hooks
	poetry run pre-commit run --all-files --show-diff-on-failure

# Start a jupyer lab instance.
lab: .env/create
	poetry run jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

# Serve the documentation website.
docs-serve: .env/create
	poetry run mkdocs serve -a 127.0.0.1:8000

# Serve the documentation website.
docs-serve-debug: .env/create
	poetry run mkdocs serve -a 127.0.0.1:8000 --verbose

# The --dirty flag makes mkdocs not regenerate everything when change is detected but also seems to
# break references.
docs-serve-dirty: .env/create
	poetry run mkdocs serve --dirty -a 127.0.0.1:8000

# Build the documentation website.
docs: .env/create $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs build --clean
	rm -Rf site/overrides

docs-upload: .env/create $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs gh-deploy

# Check that links in the documentation are valid. Requires the lychee tool.
docs-linkcheck: site
	lychee "site/**/*.html"

# Start the trubot slack app.
trubot:
	poetry run python -u examples/trubot/trubot.py

# Run a test with the optional flag set, meaning @optional_test decorated tests
# are run.
required-env:
	poetry install --only required,tests --sync

optional-env:
	poetry install --sync --verbose

coverage:
	ALLOW_OPTIONAL_ENV_VAR=true pytest --rootdir=. tests/* --cov src --cov-report html

# Runs required tests
test-%-required: required-env
	make test-$*

# Runs required tests, but allows optional dependencies to be installed.
test-%-allow-optional: required-env
	ALLOW_OPTIONAL_ENV_VAR=true make test-$*

# Requires the full optional environment to be set up.
test-%-optional: optional-env
	TEST_OPTIONAL=true make test-$*

# Run the unit tests, those in the tests/unit. They are run in the CI pipeline frequently.
test-unit:
	poetry run pytest --rootdir=. tests/unit/*

# Tests in the e2e folder make use of possibly costly endpoints. They
# are part of only the less frequently run release tests.
test-e2e:
	poetry run pytest --rootdir=. tests/e2e/*

# Runs the notebook test
test-notebook:
	poetry run pytest --rootdir=. tests/docs_notebooks/*

# Release Steps:
## Step: Clean repo:
clean:
	git clean -fxd

## Step: Build wheels
build-dashboard: .env/create
	rm -rf src/dashboard/*.egg-info
	poetry run python -m build src/dashboard -o $(REPO_ROOT)/dist;

build: $(POETRY_DIRS)
	for dir in $(POETRY_DIRS); do \
		pushd $$dir; \
		poetry build -o $(REPO_ROOT)/dist; \
		popd; \
	done
	make build-dashboard

## Usage: TOKEN=... make upload
upload:
	poetry run twine upload -u __token__ -p $(TOKEN) dist/*.whl
