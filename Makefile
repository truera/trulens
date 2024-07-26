# Make targets useful for developing TruLens-Eval.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .

SHELL := /bin/bash

# Make targets useful for developing TruLens.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .


# Makes all commands for each target to execute in a single shell. If this is
# not set, conda env setup must be executed with each command that needs to be
# run in a conda env.
.ONESHELL:

# Create the poetry env for building website, docs, formatting, etc.
.env/create:
	poetry install --sync

# Run the code formatter.
lint: .env/create
	ruff check --fix

format: .env/create
	ruff format

# Start a jupyer lab instance.
lab:
	poetry run jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

# Serve the documentation website.
serve: .env/create
	poetry run mkdocs serve -a 127.0.0.1:8000

# Serve the documentation website.
serve-debug: .env/create
	poetry run mkdocs serve -a 127.0.0.1:8000 --verbose

# The --dirty flag makes mkdocs not regenerate everything when change is detected but also seems to
# break references.
serve-dirty: .env/create
	poetry run mkdocs serve --dirty -a 127.0.0.1:8000

# Build the documentation website.
site: .env/create $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs build --clean
	rm -Rf site/overrides

upload-docs: .env/create $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs gh-deploy

# Check that links in the documentation are valid. Requires the lychee tool.
linkcheck: site
	lychee "site/**/*.html"

# Start the trubot slack app.
trubot:
	poetry run python -u examples/trubot/trubot.py)

# Run a test with the optional flag set, meaning @optional_test decorated tests
# are run.
required-env:
	poetry install --only main,tests --verbose

optional-env:
	poetry install --sync --verbose

# Generic target to run any test with given environment
test-%: required-env
	make test-$*

test-%-optional: optional-env
	make test-$*

# Run the unit tests, those in the tests/unit. They are run in the CI pipeline frequently.
test-unit:
	poetry run pytest --rootdir=tests/unit .

test-lens:
	poetry run pytest --rootdir=tests/unit/test_lens .

test-feedback:
	poetry run pytest --rootdir=tests/unit/test_feedback .

test-tru-basic-app:
	poetry run pytest --rootdir=tests/unit/test_tru_basic_app .

test-tru-custom:
	poetry run pytest --rootdir=tests/unit/test_tru_custom .

# Run the static unit tests only, those in the static subfolder. They are run
# for every tested python version while those outside of static are run only for
# the latest (supported) python version.
test-static:
	poetry run pytest --rootdir=tests/unit/static/test_static .

# Tests in the e2e folder make use of possibly costly endpoints. They
# are part of only the less frequently run release tests.

test-e2e:
	poetry run pytest --rootdir=tests/e2e .

test-tru:
	poetry run pytest --rootdir=tests/e2e/test_tru .

test-tru-chain:
	poetry run pytest --rootdir=tests/e2e/test_tru_chain .

test-tru-llama:
	poetry run pytest --rootdir=tests/e2e/test_tru_llama .

test-providers:
	poetry run pytest --rootdir=tests/e2e/test_providers .

test-endpoints:
	poetry run pytest --rootdir=tests/e2e/test_endpoints .

# Database integration tests for various database types supported by sqlalchemy.
# While those don't use costly endpoints, they may be more computation intensive.

.env/create/db:
	poetry install --only main,tests,db-tests --sync --verbose

test-database: .env/create/db
	docker compose --file docker/test-database.yaml up --quiet-pull --detach --wait --wait-timeout 30
	poetry run pytest tests/integration/test_database.py
	docker compose --file docker/test-database.yaml down

# These tests all operate on local file databases and don't require docker.
test-database-specification: .env/create/db
	poetry run pytest tests/integration/test_database.py::TestDBSpecifications

# The next 3 database migration/versioning tests:
test-database-versioning: test-database-v2migration test-database-legacy-migration test-database-future

# Test migrating a latest legacy sqlite database to sqlalchemy database.
test-database-v2migration: .env/create/db
	poetry run pytest tests/integration/test_database.py::TestDbV2Migration::test_migrate_legacy_sqlite_file

# Test migrating non-latest legacy databases to sqlaclhemy database.
test-database-legacy-migration: .env/create/db
	poetry run pytest tests/integration/test_database.py::TestDbV2Migration::test_migrate_legacy_legacy_sqlite_file

# Test handling of a db that is newer than expected.
test-database-future: .env/create/db
	poetry run pytest tests/integration/test_database.py::TestDbV2Migration::test_future_db

# Release Steps:

## Step: Clean repo:
clean:
	git clean -fxd

## Step: Packages trulens into .whl file
build:
	poetry build

## Step: Uploads .whl file to PyPI, run make with:
# https://portal.azure.com/#@truera.com/asset/Microsoft_Azure_KeyVault/Secret/https://trulenspypi.vault.azure.net/secrets/trulens-pypi-api-token/abe0d9a3a5aa470e84c12335c9c04c72

## TOKEN=... make upload
upload:
	twine upload -u __token__ -p $(TOKEN) dist/*.whl

# Then follow steps from ../Makefile about updating the docs.
