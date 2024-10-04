# Make targets useful for developing TruLens. How to use Makefiles:
# https://opensource.com/article/18/8/what-how-makefile . Please make sure this
# file runs on gmake:
# - Shell statements cannot span more than 1 line, see "lock" for example on
#   how newline is escaped to put the for loop on the same logical line.

SHELL := /bin/bash
REPO_ROOT := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTEST := poetry run pytest --rootdir=. -s
POETRY_DIRS := $(shell find . \
	-not -path "./dist/*" \
	-maxdepth 4 \
	-name "*pyproject.toml" \
	-exec dirname {} \;)
LAST_TRULENS_EVAL_COMMIT := 4cadb05 # commit that includes the last pre-namespace trulens_eval package

# Global setting: execute all commands of a target in a single shell session.
# Note for MAC OS, the default make is too old to support this. "brew install
# make" to get a newer version though it is called "gmake".
.ONESHELL:

# Create the poetry env for building website, docs, formatting, etc.
env:
	poetry install

env-%:
	poetry install --with $*

env-tests:
	poetry run pip install \
		pytest \
		nbconvert \
		nbformat \
		pytest-subtests \
		pytest-azurepipelines \
		ruff \
		pre-commit \
		pytest-cov \
		jsondiff

env-tests-required:
	poetry install --only required
	make env-tests

env-tests-optional: env env-tests
	poetry run pip install \
		llama-index-embeddings-huggingface \
		llama-index-embeddings-openai \
		langchain-openai \
		unstructured \
		chromadb \

env-tests-db: env-tests
	poetry run pip install \
		cryptography \
		psycopg2-binary \
		pymysql

env-tests-notebook: env-tests env-tests-optional
	poetry run pip install \
		faiss-cpu \
		ipytree \
		llama-index-readers-web

# Lock the poetry dependencies for all the subprojects.
lock: $(POETRY_DIRS)
	for dir in $(POETRY_DIRS); do \
		echo "Creating lockfile for $$dir/pyproject.toml"; \
		poetry lock -C $$dir; \
	done

# Run the ruff linter.
lint: env
	poetry run ruff check --fix

# Run the ruff formatter.
format: env
	poetry run ruff format

precommit-hooks:
	poetry run pre-commit install

run-precommit:
	poetry run pre-commit run --all-files --show-diff-on-failure

# Start a jupyter lab instance.
lab: env
	poetry run jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

# Build the documentation website.
docs: env-docs $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs build --clean
	rm -Rf site/overrides

# Serve the documentation website.
docs-serve: env-docs
	poetry run mkdocs serve -a 127.0.0.1:8000

# Serve the documentation website.
docs-serve-debug: env-docs
	poetry run mkdocs serve -a 127.0.0.1:8000 --verbose

# The --dirty flag makes mkdocs not regenerate everything when change is
# detected but also seems to break references.
docs-serve-dirty: env-docs
	poetry run mkdocs serve --dirty -a 127.0.0.1:8000

docs-upload: env-docs $(shell find docs -type f) mkdocs.yml
	poetry run mkdocs gh-deploy

# Check that links in the documentation are valid. Requires the lychee tool.
docs-linkcheck: site
	lychee "site/**/*.html"

# Start the trubot slack app.
trubot:
	poetry run python -u examples/trubot/trubot.py

# Spellchecking
codespell:
	poetry run codespell -L tru * --skip "*.svg,*.js,*.yaml,*.jsonl,*.lock,*.css.map"

# Generates a coverage report.
coverage:
	ALLOW_OPTIONALS=true poetry run pytest --rootdir=. tests/* --cov src --cov-report html

# Run the static unit tests only, those in the static subfolder. They are run
# for every tested python version while those outside of static are run only for
# the latest (supported) python version.

test-static:
	$(PYTEST) tests/unit/static/test_static.py

# Tests in the e2e folder make use of possibly costly endpoints. They
# are part of only the less frequently run release tests.

# API tests.
test-api: env-tests
	TEST_OPTIONAL=1 $(PYTEST) tests/unit/static/test_api.py
write-api: env-tests
	TEST_OPTIONAL=1 WRITE_GOLDEN=1 $(PYTEST) tests/unit/static/test_api.py || true
test-write-api:
	@echo "The 'test-write-api' target has been renamed to 'write-api'."
	@false

# Deprecation tests.
## Test for trulens_eval exported names.
test-deprecation:
	TEST_OPTIONAL=1 $(PYTEST) tests/unit/static/test_deprecation.py
## Test for trulens_eval notebooks.
.PHONY: _trulens_eval
_trulens_eval:
	git worktree add -f _trulens_eval --no-checkout --detach
	git --work-tree=_trulens_eval checkout $(LAST_TRULENS_EVAL_COMMIT) -- \
		trulens_eval/tests/docs_notebooks/notebooks_to_test \
		trulens_eval/README.md \
		trulens_eval/examples \
		docs/trulens_eval/tracking
	git worktree prune
test-legacy-notebooks: _trulens_eval
	TEST_OPTIONAL=1 $(PYTEST) -s tests/legacy/test_trulens_eval_notebooks.py

# Dummy and serial e2e tests do not involve any costly requests.
test-golden-%: tests/e2e/test_%.py # has golden file
	$(PYTEST) tests/e2e/test_$*.py
test-golden: test-golden-dummy test-golden-serial
write-golden-%: tests/e2e/test_%.py
	WRITE_GOLDEN=1 $(PYTEST) tests/e2e/test_$*.py || true
write-golden: write-golden-dummy write-golden-serial

# Run the tests for a specific file.
test-file-%: tests/e2e/%
	$(PYTEST) tests/e2e/$*
test-file-%: tests/unit/%
	$(PYTEST) tests/unit/$*
test-file-%: tests/unit/static/%
	$(PYTEST) tests/unit/static/$*

# Run the tests for a specific file with the optional flag on.
test-optional-file-%: tests/e2e/%
	TEST_OPTIONAL=true $(PYTEST) tests/e2e/$*
test-optional-file-%: tests/unit/%
	TEST_OPTIONAL=true $(PYTEST) tests/unit/$*
test-optional-file-%: tests/unit/static/%
	TEST_OPTIONAL=true $(PYTEST) tests/unit/static/$*

# Runs required tests
test-%-required: env-tests-required
	make test-$*

# Runs required tests, but allows optional dependencies to be installed.
test-%-allow-optional: env
	ALLOW_OPTIONALS=true make test-$*

# Requires the full optional environment to be set up.
test-%-optional: env-tests-optional
	TEST_OPTIONAL=true make test-$*

# Run the unit tests, those in the tests/unit. They are run in the CI pipeline
# frequently.
test-unit:
	poetry run pytest --rootdir=. tests/unit/*
# Tests in the e2e folder make use of possibly costly endpoints. They
# are part of only the less frequently run release tests.
test-e2e:
	poetry run pytest --rootdir=. tests/e2e/*

# Runs the notebook test
test-notebook:
	poetry run pytest --rootdir=. tests/docs_notebooks/*

install-wheels:
	pip install dist/*/*.whl

# Release Steps:
## Step: Clean repo:
clean:
	git clean --dry-run -fxd
	@read -p "Do you wish to remove these files? (y/N)" -n 1 -r
	echo
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		git clean -fxd; \
	fi;

## Step: Build wheels
build: $(POETRY_DIRS)
	for dir in $(POETRY_DIRS); do \
		echo "Building $$dir"; \
		pushd $$dir; \
		if [[ "$$dir" == "." ]]; then \
			pkg_name=trulens; \
		else \
			pkg_path=$${dir#./src/}; \
			pkg_name=trulens-$${pkg_path//\//-}; \
		fi; \
		echo $$pkg_name; \
		poetry build -o $(REPO_ROOT)/dist/$$pkg_name/; \
		rm -rf .venv; \
		popd; \
	done

## Step: Build zip files to upload to Snowflake staging
zip-wheels:
	poetry run ./zip_wheels.sh


# Usage: make bump-version-patch
bump-version-%: $(POETRY_DIRS)
	for dir in $(POETRY_DIRS); do \
		echo "Updating $$dir version"; \
		pushd $$dir; \
		poetry version $*; \
		popd; \
	done

## Step: Upload wheels to pypi
# Usage: TOKEN=... make upload-trulens-instrument-langchain
# In all cases, we need to clean, build, zip-wheels, then build again. The reason is because we want the final build to have the zipped wheels.
upload-%: clean build
	make zip-wheels \
		&& make build \
		&& poetry run twine upload -u __token__ -p $(TOKEN) dist/$*/*

upload-all: clean build
	make zip-wheels \
		&& make build \
		&& poetry run twine upload --skip-existing -u __token__ -p $(TOKEN) dist/**/*.whl \
		&& poetry run twine upload --skip-existing -u __token__ -p $(TOKEN) dist/**/*.tar.gz

upload-testpypi-%: clean build
	make zip-wheels \
		&& make build \
		&& poetry run twine upload -r testpypi -u __token__ -p $(TOKEN) dist/$*/*

upload-testpypi-all: clean build
	make zip-wheels \
		&& make build \
		&& poetry run twine upload -r testpypi --skip-existing -u __token__ -p $(TOKEN) dist/**/*.whl \
		&& poetry run twine upload -r testpypi --skip-existing -u __token__ -p $(TOKEN) dist/**/*.tar.gz
