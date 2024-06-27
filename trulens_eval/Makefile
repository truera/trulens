# Make targets useful for developing TruLens-Eval.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .

SHELL := /bin/bash
CONDA_ENV := py312
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA := $(CONDA_ACTIVATE) $(CONDA_ENV)

PYENV:=PYTHONPATH=$(PWD)

# Create conda enviornments with all of the supported python versions. The "req"
# ones with just the required packages and the "opt" ones with also the optional
# packages.
test-envs: \
	.conda/py-req-3.8 .conda/py-req-3.9 .conda/py-req-3.10 .conda/py-req-3.11 .conda/py-req-3.12 \
	.conda/py-opt-3.8 .conda/py-opt-3.9 .conda/py-opt-3.10 .conda/py-opt-3.11 .conda/py-opt-3.12

# Create a conda env for a particular python version with trulens-eval and just
# the required packages.
.conda/py-req-%:
	conda create -p .conda/py-req-$* python=$* -y
	$(CONDA_ACTIVATE) .conda/py-req-$*; \
		pip install -r trulens_eval/requirements.txt

# Create a conda env for a particular python version with trulens-eval and
# the required and optional packages.
.conda/py-opt-%:
	conda create -p .conda/py-opt-$* python=$* -y
	$(CONDA_ACTIVATE) .conda/py-opt-$*; \
		pip install -r trulens_eval/requirements.txt; \
		pip install -r trulens_eval/requirements.optional.txt

# Start the trubot slack app.
trubot:
	$(CONDA); ($(PYENV) python -u examples/trubot/trubot.py)

# Run a test with the optional flag set, meaning @optional_test decorated tests
# are run.
test-%-optional:
	TEST_OPTIONAL=1 make test-$*

# Run the unit tests, those in the tests/unit. They are run in the CI pipeline frequently.
test-unit:
	$(CONDA); python -m unittest discover tests.unit

test-lens:
	$(CONDA); python -m unittest tests.unit.test_lens

test-feedback:
	$(CONDA); python -m unittest tests.unit.test_feedback

test-tru-basic-app:
	$(CONDA); python -m unittest tests.unit.test_tru_basic_app

test-tru-custom:
	$(CONDA); python -m unittest tests.unit.test_tru_custom

# Run the static unit tests only, those in the static subfolder. They are run
# for every tested python version while those outside of static are run only for
# the latest (supported) python version.
test-static:
	$(CONDA); python -m unittest tests.unit.static.test_static

# Tests in the e2e folder make use of possibly costly endpoints. They
# are part of only the less frequently run release tests.

# Dummy and serial e2e tests do not involve any costly requests.
test-dummy: # has golden file
	$(CONDA); python -m unittest tests.e2e.test_dummy
test-serial: # has golden file
	$(CONDA); python -m unittest tests.e2e.test_serial
test-golden: test-dummy test-serial
test-write-golden: test-write-golden-dummy test-write-golden-serial
test-write-golden-%:
	$(CONDA); WRITE_GOLDEN=1 python -m unittest tests.e2e.test_$* || true

test-e2e:
	$(CONDA); python -m unittest discover tests.e2e

test-tru:
	$(CONDA); python -m unittest tests.e2e.test_tru

test-tru-chain:
	$(CONDA); python -m unittest tests.e2e.test_tru_chain

test-tru-llama:
	$(CONDA); python -m unittest tests.e2e.test_tru_llama

test-providers:
	$(CONDA); python -m unittest tests.e2e.test_providers

test-endpoints:
	$(CONDA); python -m unittest tests.e2e.test_endpoints

# Database integration tests for various database types supported by sqlaclhemy.
# While those don't use costly endpoints, they may be more computation intensive.
test-database:
	$(CONDA); pip install psycopg2-binary pymysql cryptography
	docker compose --file docker/test-database.yaml up --quiet-pull --detach --wait --wait-timeout 30
	$(CONDA); python -m unittest discover tests.integration.test_database
	docker compose --file docker/test-database.yaml down

# These tests all operate on local file databases and don't require docker.
test-database-specification:
	$(CONDA); python -m unittest discover tests.integration.test_database -k TestDBSpecifications

# The next 3 database migration/versioning tests:
test-database-versioning: test-database-v2migration test-database-legacy-migration test-database-future

# Test migrating a latest legacy sqlite database to sqlalchemy database.
test-database-v2migration:
	$(CONDA); python -m unittest \
		tests.integration.test_database.TestDbV2Migration.test_migrate_legacy_sqlite_file

# Test migrating non-latest legacy databases to sqlaclhemy database.
test-database-legacy-migration:
	$(CONDA); python -m unittest \
		tests.integration.test_database.TestDbV2Migration.test_migrate_legacy_legacy_sqlite_file

# Test handling of a db that is newer than expected.
test-database-future:
	$(CONDA); python -m unittest \
		tests.integration.test_database.TestDbV2Migration.test_future_db

# Run the code formatter and imports organizer.
format:
	$(CONDA); cd ..; bash format.sh --eval

# Start a jupyter lab server locally with the token being set to "deadbeef".
lab:
	$(CONDA); jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

example_app:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Example_Application.py

example_trubot:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Example_TruBot.py

# Start the dashboard.
leaderboard:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Leaderboard.py

# Rebuild the react components.
react:
	$(CONDA); \
		npm i --prefix trulens_eval/react_components/record_viewer; \
		npm run --prefix trulens_eval/react_components/record_viewer build

# Release Steps:

# Step: Clean repo:
clean:
	git clean -fxd

# Step: Packages trulens into .whl file 
build:
	python setup.py bdist_wheel

# Step: Uploads .whl file to PyPI, run make with:
# https://portal.azure.com/#@truera.com/asset/Microsoft_Azure_KeyVault/Secret/https://trulenspypi.vault.azure.net/secrets/trulens-pypi-api-token/abe0d9a3a5aa470e84c12335c9c04c72

# TOKEN=... make upload
upload:
	twine upload -u __token__ -p $(TOKEN) dist/*.whl

# Then follow steps from ../Makefile about updating the docs.
