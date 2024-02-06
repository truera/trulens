SHELL := /bin/bash
CONDA_ENV := py311_trulens
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)

PYENV:=PYTHONPATH=$(PWD)

trubot:
	$(CONDA); ($(PYENV) python -u examples/trubot/trubot.py)

test:
	$(CONDA); python -m unittest discover tests.unit

# Tests for various database types supported by sqlaclhemy.
test-database:
	$(CONDA); pip install psycopg2-binary pymysql cryptography
	docker compose --file docker/test-database.yaml up --quiet-pull --detach --wait --wait-timeout 30
	$(CONDA); python -m unittest discover tests.integration.test_database
	docker compose --file docker/test-database.yaml down

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

test-tru:
	$(CONDA); python -m unittest tests.unit.test_tru -k test_run_feedback

test-lens:
	$(CONDA); python -m unittest tests.unit.test_lens

test-feedback:
	$(CONDA); python -m unittest tests.unit.test_feedback

test-providers:
	$(CONDA); python -m unittest tests.unit.test_providers

test-tru-chain:
	$(CONDA); python -m unittest tests.unit.test_tru_chain

test-tru-llama:
	$(CONDA); python -m unittest tests.unit.test_tru_llama

test-tru-basic-app:
	$(CONDA); python -m unittest tests.unit.test_tru_basic_app

test-tru-custom:
	$(CONDA); python -m unittest tests.unit.test_tru_custom


format:
	$(CONDA); cd ..; bash format.sh --eval

lab:
	$(CONDA); jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

example_app:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Example_Application.py

example_trubot:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Example_TruBot.py

leaderboard:
	$(CONDA); $(PYENV) streamlit run trulens_eval/Leaderboard.py

react:
	$(CONDA); \
		npm i --prefix trulens_eval/react_components/record_viewer; \
		npm run --prefix trulens_eval/react_components/record_viewer build
