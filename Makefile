SHELL := /bin/bash
CONDA_ENV := demo3
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)

slackbot:
	$(CONDA); python slackbot.py

test:
	$(CONDA); python -m pytest -s test_tru_chain.py
