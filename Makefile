SHELL := /bin/bash
CONDA_ENV := demo3
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)

slackbot:
	$(CONDA); (python -u slackbot.py 1>&2 > slackbot.log)

test:
	$(CONDA); python -m pytest -s test_tru_chain.py

format:
	$(CONDA); bash format.sh
