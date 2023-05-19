SHELL := /bin/bash
CONDA_ENV := demo3
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)

slackbot:
	$(CONDA); (PYTHONPATH=. python -u trulens_evalchain/slackbot.py)
	
# 1>&2 > slackbot.log)

test:
	$(CONDA); python -m pytest -s test_tru_chain.py

format:
	$(CONDA); bash format.sh

lab:
	$(CONDA); jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

example_app:
	$(CONDA); PYTHONPATH=. streamlit run trulens_evalchain/Example_Application.py

example_trubot:
	$(CONDA); PYTHONPATH=. streamlit run trulens_evalchain/Example_TruBot.py

leaderboard:
	$(CONDA); PYTHONPATH=. streamlit run trulens_evalchain/Leaderboard.py
