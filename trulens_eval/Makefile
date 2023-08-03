SHELL := /bin/bash
CONDA_ENV := py38_trulens
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate $(CONDA_ENV)

trubot:
	$(CONDA); (PYTHONPATH=. python -u examples/trubot/trubot.py)

test:
	$(CONDA); python -m unittest discover tests.unit

test-tru-chain:
	$(CONDA); python -m unittest tests.unit.test_tru_chain

test-tru-llama:
	$(CONDA); python -m unittest tests.unit.test_tru_llama

format:
	$(CONDA); bash format.sh

lab:
	$(CONDA); jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

example_app:
	$(CONDA); PYTHONPATH=. streamlit run trulens_eval/Example_Application.py

example_trubot:
	$(CONDA); PYTHONPATH=. streamlit run trulens_eval/Example_TruBot.py

leaderboard:
	$(CONDA); PYTHONPATH=. streamlit run trulens_eval/Leaderboard.py

react:
	$(CONDA); \
		npm i --prefix trulens_eval/react_components/record_viewer; \
		npm run --prefix trulens_eval/react_components/record_viewer build
