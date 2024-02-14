SHELL := /bin/bash
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create the conda env for building website, docs, formatting, etc.
.conda/docs:
	conda create python=3.11 --yes --prefix=.conda/docs
	$(CONDA) .conda/docs; \
	pip install -r trulens_eval/trulens_eval/requirements.txt; \
	pip install -r trulens_eval/trulens_eval/requirements.optional.txt; \
	pip install -r docs/docs_requirements.txt

# Run the code formatter.
format: .conda/docs
	$(CONDA) .conda/docs; bash format.sh

# Start a jupyer lab instance.
lab:
	$(CONDA) .conda/docs; jupyter lab --ip=0.0.0.0 --no-browser --ServerApp.token=deadbeef

# Serve the documentation website.
serve: site
	$(CONDA) .conda/docs; mkdocs serve -a 127.0.0.1:8000

# Build the documentation website.
site: .conda/docs $(shell find docs -type f) mkdocs.yml
	$(CONDA) .conda/docs; mkdocs build
	rm -Rf site/overrides

# Check that links in the documentation are valid. Requires the lychee tool.
linkcheck: site
	lychee "site/**/*.html"