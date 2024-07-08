# Make targets useful for developing TruLens.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .

SHELL := /bin/bash
CONDA := source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create the conda env for building website, docs, formatting, etc.
.conda/docs:
	conda create python=3.12 --yes --prefix=.conda/docs
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
serve: .conda/docs
	$(CONDA) .conda/docs; mkdocs serve -a 127.0.0.1:8000

# Serve the documentation website.
serve-debug: .conda/docs
	$(CONDA) .conda/docs; mkdocs serve -a 127.0.0.1:8000 --verbose

# The --dirty flag makes mkdocs not regenerate everything when change is detected but also seems to
# break references.
serve-dirty: .conda/docs
	$(CONDA) .conda/docs; mkdocs serve --dirty -a 127.0.0.1:8000

# Build the documentation website.
site: .conda/docs $(shell find docs -type f) mkdocs.yml
	$(CONDA) .conda/docs; mkdocs build --clean
	rm -Rf site/overrides

upload: .conda/docs $(shell find docs -type f) mkdocs.yml
	$(CONDA) .conda/docs; mkdocs gh-deploy

# Check that links in the documentation are valid. Requires the lychee tool.
linkcheck: site
	lychee "site/**/*.html"
