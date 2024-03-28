# Make targets useful for developing TruLens-Explain.
# How to use Makefiles: https://opensource.com/article/18/8/what-how-makefile .

# Run target's commands in the same shell.
.ONESHELL:

SHELL:=bash
ACTIVATE:=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ENV:=python38
TEST_ENV:=PYTHONPATH=. CUDA_VISIBLE_DEVICES=

format:
	$(ACTIVATE) $(CONDA_ENV); bash format.sh

test_%:
	conda env create --name $(CONDA_ENV)_$* --file tools/conda_$(CONDA_ENV).yaml
	$(ACTIVATE) $(CONDA_ENV)_$*
	pip install -r tests/$*/requirements*.txt
	pip install pytest
	$(TEST_ENV) python -m pytest tests/$*

test_keras:
test_pytorch:
test_tf:
test_tf_keras:
test_tf2:
test_tf2_non_eager:

test_notebooks:
	conda env create --name $(CONDA_ENV)_notebooks --file tools/conda_$(CONDA_ENV).yaml
	$(ACTIVATE) $(CONDA_ENV)_notebooks
	pip install -r tests/notebooks/requirements.txt
	pip install pytest
	pip install ipykernel
	python -m ipykernel install --user --name "$(CONDA_ENV)" --display-name "$(CONDA_ENV)"
	$(TEST_ENV) python -m pytest tests/notebooks

test_notebooks_latest:
	conda env create --name $(CONDA_ENV)_notebooks_latest --file tools/conda_$(CONDA_ENV).yaml
	$(ACTIVATE) $(CONDA_ENV)_notebooks_latest
	pip install -r tests/notebooks/requirements_latest.txt
	pip install pytest
	pip install ipykernel
	python -m ipykernel install --user --name "$(CONDA_ENV)" --display-name "$(CONDA_ENV)"
	$(TEST_ENV) python -m pytest tests/notebooks

.PHONY: tests
tests:
	make test_keras
	make test_pytorch
	make test_tf
	make test_tf_keras
	make test_tf2
	make test_tf2_non_eager
	make test_notebooks
	make test_notebooks_latest

clean:
	conda env remove -n $(CONDA_ENV)_keras
	conda env remove -n $(CONDA_ENV)_pytorch
	conda env remove -n $(CONDA_ENV)_tf
	conda env remove -n $(CONDA_ENV)_tf_keras
	conda env remove -n $(CONDA_ENV)_tf2
	conda env remove -n $(CONDA_ENV)_tf2_non_eager
	conda env remove -n $(CONDA_ENV)_notebooks
	conda env remove -n $(CONDA_ENV)_notebooks_latest

.ONESHELL:
upload:
	read -s -p "Enter PyPi password: " temp
	# Removes previous build files
	git clean -fxd
	# Packages trulens into .whl file
	python setup.py bdist_wheel
	# Uploads .whl file to PyPi
	twine upload -u truera -p $$temp dist/*.whl
