ACTIVATE:=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ENV:=python37

format:
	$(ACTIVATE) $(CONDA_ENV); bash format.sh
