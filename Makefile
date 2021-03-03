SHELL=/bin/zsh
VENV_NAME?=ubi_vocab
CONDA_SOURCE=source $$(conda info --base)/etc/profile.d/conda.sh ;

help:
	@echo "make setup"
	@echo "    setup conda environment, use only once"
	@echo "make update"
	@echo "update environment with latest requirements"

setup:
	# Environment yml file currently broken, not sure why.
	conda env create -f environment.yml
# 	conda create -y -n $(VENV_NAME) python=3.7
	# make update
	
venv:
	$(CONDA_SOURCE) conda activate $(VENV_NAME)

update:
	make venv
	conda env update --file environment.yml

wheel:
	make venv
	python setup.py sdist bdist_wheel