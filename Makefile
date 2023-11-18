VENV_PATH=.venv
DIST_PATH=dist

clean:
	@rm -rf $(VENV_PATH)
	@rm -rf $(DIST_PATH)
	@rm -rf .pytest_cache

install:
	python3 -m venv $(VENV_PATH)
	python3 -m ensurepip --upgrade
	$(VENV_PATH)/bin/pip install poetry==1.2.0
	$(VENV_PATH)/bin/poetry --version
	$(VENV_PATH)/bin/poetry install

build:
	@$(VENV_PATH)/bin/poetry build

test:
	@$(VENV_PATH)/bin/pytest -rP

test-ci:
	@$(VENV_PATH)/bin/pytest

lint:
	@$(VENV_PATH)/bin/mypy neural_network tests
	@$(VENV_PATH)/bin/flake8 neural_network tests