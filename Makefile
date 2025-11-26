install:
	pip install --upgrade pip && \
		pip install -e ".[dev]"

editable-install:
	pip install --upgrade pip && \
		pip install -e ".[dev]" --config-settings editable_mode=strict

setup-ci:
	pip install pre-commit && \
	pre-commit install

black:
	pre-commit run black -a

interrogate:
	pre-commit run interrogate -a

lint:
	pre-commit run markdownlint -a && \
	pre-commit run check-added-large-files -a && \
	pre-commit run trailing-whitespace -a && \
	pre-commit run end-of-file-fixer -a && \
	pre-commit run check-yaml -a && \
	pre-commit run ruff -a

license:
	python tests/ci_tests/header_check.py --all-files

doctest:
	echo "Not implemented"

pytest:
	pip install -e ".[dev]" && \
	pytest tests/test_etl/
	pytest tests/test_examples/test_external_aerodynamics/
	pytest tests/test_examples/test_structural_mechanics/

coverage:
	echo "Not implemented"

all-ci: setup-ci black interrogate lint license install pytest doctest coverage
