.PHONY: install-dev test lint doctor

install-dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

doctor:
	python -m leopardi.cli doctor

