.PHONY: venv install test

venv:
	cd python && uv venv

install:
	cd python && uv pip install -e ".[tests]"

test:
	cd python && uv run pytest
