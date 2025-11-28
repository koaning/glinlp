.PHONY: install lint format serve schema test

install:
	uv sync

lint:
	uv run ruff check src

format:
	uv run ruff format src

serve:
	uv run glinlp serve examples/schema.example.yaml

schema:
	uv run glinlp validate examples/schema.example.yaml

test:
	uv run pytest
