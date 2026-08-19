.PHONY: sync test unit integration regression e2e lint format typecheck build-showcase

sync:
	uv sync --extra dev

test:
	uv run pytest

unit:
	uv run pytest tests/unit

integration:
	uv run pytest tests/integration

regression:
	uv run pytest tests/regression

e2e:
	uv run pytest tests/e2e

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

build-showcase:
	uv run atc-benchmark build --config configs/showcase_v1.yaml
