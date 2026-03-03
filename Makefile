.PHONY: build-docs test lint fmt check

build-docs:
	uv run --group docs sphinx-build -b html docs docs/_build

test:
	uv run pytest --cov=formative --cov-report=term-missing --cov-fail-under=88

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

check: lint
	uv run ruff format --check .
	uv run pytest --cov=formative --cov-report=term-missing --cov-fail-under=88
