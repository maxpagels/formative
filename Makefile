.PHONY: build-docs test

build-docs:
	uv run --group docs sphinx-build -b html docs docs/_build

test:
	uv run pytest
