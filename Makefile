.PHONY: build-docs test lint fmt check release

build-docs:
	rm -rf docs/_build
	uv run --group docs sphinx-build -W -b html docs docs/_build

test:
	uv run pytest --cov=formative --cov-report=term-missing --cov-fail-under=80

lint:
	uv run ruff check --exclude notebooks .

fmt:
	uv run ruff format --exclude notebooks .

check: lint
	uv run ruff format --check --exclude notebooks .
	uv run pytest --cov=formative --cov-report=term-missing --cov-fail-under=80

release:  # usage: make release BUMP=patch|minor|major
	@test -n "$(BUMP)" || { echo "Usage: make release BUMP=patch|minor|major"; exit 1; }
	uvx bump-my-version bump $(BUMP)
	$(MAKE) build-docs
	uv run python scripts/snapshot_docs.py
	git add site vercel.json
	git commit -m "Snapshot docs for $$(uv run python scripts/snapshot_docs.py --print-version)"
	git push --follow-tags
