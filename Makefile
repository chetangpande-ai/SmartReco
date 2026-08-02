.DEFAULT_GOAL := help
.PHONY: help install seed run test lint fmt eval migrate reset digest docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies from the lockfile
	uv sync --all-extras

seed:  ## Create the demo catalogue and users
	uv run python -m app.seed

reset:  ## Wipe everything and reseed
	uv run python -m app.seed --reset

run:  ## Start the dev server on :8000
	uv run uvicorn app.main:app --reload --port 8000

test:  ## Run the test suite (offline, no API key needed)
	uv run pytest tests/ -q

cov:  ## Run tests with a coverage report
	uv run pytest tests/ -q --cov=app --cov-report=term-missing

lint:  ## Check formatting and lint rules
	uv run ruff check app tests scripts

fmt:  ## Apply safe lint fixes
	uv run ruff check app tests scripts --fix

eval:  ## Measure retrieval quality against the probe set
	uv run python scripts/eval_retrieval.py

sweep:  ## Tune the retrieval relevance floor
	uv run python scripts/eval_retrieval.py --sweep

migrate:  ## Apply database migrations
	uv run alembic upgrade head

digest:  ## Run the daily digest job once, right now
	uv run python -c "from app.db import init_db; init_db(); from app.services.digest import send_daily_digests; print(send_daily_digests())"

docker:  ## Build and run the container
	docker compose up --build

clean:  ## Remove local data, caches and the virtualenv
	rm -rf data .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
