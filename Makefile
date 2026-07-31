.PHONY: help install lint format typecheck test test-gpu check up down logs

VENV := ai-engine/.venv/bin

help:
	@grep -E '^[a-zA-Z-]+:' $(MAKEFILE_LIST) | sed 's/:.*//' | sort

install:
	cd ai-engine && python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -e '.[dev]'

install-runtime:
	cd ai-engine && .venv/bin/pip install -e '.[dev,runtime]'

install-gpu:
	cd ai-engine && .venv/bin/pip install -e '.[dev,runtime,gpu]'

lint:
	cd ai-engine && .venv/bin/ruff check . && .venv/bin/ruff format --check .

format:
	cd ai-engine && .venv/bin/ruff format . && .venv/bin/ruff check --fix .

typecheck:
	cd ai-engine && .venv/bin/mypy

test:
	cd ai-engine && .venv/bin/python -m pytest -v -m "not gpu"

test-gpu:
	cd ai-engine && .venv/bin/python -m pytest -v -m gpu

check: lint typecheck test

up:
	docker compose -f deploy/compose/docker-compose.core.yml --profile core up -d

down:
	docker compose -f deploy/compose/docker-compose.core.yml --profile core down

logs:
	docker compose -f deploy/compose/docker-compose.core.yml logs -f
