.PHONY: help lint format type test check

help:
	@echo "lint    ruff check + ruff format --check"
	@echo "format  ruff format + ruff check --fix"
	@echo "type    mypy strict over src/"
	@echo "test    pytest"
	@echo "check   lint, type, and test"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

type:
	mypy src/

test:
	pytest -q

check: lint type test
