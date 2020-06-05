.PHONY: setup

setup:
	test -d .venv || python3 -m venv .venv
	poetry install -vvv
