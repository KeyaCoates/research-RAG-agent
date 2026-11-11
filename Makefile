.PHONY: install test eval lint
install: ; pip install -e ".[dev,api]"
test: ; pytest -q
eval: ; python -m ragassist.eval.run_eval --min-recall 0.6
lint: ; ruff check .
