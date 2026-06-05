.PHONY: download run-ecg5000 run-forda run-wafer test lint typecheck check

download:
	uv run python scripts/download_data.py

run-ecg5000:
	uv run python scripts/run_ecg5000.py --config configs/ecg5000.yaml

run-forda:
	uv run python scripts/run_forda.py --config configs/forda.yaml

run-wafer:
	uv run python scripts/run_ecg5000.py --config configs/wafer.yaml

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ scripts/ tests/
	uv run ruff format --check src/ scripts/ tests/

typecheck:
	uv run mypy src/anomdet/

check: lint typecheck test
