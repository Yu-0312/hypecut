.PHONY: install dev test lint fmt serve docker clean sample

install:
	pip install -e ".[web]"

dev:
	pip install -e ".[web,dev]"

test:
	pytest -p no:warnings

lint:
	ruff check src tests && ruff format --check src tests

fmt:
	ruff check --fix src tests && ruff format src tests

serve:
	hypecut serve --reload

docker:
	docker compose up --build

sample:
	@python scripts/make_sample.py /tmp/hypecut_sample.mp4
	@hypecut cut /tmp/hypecut_sample.mp4 -o /tmp/hypecut_reel.mp4 --target 30 --percentile 88

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache hypecut-data
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
