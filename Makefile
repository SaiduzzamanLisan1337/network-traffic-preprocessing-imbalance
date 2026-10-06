.PHONY: install test run lint clean

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

run:
	python -m network_imbalance.experiment --config configs/experiment.yaml

lint:
	ruff check src tests scripts

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f results/tables/*.csv results/tables/*.json results/figures/*.png
