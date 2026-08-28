.PHONY: help test run docker-build docker-run docker-test docker-dashboard clean

help:
	@echo "Available commands:"
	@echo "  make test             : Run full unit test suite"
	@echo "  make run              : Run full multi-asset, multi-timeframe backtest experiment"
	@echo "  make docker-build     : Build Docker container image"
	@echo "  make docker-run       : Run experiments inside Docker container"
	@echo "  make docker-test      : Run tests inside Docker container"
	@echo "  make docker-dashboard : Start web server on port 8088 to view HTML report"
	@echo "  make clean            : Remove cached files and test artifacts"

test:
	python3 -m unittest discover -s tests -p "test_*.py" -v

run:
	python3 run_experiments.py --vehicle all --timeframe all --output-report reports/experiment_summary.html

docker-build:
	docker compose build

docker-run:
	docker compose run --rm experiments

docker-test:
	docker compose run --rm test

docker-dashboard:
	docker compose up dashboard

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

