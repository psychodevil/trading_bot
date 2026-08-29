.PHONY: help install setup test web portfolio benchmark data clean docker-build docker-up

help:
	@echo "QuantumAlpha Quantitative Trading Framework"
	@echo "--------------------------------------------"
	@echo "make setup       : Create isolated .venv and bootstrap dependencies"
	@echo "make test        : Run full automated unit & integration test suite"
	@echo "make web         : Start QuantumAlpha Flask Web Application (port 8088)"
	@echo "make portfolio   : Run $$100k Causal Walk-Forward Multi-Asset Simulation"
	@echo "make benchmark   : Run 62-Asset Global Market Alpha Benchmark"
	@echo "make data        : Download latest 1-year hourly market bars"
	@echo "make docker-up   : Launch containerized web service via Docker Compose"
	@echo "make clean       : Clean temporary files and bytecode caches"

setup:
	@chmod +x setup_env.sh && ./setup_env.sh

test:
	@python3 -m unittest discover tests

web:
	@python3 web/app.py

portfolio:
	@python3 scripts/run_portfolio.py

benchmark:
	@python3 scripts/run_benchmark.py

data:
	@python3 scripts/download_data.py

docker-build:
	@docker compose build

docker-up:
	@docker compose up

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[cod]" -delete
	@find . -type f -name "*.log" -delete
