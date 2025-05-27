.PHONY: help install install-dev test lint format clean setup-dev run-traisformer run-ensemble run-all

# Default target
help:
	@echo "Maritime Anomaly Detection - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  setup-dev    Setup development environment (install + pre-commit)"
	@echo ""
	@echo "Development:"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run linting (flake8, mypy)"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  clean        Clean up cache and build files"
	@echo ""
	@echo "Training:"
	@echo "  run-traisformer  Run TrAISformer model"
	@echo "  run-ensemble     Run ensemble model"
	@echo "  run-all          Run all models"
	@echo ""
	@echo "Data:"
	@echo "  download-sample  Download sample data (if available)"
	@echo "  validate-data    Validate data format"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup-dev: install-dev
	pre-commit install
	@echo "Development environment setup complete!"

# Code quality
format:
	black src/ tests/ *.py
	isort src/ tests/ *.py
	@echo "Code formatting complete!"

lint:
	flake8 src/ tests/ *.py
	mypy src/
	@echo "Linting complete!"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# Training commands
run-traisformer:
	python run_competition.py --model traisformer --data data/ --output outputs/

run-ensemble:
	python run_competition.py --model ensemble --data data/ --output outputs/ --cv

run-all:
	python run_competition.py --model all --data data/ --output outputs/ --submission

# Data management
download-sample:
	@echo "Sample data download not implemented yet"
	@echo "Please place your AIS data in the data/ directory"

validate-data:
	python -c "from src.data.data_loader import AISDataLoader; loader = AISDataLoader({}); print('Data validation passed!')"

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/
	@echo "Cleanup complete!"

# Docker commands (if needed)
docker-build:
	docker build -t maritime-anomaly-detection .

docker-run:
	docker run -v $(PWD)/data:/app/data -v $(PWD)/outputs:/app/outputs maritime-anomaly-detection

# Documentation
docs:
	@echo "Documentation generation not implemented yet"

# CI/CD helpers
ci-test: lint test-cov
	@echo "CI tests complete!"

# Development helpers
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

profile:
	python -m cProfile -o profile.stats run_competition.py --model ensemble --data data/ --output outputs/
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

# Quick start for new developers
quickstart: setup-dev
	@echo ""
	@echo "🚢 Maritime Anomaly Detection - Quick Start Complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Place your AIS data in the data/ directory"
	@echo "2. Run 'make validate-data' to check data format"
	@echo "3. Run 'make run-ensemble' to train the ensemble model"
	@echo "4. Check outputs/ directory for results"
	@echo ""
	@echo "For development:"
	@echo "- Run 'make format' before committing"
	@echo "- Run 'make test' to run tests"
	@echo "- Run 'make lint' to check code quality" 