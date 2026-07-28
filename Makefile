.PHONY: install dev-install test test-cov lint format run ui clean

install:
	pip install -r requirements.txt

dev-install: install
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=universal_text2sql --cov-report=term-missing

lint:
	ruff check .

format:
	ruff format .
	ruff check . --fix

run:
	python main.py

ui:
	streamlit run app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist .coverage
