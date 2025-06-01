.PHONY: test lint run setup

setup:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	flake8 app/

run:
	python3 main.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	rm -rf data/uploads/*
	rm -rf embeddings/*
	rm -rf chatlogs/*
	rm -rf app.log
