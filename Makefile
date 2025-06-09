.PHONY: test lint run setup

setup:
	pip install -r requirements.txt
	python3 setup_model.py

lint:
	flake8 ragify_docs/
	flake8 ragify_docs.py
	flake8 setup_model.py

run:
	python3 ragify_docs.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	find . -type d -name "*.egg" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".data" -exec rm -r {} +
