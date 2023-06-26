SRC_DIR := cusprec
SRC_EXT := py
LOG_DIR := logs

.PHONY: clean
clean:
	rm -rf $(LOG_DIR)/*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

.PHONY: format
lint:
	flake8 $(SRC_DIR) main.py --ignore=F841,W503,F401,D107

.PHONY: format
format:
	isort $(SRC_DIR) main.py
	black --line-length 79 $(SRC_DIR) main.py
