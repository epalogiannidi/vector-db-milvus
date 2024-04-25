## MAKEFILE

.DEFAULT: help

## HELP

.PHONY: help
help:
	@echo "    black"
	@echo "        Format code using black, the Python code formatter"
	@echo "    lint"
	@echo "        Check source code with flake8"
	@echo "    isort"
	@echo "        Sort the imports"

## CODE STYLE RELATED
.PHONY: black
black:
	# run black code formatter
	black *.py

.PHONY: isort
isort:
	isort *.py

.PHONY: lint
lint:
	# run flake linter
	flake8 --max-line-length 120 --ignore E203,E402,W503 *.py

.PHONY: mypy
mypy:
	# run the mypy static typing checker
	mypy --config-file ./mypi.ini *.py
	rm -rf ./mypy_cache
