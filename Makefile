RUN = pixi run --no-install -q --no-progress -e dev

.PHONY: test list

ruff:
	$(RUN) ruff format .
	$(RUN) ruff check . --fix

mypy:
	$(RUN) mypy src/ tests/

lint: ruff mypy

TESTARGS?=tests/  # default argument for the make test target
test:
	# you can use
	# make test TESTARGS="-k mytest"
	# make test TESTARGS="-m \"unit\""
	$(RUN) pytest $(TESTARGS)

check: ruff mypy test

precommit:
	$(RUN) pre-commit run

precommit-all:
	$(RUN) pre-commit run --all-files

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(firstword $(MAKEFILE_LIST)) : 2>/dev/null \
	| awk -v RS= -F: '/(^|\n)# Files(\n|$$)/,/(^|\n)# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' \
	| sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'
