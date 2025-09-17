.PHONY: clean clean-extras build check publish install test help grade10hz grade10hz-csv grade10hz-csv-out fmt

# -------------------------------------------------------------------
# Variables (override from CLI if needed)
#   make grade10hz SPEED_KPH=50
#   make grade10hz-csv CSV=path/to/points.csv OUT=/tmp/grade.csv
# -------------------------------------------------------------------
PYTHON ?= python
CONFIG ?= service/config/altitude_ign.yaml
SPEED_KPH ?= 36
LAT ?= 49.2052 49.1995 49.1918 49.1856 49.1789 49.1730
LON ?= 1.1705 1.1752 1.1820 1.1895 1.1960 1.2025
CSV ?= src/rs3_plugin_altitude_agpl/tests/data/points_tests.csv

help:
	@echo "Targets:"
	@echo "  clean              - remove build artifacts"
	@echo "  clean-extras       - also remove stray dir rs3_plugin_altitude_agpl and .DS_Store"
	@echo "  build              - build wheel and sdist"
	@echo "  check              - twine check on dist/*"
	@echo "  publish            - upload to PyPI (twine upload)"
	@echo "  install            - pip install -e ."
	@echo "  test               - run pytest -q"
	@echo "  grade10hz          - run 10 Hz grade from LAT/LON lists"
	@echo "  grade10hz-csv      - run 10 Hz grade from CSV file"
	@echo "  grade10hz-csv-out  - same as above but requires OUT=/path/to/out.csv"
	@echo "  fmt                - format sources with black (if available)"

clean:
	@rm -rf dist build ./*.egg-info src/*.egg-info src/*/*.egg-info || true
	@rm -rf __pycache__ src/**/__pycache__ .pytest_cache .mypy_cache .ruff_cache || true
	@rm -rf logs/* || true
	@rm -rf src/rs3_plugin_altitude_agpl/tests/data/bench_out || true
	@find . -name ".DS_Store" -delete || true

# also remove stray duplicate package dir, temporary macOS files
clean-extras:
	@rm -rf rs3_plugin_altitude_agpl || true
	@find . -name ".DS_Store" -delete || true

build: clean
	@$(PYTHON) -m build

check: build
	@twine check dist/*

publish: build
	@twine check dist/*
	@twine upload dist/*

install:
	@$(PYTHON) -m pip install -e .

test:
	@$(PYTHON) -m pytest -q

# -------------------------------------------------------------------
# Tools: grade_10hz
# -------------------------------------------------------------------
# Uses defaults above; override via make VAR=value
# Example:
#   make grade10hz SPEED_KPH=50
#   make grade10hz-csv CSV=my_points.csv SPEED_KPH=36
#   make grade10hz-csv-out CSV=my_points.csv OUT=/tmp/grade.csv

grade10hz:
	@$(PYTHON) -m rs3_plugin_altitude_agpl.tools.grade_10hz \
	  -c $(CONFIG) \
	  --lat $(LAT) \
	  --lon $(LON) \
	  --speed-kph $(SPEED_KPH)

grade10hz-csv:
	@$(PYTHON) -m rs3_plugin_altitude_agpl.tools.grade_10hz \
	  -c $(CONFIG) \
	  --csv $(CSV) \
	  --speed-kph $(SPEED_KPH)

grade10hz-csv-out:
	@if [ -z "$(OUT)" ]; then \
		echo "ERROR: provide OUT=/path/to/out.csv"; exit 2; \
	fi
	@$(PYTHON) -m rs3_plugin_altitude_agpl.tools.grade_10hz \
	  -c $(CONFIG) \
	  --csv $(CSV) \
	  --speed-kph $(SPEED_KPH) \
	  --out $(OUT)

# Format (optional if black is installed)
fmt:
	@black -q src || true
