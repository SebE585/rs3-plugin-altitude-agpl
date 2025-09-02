.PHONY: clean build check publish

clean:
	@rm -rf dist build ./*.egg-info src/*.egg-info src/*/*.egg-info || true

build: clean
	@python -m build

check: build
	@twine check dist/*

publish: build
	@twine check dist/*
	@twine upload dist/*
