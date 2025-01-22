# Just some basic command collected together, so I don't need to remember these

# We generally want to install in editable mode, so make default
install:
	pip install -e .

# Without editable mode
install2:
	pip install .

lint:
	ruff check python/refractor/muses python/refractor/omi python/refractor/tropomi

format:
	ruff format python/refractor/muses python/refractor/omi python/refractor/tropomi

# Note that PYTHONPATH is required, at least as of 24.3.1. The py.typed file needed to
# tell mypy that our modules have types doesn't get translated through using
# editable mode of pip. This is an issue going back at least a few years that hasn't
# been addressed - see https://github.com/python/mypy/issues/13392. We work around this
# just by giving a explicit path to our source code skipping going through the installed
# version.
mypy:
	PYTHONPATH=$(PWD)/python mypy python/refractor/muses python/refractor/omi python/refractor/tropomi
