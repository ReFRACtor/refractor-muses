# Just some basic command collected together, so I don't need to
# remember these

# We generally want to install in editable mode, so make default
install:
	pip install -e .

# Without editable mode
install2:
	pip install .

# Generate __init__ files:

update-init:
# Don't normally run this, we have to manually add a number of # type:ignore messages
# for old code. Have a separate update-old-init for this
#	cd python/refractor/old_py_retrieve_wrapper && mkinit . -w --black --nomods --relative
	cd python/refractor/muses && mkinit . -w --black --nomods --relative
	cd python/refractor/muses_py_fm && mkinit . -w --black --nomods --relative
	cd python/refractor/omi && mkinit . -w --black --nomods --relative
#	cd python/refractor/osr_ml && mkinit . -w --black --nomods --relative
	cd python/refractor/tropomi && mkinit . -w --black --nomods --relative

update-old-init:
	@echo "Note you need to manually add '# type:ignore messages' to generated"
	@echo "__init__.py file. You can do "make mypy" and see what error pop up to"
	@echo "find this, or just look in the module files to see if it has # type: ignore"
	cd python/refractor/old_py_retrieve_wrapper && mkinit . -w --black --nomods --relative


# ------------------------------------------------------------------
# See notes in python/refractor/muses/README_developer.md about the
# use of linters and type checkers.
#
# It is *not* considered an error to fail these, however the output of
# the linter and type checker can useful. We try to get all the errors
# fixed just to reduce the noise in the output. You can also often
# just silence errors for things that aren't worth fixing. We aren't
# required to make the linter or type checker happy - just to write
# python code that works. But it is mildly useful to have everything
# cleanly passing so we can see the occasional real thing that the
# linter or type checker finds.
#
# Also it isn't a requirement to match the format that ruff or other
# formater want. But it is pretty easy just to run the tools to get
# constistent code, so we tend to do that.
# ------------------------------------------------------------------

lint:
	ruff check python/refractor tests

lint-fix:
	ruff check --fix python/refractor tests

format:
	ruff format python/refractor tests

# Note that PYTHONPATH is required, at least as of pip 24.3.1. The py.typed file needed to
# tell mypy that our modules have types doesn't get translated through using
# editable mode of pip. This is an issue going back at least a few years that hasn't
# been addressed - see https://github.com/python/mypy/issues/13392. We work around this
# just by giving a explicit path to our source code skipping going through the installed
# version. This is only for type checking, all our other tests/check use the pip installed
# version.
mypy-simpler:
	PYTHONPATH=$(PWD)/python mypy python/refractor

# This adds options to check for missing typing info. We usually want this
mypy:
	PYTHONPATH=$(PWD)/python mypy --disallow-untyped-defs python/refractor
