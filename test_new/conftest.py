import warnings
from loguru import logger
import pytest
import os

# ------------------------------------------
# Set up to warnings go to the logger
# ------------------------------------------
showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    logger.warning(message)
    # showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning

# ------------------------------------------
# Various markers we use throughout the tests
# ------------------------------------------

# Short hand for marking as unconditional skipping. Good for tests we
# don't normally run, but might want to comment out for a specific debugging
# reason.
skip = pytest.mark.skip

# Marker for long tests. Only run with --run-long
long_test = pytest.mark.long_test

# Marker for capture tests. Only run with --run-capture
capture_test = pytest.mark.capture_test

# Marker for initial capture tests. Only run with --run-initial-capture
capture_initial_test = pytest.mark.capture_initial_test

# ------------------------------------------
# Fake creation date used in muses-py file creation, to make comparing
# easier since we don't get differences that are just the time of creation
# ------------------------------------------

os.environ["MUSES_FAKE_CREATION_DATE"] = "FAKE_DATE"

# ------------------------------------------
# Includes fixtures, made available to all tests.
# ------------------------------------------

pytest_plugins = ["fixtures.dir_fixture"]
