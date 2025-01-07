import warnings
from loguru import logger
import pytest
import os
import re

# ------------------------------------------
# Set up to warnings go to the logger
# ------------------------------------------
showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    # Swig has numerous warning messages that we can't do anything about.
    # There is a ticket for this (see https://github.com/swig/swig/issues/2881), and
    # this might get fixed in swig 4.4. But for now at least, these are present. Strip
    # these out, we can't do anything about these warnings and don't want to see them.
    if not re.search(r"has no __module__ attribute", str(message)):
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
# Based on markers, we skip tests
# ------------------------------------------


def pytest_addoption(parser):
    parser.addoption("--run-long", action="store_true", help="run long tests")
    parser.addoption(
        "--skip-old-py-retrieve-test",
        action="store_true",
        help="skip tests against old py-retrieve",
    )
    parser.addoption(
        "--run-capture",
        action="store_true",
        help="run tests used to capture input arguments",
    )
    parser.addoption(
        "--run-capture-initial",
        action="store_true",
        help="run tests used to capture input arguments, the initial version",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-long"):
        skip_long_test = pytest.mark.skip(reason="need --run-long option to run")
        for item in items:
            if "long_test" in item.keywords:
                item.add_marker(skip_long_test)
    if config.getoption("--skip-old-py-retrieve-test"):
        skip_old_py_retrieve_test = pytest.mark.skip(
            reason="--skip-old-py-retrieve-test option skips this test"
        )
        for item in items:
            if "old_py_retrieve_test" in item.keywords:
                item.add_marker(skip_old_py_retrieve_test)
    if not config.getoption("--run-capture"):
        skip_capture_test = pytest.mark.skip(
            reason="input test capture test. Need --run-capture option to run"
        )
        for item in items:
            if "capture_test" in item.keywords:
                item.add_marker(skip_capture_test)
    if not config.getoption("--run-capture-initial"):
        skip_capture_initial_test = pytest.mark.skip(
            reason="input test initial capture test. Need --run-initial-capture option to run"
        )
        for item in items:
            if "capture_initial_test" in item.keywords:
                item.add_marker(skip_capture_initial_test)


# ------------------------------------------
# Includes fixtures, made available to all tests.
# ------------------------------------------

pytest_plugins = [
    "fixtures.dir_fixture",
    "fixtures.misc_fixture",
    "fixtures.retrieval_step_fixture",
    # Fixtures only used in old_py_retrieve
    "fixtures.stand_alone_obs_fixture",
    "fixtures.rf_uip_fixture",
    "fixtures.muses_retrieval_step_fixture",
]
