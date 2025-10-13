# Fixtures that don't really fit in one of the other files.
import pytest
import os
from loguru import logger
from refractor.framework import PythonFpLogger, FpLogger
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import io
import sys
import numpy as np
import numpy.testing as npt


@contextmanager
def all_output_disabled():
    """Suppress stdout, stderr, and logging, useful for some of the noisy output we
    get running muses-py code."""
    try:
        logger.remove()
        with redirect_stdout(io.StringIO()):
            with redirect_stderr(io.StringIO()):
                yield
    finally:
        logger.add(sys.stderr)


def struct_compare(s1, s2, skip_list=None, verbose=False):
    """Compare two structures/dicts to make sure they are the same."""
    if skip_list is None:
        skip_list = []
    for k in s1.keys():
        if k in skip_list:
            if verbose:
                print(f"Skipping {k}")
            continue
        if verbose:
            print(k)
        if isinstance(s1[k], np.ndarray) and np.can_cast(s1[k], np.float64):
            npt.assert_allclose(s1[k], s2[k])
        elif isinstance(s1[k], np.ndarray):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]


@pytest.fixture(scope="function")
def isolated_dir(tmpdir):
    """This is a fixture that creates a temporary directory, and uses this
    while running a unit tests. Useful for tests that write out a test file
    and then try to read it.

    This fixture changes into the temporary directory, and at the end of
    the test it changes back to the current directory.

    Note that this uses the fixture tmpdir, which keeps around the last few
    temporary directories (cleaning up after a fixed number are generated).
    So if a test fails, you can look at the output at the location of tmpdir,
    e.g. /tmp/pytest-of-smyth
    """
    curdir = os.getcwd()
    try:
        tmpdir.chdir()
        yield curdir
    finally:
        os.chdir(curdir)


@pytest.fixture(scope="function")
def python_fp_logger(tmpdir):
    """Use PythonFpLogger to put the C++ logging stuff into the python logger."""
    PythonFpLogger.turn_on_logger(logger)
    yield
    PythonFpLogger.turn_off_logger()
    FpLogger.turn_on_logger()
