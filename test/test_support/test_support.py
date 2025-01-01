import os
import sys
import pytest
from refractor.muses import (
    RefractorUip,
    DictFilterMetadata,
    MusesRunDir,
    RetrievalStrategy,
    MusesTropomiObservation,
    MusesOmiObservation,
    MusesCrisObservation,
    MusesAirsObservation,
    MusesSpectralWindow,
)
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from refractor.framework import PythonFpLogger, FpLogger
import netCDF4 as ncdf
import numpy as np
import numpy.testing as npt
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from loguru import logger
import io
import warnings

# warnings to logger
showwarning_ = warnings.showwarning


def showwarning(message, *args, **kwargs):
    logger.warning(message)
    # showwarning_(message, *args, **kwargs)


warnings.showwarning = showwarning

if "REFRACTOR_TEST_DATA" in os.environ:
    test_base_path = os.environ["REFRACTOR_TEST_DATA"]
else:
    test_base_path = os.path.abspath(
        f"{os.path.dirname(__file__)}/../../../refractor_test_data"
    )

tropomi_test_in_dir = f"{test_base_path}/tropomi/in/sounding_1"
tropomi_test_in_dir2 = f"{test_base_path}/tropomi/in/sounding_2"
tropomi_test_in_dir3 = f"{test_base_path}/tropomi/in/sounding_3"
tropomi_band7_test_top = f"{test_base_path}/tropomi/in"
tropomi_band7_test_in_dir = f"{test_base_path}/tropomi_band7/in/sounding_1"
tropomi_band7_test_in_dir2 = f"{test_base_path}/tropomi_band7/in/sounding_2"
tropomi_band7_test_state_dir2 = f"{test_base_path}/tropomi_band7/in/sounding_2_state"
tropomi_band7_expected_results_dir = f"{test_base_path}/tropomi_band7/expected/"
tropomi_band7_swir_step_test_in_dir = (
    f"{test_base_path}/tropomi_band7/in/sounding_one_step"
)
joint_tropomi_test_in_dir = f"{test_base_path}/cris_tropomi/in/sounding_1"
joint_tropomi_test_expected_dir = f"{test_base_path}/cris_tropomi/expected/sounding_1"
joint_tropomi_test_refractor_expected_dir = (
    f"{test_base_path}/cris_tropomi/expected/sounding_1_refractor"
)
joint_tropomi_test_in_dir2 = f"{test_base_path}/cris_tropomi/in/sounding_2"
joint_tropomi_test_in_dir3 = f"{test_base_path}/cris_tropomi/in/sounding_3"
omi_test_in_dir = f"{test_base_path}/omi/in/sounding_1"
omi_test_expected_results_dir = f"{test_base_path}/omi/expected"
joint_omi_test_in_dir = f"{test_base_path}/airs_omi/in/sounding_1"
joint_omi_test_expected_dir = f"{test_base_path}/airs_omi/expected/sounding_1"
joint_omi_test_refractor_expected_dir = (
    f"{test_base_path}/airs_omi/expected/sounding_1_refractor"
)
airs_irk_test_in_dir = f"{test_base_path}/airs_omi/in/sounding_1_irk"
airs_irk_test_expected_dir = f"{test_base_path}/airs_omi/expected/sounding_1_irk"
tes_test_in_dir = f"{test_base_path}/tes/in/sounding_1"
tes_test_expected_dir = f"{test_base_path}/tes/expected/sounding_1"
oco2_test_in_dir = f"{test_base_path}/oco2/in/sounding_1"
oco2_test_expected_dir = f"{test_base_path}/oco2/expected/sounding_1"

if not os.path.exists(tropomi_test_in_dir):
    raise RuntimeError(
        f"ERROR: unit test data not found at {test_base_path}. Please clone the test data repo there."
    )

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

# Fake creation date used in muses-py file creation, to make comparing
# easier since we don't get differences that are just the time of creation
os.environ["MUSES_FAKE_CREATION_DATE"] = "FAKE_DATE"


@pytest.fixture(scope="function")
def osp_dir():
    """Location of OSP directory."""
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        raise pytest.skip(
            "test requires OSP directory set by through the MUSES_OSP_PATH environment variable"
        )
    return osp_path


@pytest.fixture(scope="function")
def josh_osp_dir():
    """Location of Josh's newer OSP directory. Eventually stuff will get merged into
    the real OSP directory, but for now keep this separate"""
    osp_path = os.environ.get(
        "MUSES_JOSH_OSP_PATH", "/tb/sandbox17/laughner/OSP-mine/OSP"
    )
    if osp_path is None or not os.path.exists(osp_path):
        raise pytest.skip(
            "test requires Josh's OSP directory set by through the MUSES_JOSH_OSP_PATH environment variable"
        )
    return osp_path


@pytest.fixture(scope="function")
def gmao_dir():
    """Location of GAMO directory."""
    gmao_path = os.environ.get("MUSES_GMAO_PATH", None)
    if gmao_path is None or not os.path.exists(gmao_path):
        raise pytest.skip(
            "test requires GMAO directory set by through the MUSES_GMAO_PATH environment variable"
        )
    return gmao_path


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


@pytest.fixture(scope="function")
def tropomi_band7_simple_ils_test_data():
    simple_results_file = os.path.join(
        tropomi_band7_expected_results_dir, "ils", "simple_ils_test.nc"
    )
    with ncdf.Dataset(simple_results_file) as ds:
        return {
            "hi_res_freq": ds["hi_res_freq"][:].filled(np.nan),
            "hi_res_spec": ds["hi_res_spectrum"][:].filled(np.nan),
            "convolved_spec": ds["convolved_spectrum"][:].filled(np.nan),
        }


@pytest.fixture(scope="function")
def omi_config_dir():
    """Returns configuration directory"""
    yield (
        os.path.abspath(
            os.path.dirname(__file__) + "/../../python/refractor/omi/config"
        )
        + "/"
    )


@pytest.fixture(scope="function")
def vlidort_cli():
    return os.environ.get(
        "MUSES_VLIDORT_CLI", "~/muses/muses-vlidort/build/release/vlidort_cli"
    )


@contextmanager
def all_output_disabled():
    """Suppress stdout, stderr, and logging, useful for some of the noisy output we
    get running muses-py code."""
    try:
        logger.remove()
        with redirect_stdout(io.StringIO()) as sout:
            with redirect_stderr(io.StringIO()) as serr:
                yield
    finally:
        logger.add(sys.stderr)


def load_step(rs, step_number, dir, include_ret_state=False):
    """Load in the state information and optional retrieval results for the given
    step, and jump to that step."""
    rs.load_state_info(
        f"{dir}/state_info_step_{step_number}.pkl",
        step_number,
        ret_state_file=f"{dir}/retrieval_state_step_{step_number}.json.gz"
        if include_ret_state
        else None,
    )


def set_up_run_to_location(dir, step_number, location, include_ret_state=True):
    """Set up directory and run the given step number to the given location."""
    osp_dir = os.environ.get("MUSES_OSP_PATH", None)
    gmao_dir = os.environ.get("MUSES_GMAO_PATH", None)
    vlidort_cli = os.environ.get(
        "MUSES_VLIDORT_CLI", "~/muses/muses-vlidort/build/release/vlidort_cli"
    )
    r = MusesRunDir(dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rstep, kwargs = run_step_to_location(
        rs, step_number, dir, location, include_ret_state=include_ret_state
    )
    return rs, rstep, kwargs


def run_step_to_location(rs, step_number, dir, location, include_ret_state=True):
    """Load in the given step, and run up to the location we notify at
    (e.g., "retrieval step"). Return the retrieval_strategy_step"""

    class CaptureRs:
        def __init__(self):
            self.retrieval_strategy_step = None

        def notify_update(
            self, retrieval_strategy, loc, retrieval_strategy_step=None, **kwargs
        ):
            if loc != location:
                return
            self.retrieval_strategy_step = retrieval_strategy_step
            self.kwargs = kwargs
            raise StopIteration()

    try:
        rcap = CaptureRs()
        rs.add_observer(rcap)
        load_step(rs, step_number, dir, include_ret_state=include_ret_state)
        try:
            rs.continue_retrieval(stop_after_step=step_number)
        except StopIteration:
            return rcap.retrieval_strategy_step, rcap.kwargs
    finally:
        rs.remove_observer(rcap)


@pytest.fixture(scope="function")
def joint_omi_step_8(isolated_dir):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_omi_test_in_dir, 8, "retrieval input"
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12(isolated_dir):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval input"
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


def struct_compare(s1, s2, skip_list=None, verbose=False):
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
