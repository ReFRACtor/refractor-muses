import os
import shutil
import sys
import pickle
import pytest
import refractor.muses.muses_py as mpy
from refractor.muses import RefractorUip
from refractor.framework import load_config_module, find_config_function
from refractor.framework.factory import process_config, creator
from scipy.io import readsav
import glob
import subprocess

test_in_dir = os.path.abspath(os.path.dirname(__file__) + "/../unit_test_data/in") + "/"
test_expected_results_dir = os.path.abspath(os.path.dirname(__file__) + "/../unit_test_data/expected") + "/"

# Short hand for marking as unconditional skipping. Good for tests we
# don't normally run, but might want to comment out for a specific debugging
# reason.
skip = pytest.mark.skip

# Marker for long tests. Only run with --run-long
long_test = pytest.mark.long_test

# Marker for capture tests. Only run with --run-capture
capture_test = pytest.mark.capture_test

# Marker that skips a test if we don't have muses_py available
require_muses_py = pytest.mark.skipif(not mpy.have_muses_py,
      reason="need muses-py available to run")

@pytest.fixture(scope="function")
def osp_dir():
    '''Location of OSP directory.'''
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        raise pytest.skip('test requires OSP directory set by through the MUSES_OSP_PATH environment variable')
    return osp_path

@pytest.fixture(scope="function")
def config_dir():
    '''Returns configuration directory, but also clears the factory singleton instances created during 
    configuration instantiation after a test finishes so that other tests can also create configurations 
    without running into issues with attaching to a state vector twice'''

    yield os.path.abspath(os.path.dirname(__file__) + "/../../config") + "/"

    creator.modifier.clear_singleton_objects()

@pytest.fixture(scope="function")
def gmao_dir():
    '''Location of GAMO directory.'''
    gmao_path = os.environ.get("GMAO_PATH", None)
    if gmao_path is None or not os.path.exists(gmao_path):
        raise pytest.skip('test requires GMAO directory set by through the GMAO_PATH environment variable')
    return gmao_path

@pytest.fixture(scope="function")
def isolated_dir(tmpdir):
    '''This is a fixture that creates a temporary directory, and uses this
    while running a unit tests. Useful for tests that write out a test file
    and then try to read it.

    This fixture changes into the temporary directory, and at the end of
    the test it changes back to the current directory.

    Note that this uses the fixture tmpdir, which keeps around the last few
    temporary directories (cleaning up after a fixed number are generated).
    So if a test fails, you can look at the output at the location of tmpdir, 
    e.g. /tmp/pytest-of-smyth
    '''
    curdir = os.getcwd()
    try:
        tmpdir.chdir()
        yield curdir
    finally:
        os.chdir(curdir)

@pytest.fixture(scope="function")
def clean_up_replacement_function():
    '''Remove any replacement functions that have been added when the
    test ends'''
    if not mpy.have_muses_py:
        raise pytest.skip('test requires muses_py')
    try:
        yield
    finally:
        mpy.unregister_replacement_function_all()

@pytest.fixture(scope="function")
def original_run_dir(isolated_dir, osp_dir):
    '''Set up for an original run. This has the OSP and Table.asc that
    can be use by muses-py for running.'''
    runnum = "20160414_23_394_23"
    os.mkdir(runnum)
    run_dir = f"{os.getcwd()}/{runnum}"
    os.symlink(osp_dir, "./OSP")
    subprocess.run(["mkdir", "-p", "OML1BRUG_003/2016/04/14"], check=True)
    subprocess.run(["mkdir", "-p", "OMCLDO2_003/2016/04/14"], check=True)
    os.symlink(f"{test_in_dir}/OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4",
               "OML1BRUG_003/2016/04/14/OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4")
    os.symlink(f"{test_in_dir}/OMI-Aura_L2-OMCLDO2_2016m0414t2324-o62498_v003-2016m0415t051902.he5",
               "OMCLDO2_003/2016/04/14/OMI-Aura_L2-OMCLDO2_2016m0414t2324-o62498_v003-2016m0415t051902.he5")
    for f in ("Table", "Measurement_ID", "DateTime"):
        shutil.copy(f"{test_in_dir}/{f}_{runnum}.asc",
                    f"{runnum}/{f}.asc")
    os.environ["MUSES_DEFAULT_RUN_DIR"] = run_dir
    return run_dir
    
        
def run_retrieval_parm(step_number=1):
    '''This reads parameters that can be use to call the py-retrieve function
    run_retrieval.  The pickle file comes from py-retrieve, but we copy this 
    local so we can test w/o depending on py-retrieve.
    '''
    return pickle.load(open(test_in_dir + "run_retrieval_step_%d.pkl" %
                            step_number, "rb"))

def fm_wrapper_parm(step_number=1):
    '''This reads parameters that can be use to call the py-retrieve function
    fm_wrapper.  The pickle file comes test_capture_fm_wrapper
    '''
    return pickle.load(open(test_in_dir +
                            "fm_wrapper_step_%d.pkl" % step_number, "rb"))

def load_uip(step_number=1, osp_dir=None):
    uip = pickle.load(open(test_in_dir + "uip_step_%d.pkl" % step_number,
                           "rb"))
    uip.extract_directory(change_to_dir=True, osp_dir=osp_dir)
    return uip

@pytest.fixture(scope="function")
def uip_step_1(isolated_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(step_number=1)

@pytest.fixture(scope="function")
def uip_step_2(isolated_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(step_number=2)


