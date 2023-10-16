import os
import shutil
import sys
import pickle
import pytest
import refractor.muses.muses_py as mpy
from refractor.muses import (RefractorUip, osswrapper, MusesRetrievalStep,
                             MusesResidualFmJacobian)
from refractor.framework import load_config_module, find_config_function
from refractor.framework.factory import process_config, creator
from scipy.io import readsav
import glob
import subprocess
import numpy as np
import numpy.testing as npt
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import logging
import io

if("REFRACTOR_TEST_DATA" in os.environ):
    test_base_path = os.environ["REFRACTOR_TEST_DATA"]
else:
    test_base_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../../refractor_test_data")

tropomi_test_in_dir = f"{test_base_path}/tropomi/in/sounding_1"
joint_tropomi_test_in_dir = f"{test_base_path}/cris_tropomi/in/sounding_1"
omi_test_in_dir = f"{test_base_path}/omi/in/sounding_1"
joint_omi_test_in_dir = f"{test_base_path}/airs_omi/in/sounding_1"

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

# Fake creation date used in muses-py file creation, to make comparing easier since
# we don't get differences that are just the time of creation
os.environ["MUSES_FAKE_CREATION_DATE"] = "FAKE_DATE"

@pytest.fixture(scope="function")
def osp_dir():
    '''Location of OSP directory.'''
    osp_path = os.environ.get("MUSES_OSP_PATH", None)
    if osp_path is None or not os.path.exists(osp_path):
        raise pytest.skip('test requires OSP directory set by through the MUSES_OSP_PATH environment variable')
    return osp_path

@pytest.fixture(scope="function")
def gmao_dir():
    '''Location of GAMO directory.'''
    gmao_path = os.environ.get("MUSES_GMAO_PATH", None)
    if gmao_path is None or not os.path.exists(gmao_path):
        raise pytest.skip('test requires GMAO directory set by through the MUSES_GMAO_PATH environment variable')
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
        osswrapper.register_with_muses_py()

def load_muses_retrieval_step(dir_in, step_number=1, osp_dir=None,
                              gmao_dir=None, change_to_dir=True):
    '''This reads parameters that can be use to call the py-retrieve function
    run_retrieval. See muses_capture in refractor-muses for collecting this.
    '''
    return MusesRetrievalStep.load_retrieval_step(
        f"{dir_in}/run_retrieval_step_{step_number}.pkl",
        osp_dir=osp_dir, gmao_dir=gmao_dir,change_to_dir=change_to_dir)

def muses_residual_fm_jac(dir_in, step_number=1, iteration=1,
                          osp_dir=None, gmao_dir=None,
                          path=".",
                          change_to_dir=True):
    '''This reads parameters that can be use to call the py-retrieve function
    residual_fm_jac. See muses_capture in refractor-muses for collecting this.
    '''
    return MusesResidualFmJacobian.load_residual_fm_jacobian(
        f"{dir_in}/residual_fm_jac_{step_number}_{iteration}.pkl",
        osp_dir=osp_dir, gmao_dir=gmao_dir,path=path,
        change_to_dir=change_to_dir)

def load_uip(dir_in, step_number=1, osp_dir=None, gmao_dir=None):
    return  RefractorUip.load_uip(
        f"{dir_in}/uip_step_{step_number}.pkl",
        change_to_dir=True,
        osp_dir=osp_dir,gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def tropomi_uip_step_1(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(tropomi_test_in_dir, step_number=1,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def tropomi_uip_step_2(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 2, and also unpack all the 
    support files into a directory'''
    return load_uip(tropomi_test_in_dir, step_number=2,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)
                    

@pytest.fixture(scope="function")
def omi_uip_step_1(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(omi_test_in_dir, step_number=1,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def omi_uip_step_2(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 2, and also unpack all the 
    support files into a directory'''
    return load_uip(omi_test_in_dir, step_number=2,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)
                    

@pytest.fixture(scope="function")
def joint_tropomi_uip_step_12(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(joint_tropomi_test_in_dir, step_number=12,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def joint_omi_uip_step_8(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(joint_omi_test_in_dir, step_number=8,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def vlidort_cli():
    return os.environ.get("MUSES_VLIDORT_CLI", "~/muses/muses-vlidort/build/release/vlidort_cli")

@contextmanager
def all_output_disabled():
    '''Suppress stdout, stderr, and logging, useful for some of the noisy output we
    get running muses-py code.'''
    previous_level = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with redirect_stdout(io.StringIO()) as sout:
            with redirect_stderr(io.StringIO()) as serr:
                yield
    finally:
        logging.disable(previous_level)
