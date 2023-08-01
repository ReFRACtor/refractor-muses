import os
import shutil
import sys
import pickle
import pytest
import refractor.muses.muses_py as mpy
from refractor.muses import RefractorUip, osswrapper
from refractor.framework import load_config_module, find_config_function
from refractor.framework.factory import process_config, creator
from scipy.io import readsav
import glob
import subprocess

if("REFRACTOR_TEST_DATA" in os.environ):
    test_base_path = os.environ["REFRACTOR_TEST_DATA"]
else:
    test_base_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../../refractor_test_data")

# Short hand for marking as unconditional skipping. Good for tests we
# don't normally run, but might want to comment out for a specific debugging
# reason.
skip = pytest.mark.skip

# Marker for long tests. Only run with --run-long
long_test = pytest.mark.long_test

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


def load_tropomi_uip(step_number=1, osp_dir=None, gmao_dir=None):
    return  RefractorUip.load_uip(
        test_base_path + "/tropomi/in/sounding_1/uip_step_%d.pkl" % step_number,
        change_to_dir=True,
        osp_dir=osp_dir,gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def tropomi_uip_step_1(isolated_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_tropomi_uip(step_number=1)


@pytest.fixture(scope="function")
def tropomi_uip_step_2(isolated_dir):
    '''Return a RefractorUip for strategy step 2, and also unpack all the 
    support files into a directory'''
    return load_tropomi_uip(step_number=2)

@pytest.fixture(scope="function")
def tropomi_uip_step_3(isolated_dir):
    '''Return a RefractorUip for strategy step 3, and also unpack all the 
    support files into a directory'''
    return load_tropomi_uip(step_number=3)

@pytest.fixture(scope="function")
def vlidort_cli():
    return os.environ.get("MUSES_VLIDORT_CLI", "~/muses/muses-vlidort/build/release/vlidort_cli")
