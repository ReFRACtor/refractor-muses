import os
import shutil
import sys
import pickle
import pytest
import refractor.muses.muses_py as mpy
from refractor.muses import (RefractorUip, osswrapper, StrategyTable,
                             MusesTropomiObservation, MusesOmiObservation,
                             MusesCrisObservation, MusesAirsObservation)
from refractor.framework import load_config_module, find_config_function
from refractor.framework.factory import process_config, creator
from refractor.old_py_retrieve_wrapper import MusesResidualFmJacobian, MusesRetrievalStep
from scipy.io import readsav
import netCDF4 as ncdf
import glob
import subprocess
import numpy as np
import numpy.testing as npt
import os
from contextlib import redirect_stdout, redirect_stderr, contextmanager
import logging
import io
import subprocess

if("REFRACTOR_TEST_DATA" in os.environ):
    test_base_path = os.environ["REFRACTOR_TEST_DATA"]
else:
    test_base_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../../refractor_test_data")

tropomi_test_in_dir = f"{test_base_path}/tropomi/in/sounding_1"
tropomi_test_in_dir2 = f"{test_base_path}/tropomi/in/sounding_2"
tropomi_test_in_dir3 = f"{test_base_path}/tropomi/in/sounding_3"
tropomi_band7_test_in_dir = f"{test_base_path}/tropomi_band7/in/sounding_1"
tropomi_band7_expected_results_dir = f"{test_base_path}/tropomi_band7/expected/"
tropomi_band7_swir_step_test_in_dir = f"{test_base_path}/tropomi_band7/in/sounding_one_step"
joint_tropomi_test_in_dir = f"{test_base_path}/cris_tropomi/in/sounding_1"
joint_tropomi_test_expected_dir = f"{test_base_path}/cris_tropomi/expected/sounding_1"
joint_tropomi_test_refractor_expected_dir = f"{test_base_path}/cris_tropomi/expected/sounding_1_refractor"
joint_tropomi_test_in_dir2 = f"{test_base_path}/cris_tropomi/in/sounding_2"
omi_test_in_dir = f"{test_base_path}/omi/in/sounding_1"
omi_test_expected_results_dir = f"{test_base_path}/omi/expected"
joint_omi_test_in_dir = f"{test_base_path}/airs_omi/in/sounding_1"
joint_omi_test_expected_dir = f"{test_base_path}/airs_omi/expected/sounding_1"
joint_omi_test_refractor_expected_dir = f"{test_base_path}/airs_omi/expected/sounding_1_refractor"
if not os.path.exists(tropomi_test_in_dir):
    raise RuntimeError(f"ERROR: unit test data not found at {test_base_path}. Please clone the test data repo there.")

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
def tropomi_obs_step_1(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_list = [226,]
    atrack = 359
    filename_list = [f"{tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T001931_20190807T020100_09401_01_010000_20190807T034730.nc",]
    irr_filename = f"{tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    cld_filename = f"{tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T001931_20190807T020100_09401_01_010107_20190812T234805.nc"
    utc_time = "2019-08-07T00:46:06.179000Z"
    filter_list = ["BAND3",]
    stable = StrategyTable(f"{tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation(filename_list, irr_filename, cld_filename,
                                     xtrack_list, atrack, utc_time, filter_list,
                                     osp_dir=osp_dir)
    swin = stable.spectral_window("TROPOMI", stp=0)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("TROPOMI", stp=0)
    obs.spectral_window_with_bad_sample = swin2
    return obs

@pytest.fixture(scope="function")
def tropomi_obs_sounding_2_band7(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_list = [205,]
    atrack = 2297
    filename_list = [f"{tropomi_test_in_dir2}/../S5P_RPRO_L1B_RA_BD7_20220628T185806_20220628T203935_24394_03_020100_20230104T092546.nc",]
    irr_filename = f"{tropomi_test_in_dir2}/../S5P_RPRO_L1B_IR_SIR_20220628T084907_20220628T103037_24388_03_020100_20230104T091244.nc"
    cld_filename = f"{tropomi_test_in_dir2}/../S5P_RPRO_L2__CLOUD__20220628T171636_20220628T185806_24393_03_020401_20230119T091435.nc"
    utc_time = "2022-06-28T18:07:51.984098Z"
    filter_list = ["BAND7",]
    stable = StrategyTable(f"{tropomi_test_in_dir2}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation(filename_list, irr_filename, cld_filename,
                                     xtrack_list, atrack, utc_time, filter_list,
                                     osp_dir=osp_dir)
    swin = stable.spectral_window("TROPOMI", stp=0)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("TROPOMI", stp=0)
    obs.spectral_window_with_bad_sample = swin2
    return obs

@pytest.fixture(scope="function")
def tropomi_obs_step_2(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_list = [226,]
    atrack = 359
    filename_list = [f"{tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T001931_20190807T020100_09401_01_010000_20190807T034730.nc",]
    irr_filename = f"{tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    cld_filename = f"{tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T001931_20190807T020100_09401_01_010107_20190812T234805.nc"
    utc_time = "2019-08-07T00:46:06.179000Z"
    filter_list = ["BAND3",]
    stable = StrategyTable(f"{tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation(filename_list, irr_filename, cld_filename,
                                     xtrack_list, atrack, utc_time, filter_list,
                                     osp_dir=osp_dir)
    swin = stable.spectral_window("TROPOMI", stp=1)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("TROPOMI", stp=1)
    obs.spectral_window_with_bad_sample = swin2
    return obs

@pytest.fixture(scope="function")
def joint_tropomi_obs_step_12(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_list = [226,]
    atrack = 2995
    filename_list = [f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc",]
    irr_filename = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    cld_filename = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
    utc_time = "2019-08-07T06:24:33.584090Z"
    filter_list = ["BAND3",]
    stable = StrategyTable(f"{joint_tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation(filename_list, irr_filename, cld_filename,
                                     xtrack_list, atrack, utc_time, filter_list,
                                     osp_dir=osp_dir)
    swin = stable.spectral_window("TROPOMI", stp=12+1)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("TROPOMI", stp=12+1)
    obs.spectral_window_with_bad_sample = swin2
    granule = 65
    xtrack = 8
    atrack = 4
    pixel_index = 5
    fname = f"{joint_tropomi_test_in_dir}/../nasa_fsr_SNDR.SNPP.CRIS.20190807T0624.m06.g065.L1B.std.v02_22.G.190905161252.nc"
    obscris = MusesCrisObservation.create_from_filename(
        fname, granule, xtrack, atrack, pixel_index, osp_dir=osp_dir)
    swin = stable.spectral_window("CRIS", stp=12+1)
    swin.bad_sample_mask(obscris.bad_sample_mask(0), 0)
    obscris.spectral_window = swin
    swin2 = stable.spectral_window("CRIS", stp=12+1)
    obscris.spectral_window_with_bad_sample = swin2
    return [obscris, obs]

@pytest.fixture(scope="function")
def joint_omi_obs_step_8(osp_dir):
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    cld_filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    utc_time = "2016-04-01T23:07:33.676106Z"
    filter_list = ["UV1", "UV2"]
    stable = StrategyTable(f"{joint_omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesOmiObservation(filename, xtrack_uv1, xtrack_uv2, atrack,
                                 utc_time, calibration_filename,
                                 filter_list,
                                 cld_filename=cld_filename,
                                 osp_dir=osp_dir)
    swin = stable.spectral_window("OMI", stp=8+1)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    swin.bad_sample_mask(obs.bad_sample_mask(1), 1)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("OMI", stp=8+1)
    obs.spectral_window_with_bad_sample = swin2
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    granule = 231
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    obs_airs = MusesAirsObservation.create_from_filename(
        fname, granule, xtrack, atrack, channel_list, osp_dir=osp_dir)
    swin = stable.spectral_window("AIRS", stp=8+1)
    swin.bad_sample_mask(obs_airs.bad_sample_mask(0), 0)
    obs_airs.spectral_window = swin
    swin2 = stable.spectral_window("AIRS", stp=8+1)
    obs_airs.spectral_window_with_bad_sample = swin2
    return [obs_airs, obs]
    

@pytest.fixture(scope="function")
def tropomi_uip_step_2(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 2, and also unpack all the 
    support files into a directory'''
    return load_uip(tropomi_test_in_dir, step_number=2,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)
                    
@pytest.fixture(scope="function")
def tropomi_uip_sounding_2_step_1(isolated_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(tropomi_test_in_dir2, step_number=1)

@pytest.fixture(scope="function")
def tropomi_uip_band7_step_1(isolated_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(tropomi_band7_test_in_dir, step_number=1)


@pytest.fixture(scope="function")
def tropomi_uip_band7_swir_step(isolated_dir):
    '''Return a RefractorUip for step 1 of a test-only strategy table with
    only the Band 7 SWIR step, and also unpack all the support files into a
    directory'''
    return load_uip(tropomi_band7_swir_step_test_in_dir, step_number=1)

@pytest.fixture(scope="function")
def tropomi_obs_band7_swir_step(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_list = [108,]
    atrack = 1008
    filename_list = [f"{tropomi_band7_swir_step_test_in_dir}/../S5P_OFFL_L1B_RA_BD7_20220628T185806_20220628T203935_24394_02_020000_20220628T222834.nc",]
    irr_filename = f"{tropomi_band7_swir_step_test_in_dir}/../S5P_RPRO_L1B_IR_SIR_20220628T084907_20220628T103037_24388_03_020100_20230104T091244.nc"
    cld_filename = f"{tropomi_band7_swir_step_test_in_dir}/../S5P_RPRO_L2__CLOUD__20220628T185806_20220628T203935_24394_03_020401_20230119T091438.nc"
    utc_time = "2022-06-28T19:33:47.130000Z"
    filter_list = ["BAND7",]
    stable = StrategyTable(f"{tropomi_band7_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation(filename_list, irr_filename, cld_filename,
                                     xtrack_list, atrack, utc_time, filter_list,
                                     osp_dir=osp_dir)
    swin = stable.spectral_window("TROPOMI", stp=0)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("TROPOMI", stp=0)
    obs.spectral_window_with_bad_sample = swin2
    return obs

@pytest.fixture(scope="function")
def tropomi_band7_simple_ils_test_data():
    simple_results_file = os.path.join(tropomi_band7_expected_results_dir, 'ils', 'simple_ils_test.nc')
    with ncdf.Dataset(simple_results_file) as ds:
        return {
            'hi_res_freq': ds['hi_res_freq'][:].filled(np.nan),
            'hi_res_spec': ds['hi_res_spectrum'][:].filled(np.nan),
            'convolved_spec': ds['convolved_spectrum'][:].filled(np.nan)
        }

@pytest.fixture(scope="function")
def omi_config_dir():
    '''Returns configuration directory'''
    yield os.path.abspath(os.path.dirname(__file__) + "/../../python/refractor/omi/config") + "/"

@pytest.fixture(scope="function")
def omi_uip_step_1(isolated_dir, osp_dir, gmao_dir):
    '''Return a RefractorUip for strategy step 1, and also unpack all the 
    support files into a directory'''
    return load_uip(omi_test_in_dir, step_number=1,
                    osp_dir=osp_dir, gmao_dir=gmao_dir)

@pytest.fixture(scope="function")
def omi_obs_step_1(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_uv1 = 11
    xtrack_uv2 = 23
    atrack = 394
    filename = f"{omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    cld_filename = f"{omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0414t2324-o62498_v003-2016m0415t051902.he5"
    utc_time = "2016-04-14T23:59:46.000000Z"
    filter_list = ["UV1", "UV2"]
    stable = StrategyTable(f"{omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesOmiObservation(filename, xtrack_uv1, xtrack_uv2, atrack,
                                 utc_time, calibration_filename,
                                 filter_list,
                                 cld_filename=cld_filename,
                                 osp_dir=osp_dir)
    swin = stable.spectral_window("OMI", stp=0)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("OMI", stp=0)
    obs.spectral_window_with_bad_sample = swin2
    return obs

@pytest.fixture(scope="function")
def omi_obs_step_2(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_uv1 = 11
    xtrack_uv2 = 23
    atrack = 394
    filename = f"{omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0414t2324-o62498_v003-2016m0415t050532.he4"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    cld_filename = f"{omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0414t2324-o62498_v003-2016m0415t051902.he5"
    utc_time = "2016-04-14T23:59:46.000000Z"
    filter_list = ["UV1", "UV2"]
    stable = StrategyTable(f"{omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesOmiObservation(filename, xtrack_uv1, xtrack_uv2, atrack,
                                 utc_time, calibration_filename,
                                 filter_list,
                                 cld_filename=cld_filename,
                                 osp_dir=osp_dir)
    swin = stable.spectral_window("OMI", stp=1)
    swin.bad_sample_mask(obs.bad_sample_mask(0), 0)
    obs.spectral_window = swin
    swin2 = stable.spectral_window("OMI", stp=1)
    obs.spectral_window_with_bad_sample = swin2
    return obs

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

def struct_compare(s1, s2, skip_list=None, verbose=False):
    if(skip_list is None):
        skip_list = []
    for k in s1.keys():
        if(k in skip_list):
            if(verbose):
                print(f"Skipping {k}")
            continue
        if(verbose):
            print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

        
