from refractor.muses import (MusesSpectralWindow, StrategyTable, MusesOmiObservation)
import refractor.framework as rf
from test_support import *

def test_muses_spectral_window(osp_dir):
    # This is an observation that has some bad samples in it.
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    cld_filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    stable = StrategyTable(f"{joint_omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesOmiObservation(filename, xtrack_uv1, xtrack_uv2, atrack,
                              utc_time, calibration_filename,
                              ["UV1", "UV2"],
                              cld_filename=cld_filename,
                              osp_dir=osp_dir)
    step_number = 3
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    swin = MusesSpectralWindow(stable.spectral_window("OMI", stp=step_number+1),
                               obs)
    spec = obs.spectrum_full(1)
    # Check number of good points
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
    # Include bad points
    swin.include_bad_sample = True
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4+3
    swin.include_bad_sample = False
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
    # Check number of full band
    swin.full_band = True
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == spec.spectral_domain.data.shape[0]
    swin.full_band = False
    assert swin.apply(spec, 1).spectral_domain.data.shape[0] == 4
