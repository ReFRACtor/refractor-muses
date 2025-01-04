# This is some stand alone versions of MusesObservation. This is used in a few of
# the old_py_retrieve tests, where because of the set up it is easier to have hard coded
# version of these observations vs. creating from our ObservationHandleSet.
import pytest
from refractor.muses import (
    MusesSpectralWindow,
    MusesTropomiObservation,
    MusesCrisObservation,
    MusesOmiObservation,
    MusesAirsObservation,
)


@pytest.fixture(scope="function")
def joint_tropomi_obs_step_12(osp_dir, joint_tropomi_test_in_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_dict = {"BAND3": 226, "CLOUD": 226, "IRR_BAND_1to6": 226}
    atrack_dict = {"BAND3": 2995, "CLOUD": 2995}
    filename_dict = {}
    filename_dict["BAND3"] = (
        joint_tropomi_test_in_dir.parent
        / "S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc"
    )
    filename_dict["IRR_BAND_1to6"] = (
        joint_tropomi_test_in_dir.parent
        / "S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    )
    filename_dict["CLOUD"] = (
        joint_tropomi_test_in_dir.parent
        / "S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
    )
    utc_time = "2019-08-07T06:24:33.584090Z"
    filter_list = [
        "BAND3",
    ]
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs = MusesTropomiObservation.create_from_filename(
        filename_dict, xtrack_dict, atrack_dict, utc_time, filter_list, osp_dir=osp_dir
    )
    obs.spectral_window = swin_dict["TROPOMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    granule = 65
    xtrack = 8
    atrack = 4
    pixel_index = 5
    fname = (
        joint_tropomi_test_in_dir.parent
        / "nasa_fsr_SNDR.SNPP.CRIS.20190807T0624.m06.g065.L1B.std.v02_22.G.190905161252.nc"
    )
    obscris = MusesCrisObservation.create_from_filename(
        fname, granule, xtrack, atrack, pixel_index, osp_dir=osp_dir
    )
    obscris.spectral_window = swin_dict["CRIS"]
    obscris.spectral_window.add_bad_sample_mask(obscris)
    return [obscris, obs]


@pytest.fixture(scope="function")
def joint_omi_obs_step_8(osp_dir, joint_omi_test_in_dir):
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    )
    calibration_filename = osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    cld_filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    utc_time = "2016-04-01T23:07:33.676106Z"
    filter_list = ["UV1", "UV2"]
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    channel_list = ["1A1", "2A1", "1B2", "2B1"]
    swin_dict = MusesSpectralWindow.create_dict_from_file(
        mwfile, filter_list_dict={"OMI": filter_list, "AIRS": channel_list}
    )
    obs = MusesOmiObservation.create_from_filename(
        filename,
        xtrack_uv1,
        xtrack_uv2,
        atrack,
        utc_time,
        calibration_filename,
        filter_list,
        cld_filename=cld_filename,
        osp_dir=osp_dir,
    )
    obs.spectral_window = swin_dict["OMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    granule = 231
    xtrack = 29
    atrack = 49
    fname = (
        joint_omi_test_in_dir.parent
        / "AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    )
    obs_airs = MusesAirsObservation.create_from_filename(
        fname, granule, xtrack, atrack, channel_list, osp_dir=osp_dir
    )
    obs_airs.spectral_window = swin_dict["AIRS"]
    obs_airs.spectral_window.add_bad_sample_mask(obs_airs)
    return [obs_airs, obs]
