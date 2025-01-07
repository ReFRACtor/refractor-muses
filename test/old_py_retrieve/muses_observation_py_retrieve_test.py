# Tests of our MusesObservation that we compare against the old py-retrieve code
from refractor.old_py_retrieve_wrapper import (
    TropomiRadiancePyRetrieve,
    OmiRadiancePyRetrieve,
    MusesAirsObservationOld,
)
from refractor.muses import (
    MusesAirsObservation,
    MusesTropomiObservation,
    MusesOmiObservation,
    MusesSpectralWindow,
    RefractorUip,
)
import refractor.framework as rf
import pickle
import numpy.testing as npt
from fixtures.residual_fm import (
    joint_omi_residual_fm_jac,
    joint_tropomi_residual_fm_jac,
)
import pytest
import numpy as np


@pytest.mark.old_py_retrieve_test
def test_muses_airs_observation(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    """This compares MusesAirsObservation against the old py-retrieve code.
    We don't actually use the old py-retrieve code anymore.

    We'll leave this test in for now, in case we run into some issue. But
    at some point we may remove this, the cost of maintaining the old
    interface in MusesAirsObservationOld might not be worth the effort.
    """
    channel_list = ["1A1", "2A1", "1B2", "2B1"]
    granule = 231
    xtrack = 29
    atrack = 49
    fname = (
        joint_omi_test_in_dir.parent
        / "AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    )
    obs = MusesAirsObservation.create_from_filename(
        fname, granule, xtrack, atrack, channel_list, osp_dir=osp_dir
    )
    rrefractor = joint_omi_residual_fm_jac(
        osp_dir, gmao_dir, joint_omi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    obs_old = MusesAirsObservationOld(
        rf_uip,
        rrefractor.params["ret_info"]["obs_rad"],
        rrefractor.params["ret_info"]["meas_err"],
    )
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["AIRS"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    print(obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(
        obs.radiance(0).spectral_range.data, obs_old.radiance(0).spectral_range.data
    )
    print(
        [
            obs_old.rf_uip.uip["microwindows_all"][i]
            for i in range(len(obs_old.rf_uip.uip["microwindows_all"]))
            if obs_old.rf_uip.uip["microwindows_all"][i]["instrument"] == "AIRS"
        ]
    )
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)
    print(obs2)


@pytest.mark.old_py_retrieve_test
def test_muses_tropomi_observation(
    isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir
):
    """This compares MusesTropomiObservation against the old py-retrieve code.
    We don't actually use the old py-retrieve code anymore.

    We'll leave this test in for now, in case we run into some issue. But
    at some point we may remove this, the cost of maintaining the old
    interface in TropomiRadiancePyRetrieve might not be worth the effort.
    """
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
    obs = MusesTropomiObservation.create_from_filename(
        filename_dict, xtrack_dict, atrack_dict, utc_time, filter_list, osp_dir=osp_dir
    )
    rrefractor = joint_tropomi_residual_fm_jac(
        osp_dir, gmao_dir, joint_tropomi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    # The initial shift for everything is 0. Change to something so we can test that
    # this actually gets used.
    rf_uip.tropomi_params["solarshift_BAND3"] = 0.01
    rf_uip.tropomi_params["radianceshift_BAND3"] = 0.02
    rf_uip.tropomi_params["radsqueeze_BAND3"] = 0.03
    obs_old = TropomiRadiancePyRetrieve(rf_uip)
    sv = rf.StateVector()
    sv.add_observer(obs)
    sv2 = rf.StateVector()
    sv2.add_observer(obs_old)
    x2 = np.array(
        [
            rf_uip.tropomi_params["solarshift_BAND3"],
            rf_uip.tropomi_params["radianceshift_BAND3"],
            rf_uip.tropomi_params["radsqueeze_BAND3"],
        ]
    )
    sv.update_state(x2)
    sv2.update_state(x2)
    # This is the microwindows file for step 12, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["TROPOMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(0).spectral_range.data - obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(
        obs.radiance(0).spectral_range.data,
        obs_old.radiance(0).spectral_range.data,
        atol=1e-6,
    )
    print(obs.radiance_all())
    print(obs.radiance_all().spectral_range.uncertainty)
    print(obs_old.radiance_all().spectral_range.uncertainty)
    npt.assert_allclose(
        obs.radiance_all().spectral_range.uncertainty,
        obs_old.radiance_all().spectral_range.uncertainty,
        atol=1e-9,
    )
    print(
        obs.radiance_all().spectral_range.uncertainty
        - obs_old.radiance_all().spectral_range.uncertainty
    )
    print(obs.radiance(0).spectral_domain.sample_index)
    print(
        [
            obs_old.rf_uip.uip["microwindows_all"][i]
            for i in range(len(obs_old.rf_uip.uip["microwindows_all"]))
            if obs_old.rf_uip.uip["microwindows_all"][i]["instrument"] == "TROPOMI"
        ]
    )
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)
    print(obs2)


@pytest.mark.old_py_retrieve_test
def test_muses_omi_observation(isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir):
    """This compares MusesOmiObservation against the old py-retrieve code.
    We don't actually use the old py-retrieve code anymore.

    We'll leave this test in for now, in case we run into some issue. But
    at some point we may remove this, the cost of maintaining the old
    interface in OmiRadiancePyRetrieve might not be worth the effort.
    """
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    )
    cld_filename = (
        joint_omi_test_in_dir.parent
        / "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    )
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = osp_dir / "OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    obs = MusesOmiObservation.create_from_filename(
        filename,
        xtrack_uv1,
        xtrack_uv2,
        atrack,
        utc_time,
        calibration_filename,
        ["UV1", "UV2"],
        cld_filename=cld_filename,
        osp_dir=osp_dir,
    )
    rrefractor = joint_omi_residual_fm_jac(
        osp_dir, gmao_dir, joint_omi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.omi_params["nradwav_uv1"] = 0.01
    rf_uip.omi_params["nradwav_uv2"] = 0.02
    rf_uip.omi_params["odwav_uv1"] = 0.03
    rf_uip.omi_params["odwav_uv2"] = 0.04
    rf_uip.omi_params["odwav_slope_uv1"] = 0.001
    rf_uip.omi_params["odwav_slope_uv2"] = 0.002
    rf_uip.run_dir = rrefractor.run_dir
    obs_old = OmiRadiancePyRetrieve(rf_uip)
    sv = rf.StateVector()
    sv.add_observer(obs)
    sv2 = rf.StateVector()
    sv2.add_observer(obs_old)
    x2 = [0.01, 0.02, 0.03, 0.04, 0.001, 0.002]
    sv.update_state(x2)
    sv2.update_state(x2)
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = (
        osp_dir
        / "Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    )
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["OMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    print(obs.spectral_domain(1).data)
    print(obs_old.spectral_domain(1).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(1).data, obs_old.spectral_domain(1).data)
    print("Solar radiance mine")
    print(
        [
            v.value
            for i, v in enumerate(obs.solar_radiance(0))
            if obs.bad_sample_mask(0)[i] != True
        ]
    )
    print(
        [
            v.value
            for i, v in enumerate(obs.solar_radiance(1))
            if obs.bad_sample_mask(1)[i] != True
        ]
    )
    print(obs.radiance(0).spectral_range.data)
    print(obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(1).spectral_range.data)
    print(obs_old.radiance(1).spectral_range.data)
    # This is actually different, the interpolation we use vs muses-py is similiar but
    # not identical. We except small differences
    print(obs.radiance(0).spectral_range.data - obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(1).spectral_range.data - obs_old.radiance(1).spectral_range.data)
    npt.assert_allclose(
        obs.radiance(0).spectral_range.data, obs_old.radiance(0).spectral_range.data
    )
    npt.assert_allclose(
        obs.radiance(1).spectral_range.data, obs_old.radiance(1).spectral_range.data
    )
    print(obs.radiance_all())
    print(obs.radiance_all().spectral_range.uncertainty)
    print(obs_old.radiance_all().spectral_range.uncertainty)
    npt.assert_allclose(
        obs.radiance_all().spectral_range.uncertainty,
        obs_old.radiance_all().spectral_range.uncertainty,
    )
    print(
        obs.radiance_all().spectral_range.uncertainty
        - obs_old.radiance_all().spectral_range.uncertainty
    )
    print(obs.radiance(0).spectral_domain.sample_index)
    print(
        [
            obs_old.rf_uip.uip["microwindows_all"][i]
            for i in range(len(obs_old.rf_uip.uip["microwindows_all"]))
            if obs_old.rf_uip.uip["microwindows_all"][i]["instrument"] == "OMI"
        ]
    )
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)
    print(obs2)
