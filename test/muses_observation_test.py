from refractor.muses import (MusesRunDir, MusesAirsObservation,
                             MusesCrisObservation,
                             MusesTropomiObservation, MusesOmiObservation,
                             ObservationHandleSet,
                             SimulatedObservation,
                             MeasurementIdFile, RetrievalConfiguration,
                             CurrentStateDict, MusesSpectralWindow)
from refractor.old_py_retrieve_wrapper import (TropomiRadiancePyRetrieve, OmiRadiancePyRetrieve,
                                               MusesCrisObservationOld, MusesAirsObservationOld)
import refractor.framework as rf
from test_support import *
import copy


def test_measurement_id(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc",
                                                               osp_dir=osp_dir)
    flist = {'OMI': ['UV1', 'UV2'], 'AIRS': ['2B1', '1B2', '2A1', '1A1']}
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", rconfig, flist)
    assert mid.filter_list_dict == flist
    assert float(mid["OMI_Longitude"]) == pytest.approx(-154.7512664794922)
    assert int(mid["OMI_XTrack_UV1_Index"]) == 10
    assert os.path.basename(mid["OMI_Cloud_filename"]) == "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    assert mid["omi_calibrationFilename"] == f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"

def test_muses_airs_observation(isolated_dir, osp_dir, gmao_dir):
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    granule = 231
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    obs = MusesAirsObservation.create_from_filename(fname, granule, xtrack, atrack,
                                                    channel_list, osp_dir=osp_dir)
    step_number = 8
    iteration = 2
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    obs_old = MusesAirsObservationOld(rf_uip, rrefractor.params["ret_info"]["obs_rad"],
                                      rrefractor.params["ret_info"]["meas_err"])
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["AIRS"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    print(obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(obs.radiance(0).spectral_range.data, obs_old.radiance(0).spectral_range.data)
    print([obs_old.rf_uip.uip['microwindows_all'][i] for i in
           range(len(obs_old.rf_uip.uip['microwindows_all']))
           if obs_old.rf_uip.uip['microwindows_all'][i]['instrument'] == "AIRS"])
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)

def test_create_muses_airs_observation(isolated_dir, osp_dir, gmao_dir,
                                       vlidort_cli):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    # Determined by looking a the full run
    filter_list_dict = {'OMI': ['UV1', 'UV2'], 'AIRS': ['2B1', '1B2', '2A1', '1A1']}
    measurement_id = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc",
                                       rconfig, filter_list_dict)
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs = MusesAirsObservation.create_from_id(measurement_id, None,
                                        None, swin_dict["AIRS"], None, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
        
def test_muses_tropomi_observation(isolated_dir, osp_dir, gmao_dir):
    xtrack_dict = {"BAND3" : 226, 'CLOUD' : 226, 'IRR_BAND_1to6' : 226}
    atrack_dict = {"BAND3" : 2995, "CLOUD" : 2995}
    filename_dict = {}
    filename_dict["BAND3"] = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc"
    filename_dict['IRR_BAND_1to6'] = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    filename_dict["CLOUD"] = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
    utc_time = "2019-08-07T06:24:33.584090Z"
    filter_list = ["BAND3",]
    obs = MusesTropomiObservation.create_from_filename(
        filename_dict, xtrack_dict, atrack_dict, utc_time, filter_list, osp_dir=osp_dir)
    step_number = 12
    iteration = 2
    rrefractor = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    # The initial shift for everything is 0. Change to something so we can test that
    # this actually gets used.
    rf_uip.tropomi_params["solarshift_BAND3"] = 0.01
    rf_uip.tropomi_params["radianceshift_BAND3"] = 0.02
    rf_uip.tropomi_params["radsqueeze_BAND3"] = 0.03
    fname = glob.glob(f"{rf_uip.run_dir}/Input/Radiance_TROPOMI*.pkl")[0]
    obs_old = TropomiRadiancePyRetrieve(rf_uip)
    sv = rf.StateVector()
    sv.add_observer(obs)
    sv2 = rf.StateVector()
    sv2.add_observer(obs_old)
    x2 = np.array([rf_uip.tropomi_params["solarshift_BAND3"],
                   rf_uip.tropomi_params["radianceshift_BAND3"],
                   rf_uip.tropomi_params["radsqueeze_BAND3"],
                   ])
    sv.update_state(x2)
    sv2.update_state(x2)
    # This is the microwindows file for step 12, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["TROPOMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(0).spectral_range.data-obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(obs.radiance(0).spectral_range.data, obs_old.radiance(0).spectral_range.data, atol=1e-6)
    print(obs.radiance_all())
    print(obs.radiance_all().spectral_range.uncertainty)
    print(obs_old.radiance_all().spectral_range.uncertainty)
    npt.assert_allclose(obs.radiance_all().spectral_range.uncertainty,
                        obs_old.radiance_all().spectral_range.uncertainty, atol=1e-9)
    print(obs.radiance_all().spectral_range.uncertainty -obs_old.radiance_all().spectral_range.uncertainty)
    print(obs.radiance(0).spectral_domain.sample_index)
    print([obs_old.rf_uip.uip['microwindows_all'][i] for i in
           range(len(obs_old.rf_uip.uip['microwindows_all']))
           if obs_old.rf_uip.uip['microwindows_all'][i]['instrument'] == "TROPOMI"])
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)

def test_create_muses_tropomi_observation(isolated_dir, osp_dir, gmao_dir,
                                       vlidort_cli):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    # Determined by looking a the full run
    filter_list_dict = {'TROPOMI': ['BAND3'], 'CRIS': ['2B1', '1B2', '2A1', '1A1']}
    measurement_id = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc",
                                       rconfig, filter_list_dict)
    # This is the microwindows file for step 12, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    cs = CurrentStateDict({"TROPOMISOLARSHIFTBAND3" : 0.1,
                           "TROPOMIRADIANCESHIFTBAND3" : 0.2,
                           "TROPOMIRADSQUEEZEBAND3" : 0.3,}
                           , ["TROPOMISOLARSHIFTBAND3",])
    obs = MusesTropomiObservation.create_from_id(measurement_id, None,
                                                 cs, swin_dict["TROPOMI"], None,
                                                 osp_dir=osp_dir,
                                                 write_tropomi_radiance_pickle=True)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)

def test_create_muses_cris_observation(isolated_dir, osp_dir, gmao_dir,
                                       vlidort_cli):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    # Determined by looking a the full run
    filter_list_dict = {'TROPOMI': ['BAND3'], 'CRIS': ['2B1', '1B2', '2A1', '1A1']}
    measurement_id = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc",
                                       rconfig, filter_list_dict)
    # This is the microwindows file for step 12, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs = MusesCrisObservation.create_from_id(measurement_id, None,
                                                 None, swin_dict["CRIS"], None,
                                                 osp_dir=osp_dir,
                                                 write_tropomi_radiance_pickle=True)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    
def test_muses_omi_observation(isolated_dir, osp_dir, gmao_dir):
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    cld_filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    obs = MusesOmiObservation.create_from_filename(
        filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
        ["UV1", "UV2"], cld_filename=cld_filename, osp_dir=osp_dir)
    step_number = 8
    iteration = 2
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
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
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
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
    print([v.value for i,v in enumerate(obs.solar_radiance(0)) if obs.bad_sample_mask(0)[i] != True])
    print([v.value for i,v in enumerate(obs.solar_radiance(1)) if obs.bad_sample_mask(1)[i] != True])
    print(obs.radiance(0).spectral_range.data)
    print(obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(1).spectral_range.data)
    print(obs_old.radiance(1).spectral_range.data)
    # This is actually different, the interpolation we use vs muses-py is similiar but
    # not identical. We except small differences
    print(obs.radiance(0).spectral_range.data-obs_old.radiance(0).spectral_range.data)
    print(obs.radiance(1).spectral_range.data-obs_old.radiance(1).spectral_range.data)
    npt.assert_allclose(obs.radiance(0).spectral_range.data,
                        obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(obs.radiance(1).spectral_range.data,
                        obs_old.radiance(1).spectral_range.data)
    print(obs.radiance_all())
    print(obs.radiance_all().spectral_range.uncertainty)
    print(obs_old.radiance_all().spectral_range.uncertainty)
    npt.assert_allclose(obs.radiance_all().spectral_range.uncertainty,
                        obs_old.radiance_all().spectral_range.uncertainty)
    print(obs.radiance_all().spectral_range.uncertainty -obs_old.radiance_all().spectral_range.uncertainty)
    print(obs.radiance(0).spectral_domain.sample_index)
    print([obs_old.rf_uip.uip['microwindows_all'][i] for i in
           range(len(obs_old.rf_uip.uip['microwindows_all']))
           if obs_old.rf_uip.uip['microwindows_all'][i]['instrument'] == "OMI"])
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)

def test_create_muses_omi_observation(isolated_dir, osp_dir, gmao_dir,
                                       vlidort_cli):
    # Don't need a lot from run dir, but this modifies the path in Measurement_ID.asc
    # to point to our test data rather than original location of these files, so go
    # ahead and set this up
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    # Determined by looking a the full run
    filter_list_dict = {'OMI': ['UV1', 'UV2'], 'AIRS': ['2B1', '1B2', '2A1', '1A1']}
    measurement_id = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc",
                                       rconfig, filter_list_dict)
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    cs = CurrentStateDict({"OMINRADWAVUV1" : 0.1,
                           "OMINRADWAVUV2" : 0.11,
                           "OMIODWAVUV1" : 0.2,
                           "OMIODWAVUV2" : 0.21,
                           "OMIODWAVSLOPEUV1" : 0.3,
                           "OMIODWAVSLOPEUV2" : 0.31,}
                           , ["OMINRADWAVUV1",])
    obs = MusesOmiObservation.create_from_id(measurement_id, None,
                                              cs, swin_dict["OMI"], None, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    print(obs.filter_data)
    
def test_omi_bad_sample(isolated_dir, osp_dir, gmao_dir):
    xtrack_uv1 = 10
    xtrack_uv2 = 20
    atrack = 1139
    filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L1-OML1BRUG_2016m0401t2215-o62308_v003-2016m0402t041806.he4"
    cld_filename = f"{joint_omi_test_in_dir}/../OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    utc_time = "2016-04-01T23:07:33.676106Z"
    calibration_filename = f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"
    obs = MusesOmiObservation.create_from_filename(
        filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
        ["UV1", "UV2"], cld_filename=cld_filename, osp_dir=osp_dir)
    step_number = 3
    sv = rf.StateVector()
    sv.add_observer(obs)
    x2 = [0.01, 0.02, 0.03, 0.04, 0.001, 0.002]
    sv.update_state(x2)
    # This is the microwindows file for step 8, determined by just running the full
    # retrieval and noting the file used
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs.spectral_window = swin_dict["OMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    print(obs.spectral_domain(1).data)
    print(obs.spectral_domain(1).sample_index)
    # Check handling of data with bad samples. Should get set to -999
    print(obs.radiance(1).spectral_range.data)

def test_simulated_obs(tropomi_obs_step_1):
    rad = [copy.copy(tropomi_obs_step_1.radiance(0).spectral_range.data),]
    rad[0] *= 0.75
    obs = SimulatedObservation(tropomi_obs_step_1, rad)
    npt.assert_allclose(obs.spectral_domain(0).data, tropomi_obs_step_1.spectral_domain(0).data)
    npt.assert_allclose(obs.radiance(0).spectral_range.data,
                        tropomi_obs_step_1.radiance(0).spectral_range.data*0.75)
    
