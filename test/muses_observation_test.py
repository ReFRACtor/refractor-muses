from refractor.muses import (MusesRunDir, MusesAirsObservation,
                             StrategyTable, RetrievalStrategy, ObservationHandleSet,
                             MusesCrisObservation,
                             MusesTropomiObservation, MusesOmiObservation,
                             MeasurementIdFile,
                             CurrentStateDict)
from refractor.old_py_retrieve_wrapper import (TropomiRadiancePyRetrieve, OmiRadiancePyRetrieve,
                                               MusesCrisObservationOld, MusesAirsObservationOld)
import refractor.framework as rf
from test_support import *

def test_measurement_id(isolated_dir, osp_dir, gmao_dir):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    print(osp_dir)
    s = StrategyTable(f"{r.run_dir}/Table.asc", osp_dir=osp_dir)
    mid = MeasurementIdFile(f"{r.run_dir}/Measurement_ID.asc", s)
    assert mid.filter_list == {'OMI': ['UV1', 'UV2'], 'AIRS': ['2B1', '1B2', '2A1', '1A1']}
    assert mid.value_float("OMI_Longitude") == pytest.approx(-154.7512664794922)
    assert mid.value_int("OMI_XTrack_UV1_Index") == 10
    assert os.path.basename(mid.filename("OMI_Cloud_filename")) == "OMI-Aura_L2-OMCLDO2_2016m0401t2215-o62308_v003-2016m0402t044340.he5"
    assert mid.filename("omi_calibrationFilename") == f"{osp_dir}/OMI/OMI_Rad_Cal/JPL_OMI_RadCaL_2006.h5"

def test_muses_airs_observation(isolated_dir, osp_dir, gmao_dir):
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    granule = 231
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    stable = StrategyTable(f"{joint_omi_test_in_dir}/Table.asc", osp_dir=osp_dir)
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
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = MusesSpectralWindow(stable.spectral_window("AIRS", stp=step_number+1),
                                              obs)
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
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    step_number = 8
    rs.strategy_table.table_step = step_number+1
    swin = rs.strategy_table.spectral_window("CRIS")
    fm_sv = rf.StateVector()
    cs = CurrentStateDict(dict(), ["fake",])
    obs = MusesAirsObservation.create_from_id(rs.measurement_id, None,
                                              cs, swin, fm_sv, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)

def test_muses_cris_observation(isolated_dir, osp_dir, gmao_dir):
    granule = 65
    xtrack = 8
    atrack = 4
    pixel_index = 5
    fname = f"{joint_tropomi_test_in_dir}/../nasa_fsr_SNDR.SNPP.CRIS.20190807T0624.m06.g065.L1B.std.v02_22.G.190905161252.nc"
    stable = StrategyTable(f"{joint_tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesCrisObservation.create_from_filename(fname, granule, xtrack, atrack,
                                                    pixel_index, osp_dir=osp_dir)
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
    obs_old = MusesCrisObservationOld(rf_uip, rrefractor.params["ret_info"]["obs_rad"],
                                      rrefractor.params["ret_info"]["meas_err"])
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = MusesSpectralWindow(stable.spectral_window("CRIS", step_number+1),
                                              obs)
    print(obs.spectral_domain(0).data)
    print(obs_old.spectral_domain(0).data)
    npt.assert_allclose(obs.spectral_domain(0).data, obs_old.spectral_domain(0).data)
    print(obs_old.radiance(0).spectral_range.data)
    npt.assert_allclose(obs.radiance(0).spectral_range.data, obs_old.radiance(0).spectral_range.data)
    print([obs_old.rf_uip.uip['microwindows_all'][i] for i in
           range(len(obs_old.rf_uip.uip['microwindows_all']))
           if obs_old.rf_uip.uip['microwindows_all'][i]['instrument'] == "CRIS"])
    # Basic test of serialization, just want to make sure we get no errors
    t = pickle.dumps(obs)
    obs2 = pickle.loads(t)
    
def test_create_muses_cris_observation(isolated_dir, osp_dir, gmao_dir,
                                       vlidort_cli):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    step_number = 12
    rs.strategy_table.table_step = step_number+1
    swin = rs.strategy_table.spectral_window("CRIS")
    fm_sv = rf.StateVector()
    cs = CurrentStateDict(dict(), ["fake",])
    obs = MusesCrisObservation.create_from_id(rs.measurement_id, None,
                                              cs, swin, fm_sv, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
        
def test_muses_tropomi_observation(isolated_dir, osp_dir, gmao_dir):
    xtrack_list = [226,]
    atrack = 2995
    filename_list = [f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc",]
    irr_filename = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    cld_filename = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
    utc_time = "2019-08-07T06:24:33.584090Z"
    filter_list = ["BAND3",]
    stable = StrategyTable(f"{joint_tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    obs = MusesTropomiObservation.create_from_filename(
        filename_list, irr_filename, cld_filename, xtrack_list, atrack, utc_time,
        filter_list, osp_dir=osp_dir)
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
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = MusesSpectralWindow(stable.spectral_window("TROPOMI", step_number+1),
                                              obs)
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
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    step_number = 12
    rs.strategy_table.table_step = step_number+1
    swin = rs.strategy_table.spectral_window("TROPOMI")
    fm_sv = rf.StateVector()
    cs = CurrentStateDict({"TROPOMISOLARSHIFTBAND3" : 0.1,
                           "TROPOMIRADIANCESHIFTBAND3" : 0.2,
                           "TROPOMIRADSQUEEZEBAND3" : 0.3,}
                           , ["TROPOMISOLARSHIFTBAND3",])
    obs = MusesTropomiObservation.create_from_id(rs.measurement_id, None,
                                              cs, swin, fm_sv, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    
def test_muses_omi_observation(isolated_dir, osp_dir, gmao_dir):
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
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = MusesSpectralWindow(stable.spectral_window("OMI", step_number+1),
                                              obs)
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
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    step_number = 8
    rs.strategy_table.table_step = step_number+1
    swin = rs.strategy_table.spectral_window("OMI")
    fm_sv = rf.StateVector()
    cs = CurrentStateDict({"OMINRADWAVUV1" : 0.1,
                           "OMINRADWAVUV2" : 0.11,
                           "OMIODWAVUV1" : 0.2,
                           "OMIODWAVUV2" : 0.21,
                           "OMIODWAVSLOPEUV1" : 0.3,
                           "OMIODWAVSLOPEUV2" : 0.31,}
                           , ["OMINRADWAVUV1",])
    obs = MusesOmiObservation.create_from_id(rs.measurement_id, None,
                                              cs, swin, fm_sv, osp_dir=osp_dir)
    print(obs.spectral_domain(0).data)
    print(obs.radiance(0).spectral_range.data)
    
def test_omi_bad_sample(isolated_dir, osp_dir, gmao_dir):
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
    sv = rf.StateVector()
    sv.add_observer(obs)
    x2 = [0.01, 0.02, 0.03, 0.04, 0.001, 0.002]
    sv.update_state(x2)
    # Note this is off by 1. The table numbering get redone after the BT step. It might
    # be nice to straighten this out - this is actually kind of confusing. Might be better to
    # just have a way to skip steps - but this is at least how the code works. The
    # code mpy.modify_from_bt changes the number of steps
    obs.spectral_window = MusesSpectralWindow(stable.spectral_window("OMI", step_number+1),
                                              obs)
    print(obs.spectral_domain(1).data)
    print(obs.spectral_domain(1).sample_index)
    # Check handling of data with bad samples. Should get set to -999
    print(obs.radiance(1).spectral_range.data)
