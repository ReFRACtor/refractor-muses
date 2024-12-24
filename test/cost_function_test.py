from test_support import *
from test_support.old_py_retrieve_test_support import *
from refractor.muses import (CostFunctionCreator, muses_py_call,
                             osswrapper, RetrievalConfiguration,
                             MeasurementIdFile)
from refractor.omi import OmiForwardModelHandle
from refractor.tropomi import TropomiForwardModelHandle
import refractor.muses.muses_py as mpy

@pytest.fixture(scope="function")
def joint_tropomi_obs_step_12(osp_dir):
    # Observation going with trompomi_uip_step_1
    xtrack_dict = {"BAND3" : 226, 'CLOUD' : 226, 'IRR_BAND_1to6' : 226}
    atrack_dict = {"BAND3" : 2995, "CLOUD" : 2995}
    filename_dict = {}
    filename_dict["BAND3"] = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_RA_BD3_20190807T052359_20190807T070529_09404_01_010000_20190807T084854.nc"
    filename_dict['IRR_BAND_1to6'] = f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L1B_IR_UVN_20190807T034230_20190807T052359_09403_01_010000_20190807T070824.nc"
    filename_dict["CLOUD"]= f"{joint_tropomi_test_in_dir}/../S5P_OFFL_L2__CLOUD__20190807T052359_20190807T070529_09404_01_010107_20190813T045051.nc"
    utc_time = "2019-08-07T06:24:33.584090Z"
    filter_list = ["BAND3",]
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-CrIS-TROPOMI-v7/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile)
    obs = MusesTropomiObservation.create_from_filename(
        filename_dict, xtrack_dict, atrack_dict, utc_time, filter_list, osp_dir=osp_dir)
    obs.spectral_window = swin_dict["TROPOMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    granule = 65
    xtrack = 8
    atrack = 4
    pixel_index = 5
    fname = f"{joint_tropomi_test_in_dir}/../nasa_fsr_SNDR.SNPP.CRIS.20190807T0624.m06.g065.L1B.std.v02_22.G.190905161252.nc"
    obscris = MusesCrisObservation.create_from_filename(
        fname, granule, xtrack, atrack, pixel_index, osp_dir=osp_dir)
    obscris.spectral_window = swin_dict["CRIS"]
    obscris.spectral_window.add_bad_sample_mask(obscris)
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
    mwfile = f"{osp_dir}/Strategy_Tables/ops/OSP-OMI-AIRS-v10/MWDefinitions/Windows_Nadir_H2O_O3_joint.asc"
    channel_list = ['1A1', '2A1', '1B2', '2B1']
    swin_dict = MusesSpectralWindow.create_dict_from_file(mwfile, filter_list_dict={"OMI" : filter_list, "AIRS" : channel_list})
    obs = MusesOmiObservation.create_from_filename(
        filename, xtrack_uv1, xtrack_uv2, atrack, utc_time, calibration_filename,
        filter_list, cld_filename=cld_filename, osp_dir=osp_dir)
    obs.spectral_window = swin_dict["OMI"]
    obs.spectral_window.add_bad_sample_mask(obs)
    granule = 231
    xtrack = 29
    atrack = 49
    fname = f"{joint_omi_test_in_dir}/../AIRS.2016.04.01.231.L1B.AIRS_Rad.v5.0.23.0.G16093121520.hdf"
    obs_airs = MusesAirsObservation.create_from_filename(
        fname, granule, xtrack, atrack, channel_list, osp_dir=osp_dir)
    obs_airs.spectral_window = swin_dict["AIRS"]
    obs_airs.spectral_window.add_bad_sample_mask(obs_airs)
    return [obs_airs, obs]


@old_py_retrieve_test    
def test_fm_wrapper_tropomi(joint_tropomi_step_12, vlidort_cli, osp_dir):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    rs, rstep, _ = joint_tropomi_step_12
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_cris = rs.observation_handle_set.observation(
        "CRIS", rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["CRIS"],
        None,osp_dir=osp_dir)
    obs_tropomi = rs.observation_handle_set.observation(
        "TROPOMI", rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["TROPOMI"],
        None,osp_dir=osp_dir, write_tropomi_radiance_pickle=True)
    creator = CostFunctionCreator()
    obs_cris.spectral_window.include_bad_sample = True
    obs_tropomi.spectral_window.include_bad_sample = True    
    cfunc = creator.cost_function_from_uip(rf_uip, [obs_cris, obs_tropomi],
                                           None, vlidort_cli=vlidort_cli)
    (o_radiance, jac_fm, bad_flag,
     o_measured_radiance_omi, o_measured_radiance_tropomi) = \
         cfunc.fm_wrapper(rf_uip.uip, None, {})
    with osswrapper(rf_uip.uip):
        with muses_py_call(rf_uip.run_dir, vlidort_cli=vlidort_cli):
            (o_radiance2, jac_fm2, bad_flag2,
             o_measured_radiance_omi2, o_measured_radiance_tropomi2) = \
                 mpy.fm_wrapper(rf_uip.uip, None, {})
    for k in o_radiance.keys():
        if(isinstance(o_radiance[k], np.ndarray) and
           np.can_cast(o_radiance[k], np.float64)):
            npt.assert_allclose(o_radiance[k], o_radiance2[k])
        elif(isinstance(o_radiance[k], np.ndarray)):
            assert np.all(o_radiance[k] == o_radiance2[k])
        else:
            assert o_radiance[k] == o_radiance2[k]
    npt.assert_allclose(jac_fm, jac_fm2)
    assert bad_flag == bad_flag2
    # Note that o_measured_radiance_omi, o_measured_radiance_tropomi don't
    # compare the same, because we don't bother filling that in in our
    # CostFunction.fm_wrapper. We could if it matters, but it would be a bit
    # involved to generate and isn't used by run_forward_model which is really
    # the target of our fm_wrapper.

@old_py_retrieve_test    
def test_fm_wrapper_omi(joint_omi_step_8, vlidort_cli, osp_dir):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    rs, rstep, _ = joint_omi_step_8
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    obs_airs = rs.observation_handle_set.observation(
        "AIRS", rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["AIRS"],
        None,osp_dir=osp_dir)
    obs_omi = rs.observation_handle_set.observation(
        "OMI", rs.current_state(),
        rs.current_strategy_step.spectral_window_dict["OMI"],
        None,osp_dir=osp_dir, write_omi_radiance_pickle=True)
    creator = CostFunctionCreator()
    obs_airs.spectral_window.include_bad_sample = True
    obs_omi.spectral_window.include_bad_sample = True
    cfunc = creator.cost_function_from_uip(rf_uip, [obs_airs, obs_omi], None,
                                           vlidort_cli=vlidort_cli)
    (o_radiance, jac_fm, bad_flag,
     o_measured_radiance_omi, o_measured_radiance_tropomi) = \
         cfunc.fm_wrapper(rf_uip.uip, None, {})
    with osswrapper(rf_uip.uip):
        with muses_py_call(rf_uip.run_dir, vlidort_cli=vlidort_cli):
            (o_radiance2, jac_fm2, bad_flag2,
             o_measured_radiance_omi2, o_measured_radiance_tropomi2) = \
                 mpy.fm_wrapper(rf_uip.uip, None, {})
    for k in o_radiance.keys():
        if(isinstance(o_radiance[k], np.ndarray) and
           np.can_cast(o_radiance[k], np.float64)):
            npt.assert_allclose(o_radiance[k], o_radiance2[k])
        elif(isinstance(o_radiance[k], np.ndarray)):
            assert np.all(o_radiance[k] == o_radiance2[k])
        else:
            assert o_radiance[k] == o_radiance2[k]
    npt.assert_allclose(jac_fm, jac_fm2)
    assert bad_flag == bad_flag2
    # Note that o_measured_radiance_omi, o_measured_radiance_tropomi don't
    # compare the same, because we don't bother filling that in in our
    # CostFunction.fm_wrapper. We could if it matters, but it would be a bit
    # involved to generate and isn't used by run_forward_model which is really
    # the target of our fm_wrapper.
    
@old_py_retrieve_test    
def test_residual_fm_jac_tropomi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                                 joint_tropomi_obs_step_12):
    '''Compare the results from our CostFunction with directly calling
    mpy.residual_fm_jacobian
    
    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    curdir = os.path.curdir
    rrefractor = joint_tropomi_residual_fm_jac(path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    obs_cris, obs_tropomi = joint_tropomi_obs_step_12
    # Set observation parameters to match what is in the UIP
    obs_tropomi.init([rf_uip.tropomi_params["solarshift_BAND3"],
                      rf_uip.tropomi_params["radianceshift_BAND3"],
                      rf_uip.tropomi_params["radsqueeze_BAND3"]])
    cfunc = creator.cost_function_from_uip(rf_uip, [obs_cris, obs_tropomi],
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)
    
    os.chdir(curdir)
    rmuses_py = joint_tropomi_residual_fm_jac(path="muses_py")
    # Results to compare against
    (uip2, o_residual2, o_jacobian_ret2, radiance_out2,
     o_jacobianOut2, o_stop_flag2) = rmuses_py.residual_fm_jacobian(vlidort_cli=vlidort_cli)
    assert o_stop_flag == o_stop_flag2
    # Note we put in fill values for bad samples, while muses-py actually
    # runs the forward model on all the data. It isn't clear what we want
    # here, but for now just compare good points
    gpt = radiance_out > -999
    npt.assert_allclose(radiance_out[gpt], radiance_out2[gpt])
    npt.assert_allclose(o_residual, o_residual2, atol=1e-3)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2, rtol=2e-6)
    npt.assert_allclose(o_jacobianOut[:,gpt], o_jacobianOut2[:,gpt])
    npt.assert_allclose(rrefractor.params["ret_info"]["obs_rad"],
                        rmuses_py.params["ret_info"]["obs_rad"], atol=1e-6)
    npt.assert_allclose(rrefractor.params["ret_info"]["meas_err"],
                        rmuses_py.params["ret_info"]["meas_err"],atol=1e-9)

@old_py_retrieve_test    
def test_residual_fm_jac_omi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                             joint_omi_obs_step_8):
    '''Compare the results from our CostFunction with directly calling
    mpy.residual_fm_jacobian
    
    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    curdir = os.path.curdir
    rrefractor = joint_omi_residual_fm_jac(path="refractor")
    
    # Note OMI already has a number of bad pixels, so we don't need to
    # dummy any for testing
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    obs_airs, obs_omi = joint_omi_obs_step_8
    # Set observation parameters to match what is in the UIP
    obs_omi.init([rf_uip.omi_params["nradwav_uv1"],
                  rf_uip.omi_params["nradwav_uv2"],
                  rf_uip.omi_params["odwav_uv1"],
                  rf_uip.omi_params["odwav_uv2"],
                  rf_uip.omi_params["odwav_slope_uv1"],
                  rf_uip.omi_params["odwav_slope_uv2"]])
    cfunc = creator.cost_function_from_uip(rf_uip, [obs_airs, obs_omi],
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)

    os.chdir(curdir)
    rmuses_py = joint_omi_residual_fm_jac(path="muses_py")
    # Results to compare against
    (uip2, o_residual2, o_jacobian_ret2, radiance_out2,
     o_jacobianOut2, o_stop_flag2) = rmuses_py.residual_fm_jacobian(vlidort_cli=vlidort_cli)
    assert o_stop_flag == o_stop_flag2
    # Note we put in fill values for bad samples, while muses-py actually
    # runs the forward model on all the data. It isn't clear what we want
    # here, but for now just compare good points
    gpt = radiance_out > -999
    npt.assert_allclose(radiance_out[gpt], radiance_out2[gpt])
    npt.assert_allclose(o_residual, o_residual2)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2)
    npt.assert_allclose(o_jacobianOut[:,gpt], o_jacobianOut2[:,gpt])
    npt.assert_allclose(rrefractor.params["ret_info"]["obs_rad"],
                        rmuses_py.params["ret_info"]["obs_rad"])
    npt.assert_allclose(rrefractor.params["ret_info"]["meas_err"],
                        rmuses_py.params["ret_info"]["meas_err"])
    
@old_py_retrieve_test    
def test_residual_fm_jac_omi2(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                              joint_omi_obs_step_8):
    '''Test out the CostFunction residual_fm_jacobian using our
    forward model. Note that this just tests that we can make the
    call, to debug any problems there.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    curdir = os.path.curdir
    rrefractor = joint_omi_residual_fm_jac(path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    ihandle = OmiForwardModelHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False)
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{test_base_path}/omi/in/sounding_1/Table.asc", osp_dir=osp_dir)
    flist = {'OMI' : ['UV1', 'UV2']}
    mid = MeasurementIdFile(f"{test_base_path}/omi/in/sounding_1/Measurement_ID.asc",
                            rconf, flist)
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(rf_uip, joint_omi_obs_step_8,
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)

@old_py_retrieve_test    
def test_residual_fm_jac_tropomi2(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                                 joint_tropomi_obs_step_12):
    '''Test out the CostFunction residual_fm_jacobian using our
    forward model. Note that this just tests that we can make the
    call, to debug any problems there.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    '''
    step_number = 12
    iteration = 2
    
    curdir = os.path.curdir
    rrefractor = joint_tropomi_residual_fm_jac(path="refractor")
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = CostFunctionCreator()
    ihandle = TropomiForwardModelHandle(use_pca=False, use_lrad=False,
                                      lrad_second_order=False)
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    obslist = joint_tropomi_obs_step_12
    rconf = RetrievalConfiguration.create_from_strategy_file(
        f"{joint_tropomi_test_in_dir}/Table.asc", osp_dir=osp_dir)
    flist = {'TROPOMI' : ['BAND3']}
    mid = MeasurementIdFile(f"{joint_tropomi_test_in_dir}/Measurement_ID.asc",
                            rconf, flist)
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(rf_uip, obslist,
                                           rrefractor.params["ret_info"],
                                           vlidort_cli=vlidort_cli)
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)
    
