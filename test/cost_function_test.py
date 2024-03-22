from test_support import *
from refractor.muses import (CostFunctionCreator, CostFunction, muses_py_call,
                             osswrapper)
import refractor.muses.muses_py as mpy

@require_muses_py
def test_fm_wrapper_tropomi(joint_tropomi_uip_step_12, joint_tropomi_obs_step_12, vlidort_cli):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    rf_uip = joint_tropomi_uip_step_12
    creator = CostFunctionCreator()
    cfunc = creator.cost_function_from_uip(rf_uip, joint_tropomi_obs_step_12,
                                           None, vlidort_cli=vlidort_cli,
                                           include_bad_sample=True)
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

@require_muses_py
def test_fm_wrapper_omi(joint_omi_uip_step_8, joint_omi_obs_step_8, vlidort_cli):
    rf_uip = joint_omi_uip_step_8
    # Little bit of a kludge, I think this is just needed to test against the old
    # fm_wrapper code that includes bad samples. We can worry about this if
    # it is a real issue.
    obs_airs, obs_omi = joint_omi_obs_step_8
    obs_airs.spectral_window = obs_airs.spectral_window_with_bad_sample
    obs_omi.spectral_window = obs_omi.spectral_window_with_bad_sample
    creator = CostFunctionCreator()
    cfunc = creator.cost_function_from_uip(rf_uip, [obs_airs, obs_omi], None,
                                           vlidort_cli=vlidort_cli,
                                           include_bad_sample=True)
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
    
@require_muses_py
def test_residual_fm_jac_tropomi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                                 joint_tropomi_obs_step_12):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    step_number = 12
    iteration = 2

    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
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
    rmuses_py = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                      step_number=step_number,
                                      iteration=iteration,
                                      osp_dir=osp_dir,
                                      gmao_dir=gmao_dir,
                                      path="muses_py")
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

@require_muses_py
def test_residual_fm_jac_omi(isolated_dir, vlidort_cli, osp_dir, gmao_dir,
                             joint_omi_obs_step_8):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    step_number = 8
    iteration = 2
    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
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
    rmuses_py = muses_residual_fm_jac(joint_omi_test_in_dir,
                                      step_number=step_number,
                                      iteration=iteration,
                                      osp_dir=osp_dir,
                                      gmao_dir=gmao_dir,
                                      path="muses_py")
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
    
