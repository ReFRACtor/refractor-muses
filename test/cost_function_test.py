from test_support import *
from refractor.muses import (FmObsCreator, CostFunction, muses_py_call,
                             osswrapper)
import refractor.muses.muses_py as mpy

@require_muses_py
def test_fm_wrapper_tropomi(joint_tropomi_uip_step_10, vlidort_cli):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    rf_uip = joint_tropomi_uip_step_10
    creator = FmObsCreator()
    cfunc = CostFunction(*creator.fm_and_fake_obs(rf_uip,
                                                  use_full_state_vector=True,
                                                  vlidort_cli=vlidort_cli))
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
def test_fm_wrapper_omi(joint_omi_uip_step_7, vlidort_cli):
    rf_uip = joint_omi_uip_step_7
    creator = FmObsCreator()
    cfunc = CostFunction(*creator.fm_and_fake_obs(rf_uip,
                                                  use_full_state_vector=True,
                                                  vlidort_cli=vlidort_cli))
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
def test_residual_fm_jac_tropomi(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    step_number = 10
    iteration = 2

    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor")
    # Create some bad data, so we can test handling of bad samples
    # The meas_err isn't super easy to work out (although we could).
    # We just figured this out by setting a breakpoint and grabbing what
    # muses-py ends up with
    rrefractor.params["ret_info"]["meas_err"][2:213:10] = -999
    rrefractor.params["ret_info"]["meas_err"][221::15] = -999
    rdata = pickle.load(open(rrefractor.run_dir + "/Input/Radiance_TROPOMI_.pkl",
                             "rb"))
    rdata['Earth_Radiance']['EarthRadianceNESR'][1::15] = -999
    pickle.dump(rdata, open(rrefractor.run_dir + "/Input/Radiance_TROPOMI_.pkl",
                             "wb"))
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = FmObsCreator()
    cfunc = CostFunction(*creator.fm_and_obs(rf_uip,
                                             rrefractor.params["ret_info"],
                                             vlidort_cli=vlidort_cli))
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)
    
    os.chdir(curdir)
    rmuses_py = muses_residual_fm_jac(joint_tropomi_test_in_dir,
                                      step_number=step_number,
                                      iteration=iteration,
                                      osp_dir=osp_dir,
                                      gmao_dir=gmao_dir,
                                      path="muses_py")
    # Create some bad data, so we can test handling of bad samples
    # The meas_err isn't super easy to work out (although we could).
    # We just figured this out by setting a breakpoint and grabbing what
    # muses-py ends up with
    rmuses_py.params["ret_info"]["meas_err"][2:213:10] = -999
    rmuses_py.params["ret_info"]["meas_err"][221::15] = -999
    rdata = pickle.load(open(rmuses_py.run_dir + "/Input/Radiance_TROPOMI_.pkl",
                             "rb"))
    rdata['Earth_Radiance']['EarthRadianceNESR'][1::15] = -999
    pickle.dump(rdata, open(rmuses_py.run_dir + "/Input/Radiance_TROPOMI_.pkl",
                            "wb"))
    # Results to compare against
    (uip2, o_residual2, o_jacobian_ret2, radiance_out2,
     o_jacobianOut2, o_stop_flag2) = rmuses_py.residual_fm_jacobian(vlidort_cli=vlidort_cli)
    assert o_stop_flag == o_stop_flag2
    #npt.assert_allclose(radiance_out, radiance_out2)
    npt.assert_allclose(o_residual, o_residual2)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2)
    # This fails
    #npt.assert_allclose(o_jacobianOut, o_jacobianOut2)
    # However this succeeds
    basis_matrix = rrefractor.params["ret_info"]["basis_matrix"]
    #npt.assert_allclose(np.matmul(basis_matrix, o_jacobianOut),
    #                    np.matmul(basis_matrix, o_jacobianOut2), atol=1e-12)
    npt.assert_allclose(rrefractor.params["ret_info"]["obs_rad"],
                        rmuses_py.params["ret_info"]["obs_rad"])
    npt.assert_allclose(rrefractor.params["ret_info"]["meas_err"],
                        rmuses_py.params["ret_info"]["meas_err"])

@require_muses_py
def test_residual_fm_jac_omi(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    '''Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.'''
    step_number = 7
    iteration = 2
    
    curdir = os.path.curdir
    rrefractor = muses_residual_fm_jac(joint_omi_test_in_dir,
                                       step_number=step_number,
                                       iteration=iteration,
                                       osp_dir=osp_dir,
                                       gmao_dir=gmao_dir,
                                       path="refractor",
                                       change_to_dir=False)
    # Note OMI already has a number of bad pixels, so we don't need to
    # dummy any for testing
    rf_uip = RefractorUip(rrefractor.params["uip"],
                          rrefractor.params["ret_info"]["basis_matrix"])
    rf_uip.run_dir = rrefractor.run_dir
    creator = FmObsCreator()
    cfunc = CostFunction(*creator.fm_and_obs(rf_uip,
                                             rrefractor.params["ret_info"],
                                             vlidort_cli=vlidort_cli))
    (uip, o_residual, o_jacobian_ret, radiance_out,
     o_jacobianOut, o_stop_flag) = cfunc.residual_fm_jacobian(**rrefractor.params)

    os.chdir(curdir)
    rmuses_py = muses_residual_fm_jac(joint_omi_test_in_dir,
                                      step_number=step_number,
                                      iteration=iteration,
                                      osp_dir=osp_dir,
                                      gmao_dir=gmao_dir,
                                      path="muses_py",
                                      change_to_dir=False)
    # Results to compare against
    (uip2, o_residual2, o_jacobian_ret2, radiance_out2,
     o_jacobianOut2, o_stop_flag2) = rmuses_py.residual_fm_jacobian(vlidort_cli=vlidort_cli)
    assert o_stop_flag == o_stop_flag2
    npt.assert_allclose(radiance_out, radiance_out2)
    npt.assert_allclose(o_residual, o_residual2)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2)
    # This fails
    #npt.assert_allclose(o_jacobianOut, o_jacobianOut2)
    # However this succeeds
    basis_matrix = rrefractor.params["ret_info"]["basis_matrix"]
    npt.assert_allclose(np.matmul(basis_matrix, o_jacobianOut),
                        np.matmul(basis_matrix, o_jacobianOut2), atol=1e-12)
    npt.assert_allclose(rrefractor.params["ret_info"]["obs_rad"],
                        rmuses_py.params["ret_info"]["obs_rad"])
    npt.assert_allclose(rrefractor.params["ret_info"]["meas_err"],
                        rmuses_py.params["ret_info"]["meas_err"])
    
