from refractor.muses import (
    CostFunctionCreator,
    MeasurementIdFile,
    RetrievalConfiguration,
    InstrumentIdentifier,
)
from refractor.muses_py_fm import RefractorUip, osswrapper, muses_py_call
from fixtures.residual_fm import (
    joint_omi_residual_fm_jac,
    joint_tropomi_residual_fm_jac,
)
from fixtures.require_check import require_muses_py
from refractor.omi import OmiForwardModelHandle
from refractor.tropomi import TropomiForwardModelHandle
import refractor.muses_py as mpy
import pytest
import numpy as np
import numpy.testing as npt

# The CostFunction is definitely a central object in the ReFRACtor
# code. However these tests check support for the old py-retrieve
# functions that we used in the initial development of the ReFRACtor
# code where we compared against py-retrieve retrievals. Like all the
# old_py_retrieve tests this may outlive its usefulness at some point
# and not be worth maintaining. But for now, check that we still have
# this functionality working.


# Not working yet
# @pytest.mark.skip
# def test_tropomi_pickle(isolated_dir, osp_dir):
#     rs, rstep, _ = set_up_run_to_location(
#         joint_tropomi_test_in_dir, 12, "done get_initial_guess"
#     )
#     os.chdir(rs.run_dir)
#     ihandle = TropomiForwardModelHandle(
#         use_pca=False, use_lrad=False, lrad_second_order=False
#     )
#     rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
#     rs.update_target(f"{rs.run_dir}/Table.asc")
#     cfunc = rs.create_cost_function()
#     rf.write_shelve("cfunc.xml", cfunc)
#     cfunc2 = rf.read_shelve("cfunc.xml")


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_fm_wrapper_tropomi(joint_tropomi_step_12_osp_sym_link, osp_dir):
    """Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rs, rstep, _ = joint_tropomi_step_12_osp_sym_link
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
        osp_dir=osp_dir,
    )
    obs_tropomi = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
        osp_dir=osp_dir,
        write_tropomi_radiance_pickle=True,
    )
    obs_cris.spectral_window.include_bad_sample = True
    obs_tropomi.spectral_window.include_bad_sample = True
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_cris, obs_tropomi],
        rs.current_state,
        rs.retrieval_config,
    )
    cfunc = rs.cost_function_creator.cost_function_from_uip(
        rf_uip, [obs_cris, obs_tropomi], None
    )
    (
        o_radiance,
        jac_fm,
        bad_flag,
        o_measured_radiance_omi,
        o_measured_radiance_tropomi,
    ) = cfunc.fm_wrapper(rf_uip.uip, None, {})
    with muses_py_call(rf_uip.run_dir):
        with osswrapper(rf_uip.uip):
            (
                o_radiance2,
                jac_fm2,
                bad_flag2,
                o_measured_radiance_omi2,
                o_measured_radiance_tropomi2,
            ) = mpy.fm_wrapper(rf_uip.uip, None, {})
    for k in o_radiance.keys():
        if isinstance(o_radiance[k], np.ndarray) and np.can_cast(
            o_radiance[k], np.float64
        ):
            npt.assert_allclose(o_radiance[k], o_radiance2[k])
        elif isinstance(o_radiance[k], np.ndarray):
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


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_fm_wrapper_omi(joint_omi_step_8_osp_sym_link, osp_dir):
    """Compare the results from our CostFunction with directly calling
    mpy.fm_wrapper.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rs, rstep, _ = joint_omi_step_8_osp_sym_link
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
        osp_dir=osp_dir,
    )
    obs_omi = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
        osp_dir=osp_dir,
        write_omi_radiance_pickle=True,
    )
    obs_airs.spectral_window.include_bad_sample = True
    obs_omi.spectral_window.include_bad_sample = True
    rf_uip = RefractorUip.create_uip_from_refractor_objects(
        [obs_airs, obs_omi],
        rs.current_state,
        rs.retrieval_config,
    )
    cfunc = rs.cost_function_creator.cost_function_from_uip(
        rf_uip, [obs_airs, obs_omi], None
    )
    (
        o_radiance,
        jac_fm,
        bad_flag,
        o_measured_radiance_omi,
        o_measured_radiance_tropomi,
    ) = cfunc.fm_wrapper(rf_uip.uip, None, {})
    with muses_py_call(rf_uip.run_dir):
        with osswrapper(rf_uip.uip):
            (
                o_radiance2,
                jac_fm2,
                bad_flag2,
                o_measured_radiance_omi2,
                o_measured_radiance_tropomi2,
            ) = mpy.fm_wrapper(rf_uip.uip, None, {})
    for k in o_radiance.keys():
        if isinstance(o_radiance[k], np.ndarray) and np.can_cast(
            o_radiance[k], np.float64
        ):
            npt.assert_allclose(o_radiance[k], o_radiance2[k])
        elif isinstance(o_radiance[k], np.ndarray):
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


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_residual_fm_jac_tropomi(
    isolated_dir,
    osp_dir,
    gmao_dir,
    joint_tropomi_obs_step_12,
    joint_tropomi_test_in_dir,
):
    """Compare the results from our CostFunction with directly calling
    mpy.residual_fm_jacobian

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rrefractor = joint_tropomi_residual_fm_jac(
        osp_dir, gmao_dir, joint_tropomi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    obs_cris, obs_tropomi = joint_tropomi_obs_step_12
    # Set observation parameters to match what is in the UIP
    obs_tropomi.init(
        [
            rf_uip.tropomi_params["solarshift_BAND3"],
            rf_uip.tropomi_params["radianceshift_BAND3"],
            rf_uip.tropomi_params["radsqueeze_BAND3"],
        ]
    )
    creator = CostFunctionCreator()
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        rf_uip.run_dir / "Table.asc", osp_dir=osp_dir
    )
    mid = MeasurementIdFile(
        rf_uip.run_dir / "Measurement_ID.asc", rconfig, {"TROPOMI": ["BAND3"]}
    )
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(
        rf_uip,
        [obs_cris, obs_tropomi],
        rrefractor.params["ret_info"],
    )
    (uip, o_residual, o_jacobian_ret, radiance_out, o_jacobianOut, o_stop_flag) = (
        cfunc.residual_fm_jacobian(**rrefractor.params)
    )

    rmuses_py = joint_tropomi_residual_fm_jac(
        osp_dir, gmao_dir, joint_tropomi_test_in_dir, path="muses_py"
    )
    # Results to compare against
    (
        uip2,
        o_residual2,
        o_jacobian_ret2,
        radiance_out2,
        o_jacobianOut2,
        o_stop_flag2,
    ) = rmuses_py.residual_fm_jacobian()
    assert o_stop_flag == o_stop_flag2
    # Note we put in fill values for bad samples, while muses-py actually
    # runs the forward model on all the data. It isn't clear what we want
    # here, but for now just compare good points
    gpt = radiance_out > -999
    npt.assert_allclose(radiance_out[gpt], radiance_out2[gpt])
    npt.assert_allclose(o_residual, o_residual2, atol=2e-2)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2, rtol=3e-6)
    npt.assert_allclose(o_jacobianOut[:, gpt], o_jacobianOut2[:, gpt])
    npt.assert_allclose(
        rrefractor.params["ret_info"]["obs_rad"],
        rmuses_py.params["ret_info"]["obs_rad"],
        atol=1e-6,
    )
    npt.assert_allclose(
        rrefractor.params["ret_info"]["meas_err"],
        rmuses_py.params["ret_info"]["meas_err"],
        atol=1e-9,
    )


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_residual_fm_jac_omi(
    isolated_dir,
    osp_dir,
    gmao_dir,
    joint_omi_obs_step_8,
    joint_omi_test_in_dir,
):
    """Compare the results from our CostFunction with directly calling
    mpy.residual_fm_jacobian

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rrefractor = joint_omi_residual_fm_jac(
        osp_dir, gmao_dir, joint_omi_test_in_dir, path="refractor"
    )

    # Note OMI already has a number of bad pixels, so we don't need to
    # dummy any for testing
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    obs_airs, obs_omi = joint_omi_obs_step_8
    # Set observation parameters to match what is in the UIP
    obs_omi.init(
        [
            rf_uip.omi_params["nradwav_uv1"],
            rf_uip.omi_params["nradwav_uv2"],
            rf_uip.omi_params["odwav_uv1"],
            rf_uip.omi_params["odwav_uv2"],
            rf_uip.omi_params["odwav_slope_uv1"],
            rf_uip.omi_params["odwav_slope_uv2"],
        ]
    )
    creator = CostFunctionCreator()
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        rf_uip.run_dir / "Table.asc", osp_dir=osp_dir
    )
    mid = MeasurementIdFile(
        rf_uip.run_dir / "Measurement_ID.asc", rconfig, {"TROPOMI": ["BAND3"]}
    )
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(
        rf_uip,
        [obs_airs, obs_omi],
        rrefractor.params["ret_info"],
    )
    (uip, o_residual, o_jacobian_ret, radiance_out, o_jacobianOut, o_stop_flag) = (
        cfunc.residual_fm_jacobian(**rrefractor.params)
    )

    rmuses_py = joint_omi_residual_fm_jac(
        osp_dir, gmao_dir, joint_omi_test_in_dir, path="muses_py"
    )
    # Results to compare against
    (
        uip2,
        o_residual2,
        o_jacobian_ret2,
        radiance_out2,
        o_jacobianOut2,
        o_stop_flag2,
    ) = rmuses_py.residual_fm_jacobian()
    assert o_stop_flag == o_stop_flag2
    # Note we put in fill values for bad samples, while muses-py actually
    # runs the forward model on all the data. It isn't clear what we want
    # here, but for now just compare good points
    gpt = radiance_out > -999
    npt.assert_allclose(radiance_out[gpt], radiance_out2[gpt])
    npt.assert_allclose(o_residual, o_residual2)
    npt.assert_allclose(o_jacobian_ret, o_jacobian_ret2)
    npt.assert_allclose(o_jacobianOut[:, gpt], o_jacobianOut2[:, gpt])
    npt.assert_allclose(
        rrefractor.params["ret_info"]["obs_rad"],
        rmuses_py.params["ret_info"]["obs_rad"],
    )
    npt.assert_allclose(
        rrefractor.params["ret_info"]["meas_err"],
        rmuses_py.params["ret_info"]["meas_err"],
    )


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_residual_fm_jac_omi2(
    isolated_dir,
    osp_dir,
    gmao_dir,
    joint_omi_obs_step_8,
    joint_omi_test_in_dir,
):
    """Test out the CostFunction residual_fm_jacobian using our
    forward model. Note that this just tests that we can make the
    call, to debug any problems there.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rrefractor = joint_omi_residual_fm_jac(
        osp_dir, gmao_dir, joint_omi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    ihandle = OmiForwardModelHandle(
        use_pca=False, use_lrad=False, lrad_second_order=False
    )
    creator = CostFunctionCreator()
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rconf = RetrievalConfiguration.create_from_strategy_file(
        joint_omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    flist = {"OMI": ["UV1", "UV2"]}
    mid = MeasurementIdFile(joint_omi_test_in_dir / "Measurement_ID.asc", rconf, flist)
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(
        rf_uip,
        joint_omi_obs_step_8,
        rrefractor.params["ret_info"],
    )
    (uip, o_residual, o_jacobian_ret, radiance_out, o_jacobianOut, o_stop_flag) = (
        cfunc.residual_fm_jacobian(**rrefractor.params)
    )


@pytest.mark.long_test
@pytest.mark.old_py_retrieve_test
@require_muses_py
def test_residual_fm_jac_tropomi2(
    isolated_dir,
    osp_dir,
    gmao_dir,
    joint_tropomi_obs_step_12,
    joint_tropomi_test_in_dir,
):
    """Test out the CostFunction residual_fm_jacobian using our
    forward model. Note that this just tests that we can make the
    call, to debug any problems there.

    This is the old py-retrieve function. We don't actually use this
    anymore, but it is useful to make sure the old function works in case
    we need to use this in the future to track down some problem.

    We can probably eventually remove this - at some point it may be more
    work to maintain this old compatibility function than it is worth.
    """
    rrefractor = joint_tropomi_residual_fm_jac(
        osp_dir, gmao_dir, joint_tropomi_test_in_dir, path="refractor"
    )
    rf_uip = RefractorUip(
        rrefractor.params["uip"], rrefractor.params["ret_info"]["basis_matrix"]
    )
    rf_uip.run_dir = rrefractor.run_dir
    ihandle = TropomiForwardModelHandle(
        use_pca=False, use_lrad=False, lrad_second_order=False
    )
    creator = CostFunctionCreator()
    creator.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    obslist = joint_tropomi_obs_step_12
    rconf = RetrievalConfiguration.create_from_strategy_file(
        joint_tropomi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    flist = {"TROPOMI": ["BAND3"]}
    mid = MeasurementIdFile(
        joint_tropomi_test_in_dir / "Measurement_ID.asc", rconf, flist
    )
    creator.notify_update_target(mid)
    cfunc = creator.cost_function_from_uip(
        rf_uip, obslist, rrefractor.params["ret_info"]
    )
    (uip, o_residual, o_jacobian_ret, radiance_out, o_jacobianOut, o_stop_flag) = (
        cfunc.residual_fm_jacobian(**rrefractor.params)
    )
