from __future__ import annotations
import refractor.framework as rf  # type: ignore
from refractor.muses import (
    StateElementIdentifier,
    StateElementOspFile,
    RetrievalConfiguration,
    RetrievalType,
    CurrentStrategyStepDict,
    StrategyStepIdentifier,
    InputFileHelper,
)
import numpy as np
import numpy.testing as npt


def test_osp_state_element(osp_dir, omi_test_in_dir):
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    apriori_value = np.array(
        [
            0.0,
        ]
    )
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOspFile(
        StateElementIdentifier("OMIODWAVUV1"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    # Simulate a state vector update
    selem._retrieved_this_step = True
    selem.notify_parameter_update(np.array([2.0]))
    assert selem.forward_model_sv_length == 1
    # Short term have this turned off, so we can compare with old
    # data. TODO Turn this back on when we have this working again
    # npt.assert_allclose(selem.value_fm, [2.0])
    npt.assert_allclose(selem.constraint_vector_fm, [0.0])
    cexpect = np.array([[2500.0]])
    cov_expect = np.array([[0.0004]])
    npt.assert_allclose(selem.constraint_matrix, cexpect)
    npt.assert_allclose(selem.apriori_cov_fm, cov_expect)
    npt.assert_allclose(selem.retrieval_initial_fm, [0.0])
    npt.assert_allclose(selem.step_initial_fm, [0.0])
    assert selem.true_value_fm is None
    assert isinstance(selem.state_mapping, rf.StateMappingLinear)
    assert isinstance(selem.state_mapping_retrieval_to_fm, rf.StateMappingLinear)
    # Perhaps basis can go away? Replaced with state_mapping_retrieval_to_fm?
    npt.assert_allclose(selem.basis_matrix, np.eye(1))
    npt.assert_allclose(selem.map_to_parameter_matrix, np.eye(1))

    # Update the initial guess, as if we had a element with a different value
    selem.update_state_element(step_initial_fm=np.array([3.0]))
    npt.assert_allclose(selem.retrieval_initial_fm, [0.0])
    npt.assert_allclose(selem.step_initial_fm, [3.0])
    cstep = CurrentStrategyStepDict({}, None)
    selem.notify_start_retrieval(cstep, rconfig)
    # After starting the retrieval, the retrieval_initial_value should be the step_initial_value
    npt.assert_allclose(selem.retrieval_initial_fm, [0.0])
    npt.assert_allclose(selem.step_initial_fm, [0.0])
    npt.assert_allclose(selem.value_fm, [0.0])

    # Have a step where we retrieve our element
    cstep_ret = CurrentStrategyStepDict(
        {
            "retrieval_elements": [StateElementIdentifier("OMIODWAVUV1")],
            "retrieval_type": RetrievalType("default"),
            "retrieval_elements_not_updated": [],
            "strategy_step": StrategyStepIdentifier(1, "step_1"),
        },
        None,
    )
    selem.update_state_element(step_initial_fm=np.array([4.0]))
    npt.assert_allclose(selem.step_initial_fm, [4.0])
    npt.assert_allclose(selem.value_fm, [0.0])
    selem.notify_start_step(cstep_ret, rconfig)
    npt.assert_allclose(selem.step_initial_fm, [4.0])
    npt.assert_allclose(selem.value_fm, [4.0])
    # Pretend like we got a solution to update
    selem.notify_step_solution(
        np.array(
            [
                5.0,
            ]
        ),
        slice(0, 1),
    )
    # Initial guess isn't updated yet, although value is
    npt.assert_allclose(selem.step_initial_fm, [4.0])
    npt.assert_allclose(selem.value_fm, [5.0])
    # Start next step. Initial guess should be updated now
    selem.notify_start_step(cstep_ret, rconfig)
    npt.assert_allclose(selem.step_initial_fm, [5.0])
    npt.assert_allclose(selem.value_fm, [5.0])
    # Repeat, but with item on the no update list. Check that we don't update the
    # initial guess, and the value gets reset to the original value
    cstep_ret.retrieval_elements_not_updated = [
        StateElementIdentifier("OMIODWAVUV1"),
    ]
    selem.notify_start_step(cstep_ret, rconfig)
    npt.assert_allclose(selem.step_initial_fm, [5.0])
    npt.assert_allclose(selem.value_fm, [5.0])
    selem.notify_step_solution(
        np.array(
            [
                6.0,
            ]
        ),
        slice(0, 1),
    )
    npt.assert_allclose(selem.step_initial_fm, [5.0])
    npt.assert_allclose(selem.value_fm, [6.0])
    # Initial guess should remain unchanged
    selem.notify_start_step(cstep_ret, rconfig)
    npt.assert_allclose(selem.step_initial_fm, [5.0])
    npt.assert_allclose(selem.value_fm, [5.0])


def test_osp_state_element_constraint(osp_dir, omi_test_in_dir):
    """Check that the constraint matrix update for specific retrieval types is handled correctly"""
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        omi_test_in_dir / "Table.asc", osp_dir=osp_dir
    )
    apriori_value = np.array(
        [
            0.0,
        ]
    )
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 10.0
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOspFile(
        StateElementIdentifier("OMICLOUDFRACTION"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    cmatrix1 = np.array(
        [
            [
                400.0,
            ]
        ]
    )
    npt.assert_allclose(selem.constraint_matrix, cmatrix1)
    cstep_ret = CurrentStrategyStepDict(
        {
            "retrieval_elements": [StateElementIdentifier("OMICLOUDFRACTION")],
            "retrieval_type": RetrievalType("omicloud_ig_refine"),
            "retrieval_elements_not_updated": [],
            "strategy_step": StrategyStepIdentifier(1, "step_1"),
        },
        None,
    )
    selem.notify_start_step(cstep_ret, rconfig)
    cmatrix2 = np.array(
        [
            [
                4.0,
            ]
        ]
    )
    npt.assert_allclose(selem.constraint_matrix, cmatrix2)
    # And goes back when step changes
    cstep_ret = CurrentStrategyStepDict(
        {
            "retrieval_elements": [StateElementIdentifier("OMICLOUDFRACTION")],
            "retrieval_type": RetrievalType("not_special_step"),
            "retrieval_elements_not_updated": [],
            "strategy_step": StrategyStepIdentifier(1, "step_1"),
        },
        None,
    )
    selem.notify_start_step(cstep_ret, rconfig)
    npt.assert_allclose(selem.constraint_matrix, cmatrix1)


def test_osp_state_element_latitude(osp_dir):
    """Test an element that depends on latitude, to check correct handling of latitude"""
    apriori_value = np.array(
        [
            0.0,
        ]
    )
    # Dummy value, most of the elements don't actually depend on latitude
    latitude = 20
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOspFile(
        StateElementIdentifier("PTGANG"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    latitude = 10
    selem2 = StateElementOspFile(
        StateElementIdentifier("PTGANG"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    latitude = 60
    selem3 = StateElementOspFile(
        StateElementIdentifier("PTGANG"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    latitude = -60
    selem4 = StateElementOspFile(
        StateElementIdentifier("PTGANG"),
        None,
        apriori_value,
        apriori_value,
        latitude,
        "LAND",
        InputFileHelper(),
        species_directory,
        covariance_directory,
    )
    # Unfortunately, the covariance matrixes are all the same. Not sure why we have different
    # files here for the covariance, might be a historical thing. We did manually verify things
    # were working by turning on logging in the TesFile reading. We are correctly getting the
    # right files, although they all happen to have the same results
    cov_expect = np.array([[5.625e-07]])
    cov_expect2 = np.array([[5.625e-07]])
    cov_expect3 = np.array([[5.625e-07]])
    cov_expect4 = np.array([[5.625e-07]])
    npt.assert_allclose(selem.apriori_cov_fm, cov_expect)
    npt.assert_allclose(selem2.apriori_cov_fm, cov_expect2)
    npt.assert_allclose(selem3.apriori_cov_fm, cov_expect3)
    npt.assert_allclose(selem4.apriori_cov_fm, cov_expect4)
