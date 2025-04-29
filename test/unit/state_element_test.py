from __future__ import annotations
import refractor.framework as rf  # type: ignore
from refractor.muses import StateElementIdentifier, StateElementOspFile
import numpy as np
import numpy.testing as npt


def test_osp_state_element(osp_dir):
    apriori_value = np.array(
        [
            0.0,
        ]
    )
    latitude = (
        10.0  # Dummy value, most of the elements don't actually depend on latitude
    )
    species_directory = (
        osp_dir / "Strategy_Tables" / "ops" / "OSP-OMI-AIRS-v10" / "Species-66"
    )
    covariance_directory = osp_dir / "Covariance" / "Covariance"
    selem = StateElementOspFile(
        StateElementIdentifier("OMIODWAVUV1"),
        apriori_value,
        latitude,
        species_directory,
        covariance_directory,
    )
    # Simulate a state vector update
    selem.notify_parameter_update(np.array([2.0]))
    assert selem.retrieval_sv_length == 1
    assert selem.sys_sv_length == 1
    assert selem.forward_model_sv_length == 1
    npt.assert_allclose(selem.value, [2.0])
    npt.assert_allclose(selem.value_fm, [2.0])
    npt.assert_allclose(selem.apriori_value, [0.0])
    npt.assert_allclose(selem.apriori_value_fm, [0.0])
    cexpect = np.array([[2500.0]])
    cov_expect = np.array([[0.0004]])
    npt.assert_allclose(selem.constraint_matrix, cexpect)
    npt.assert_allclose(selem.apriori_cov_fm, cov_expect)
    npt.assert_allclose(selem.retrieval_initial_value, [0.0])
    npt.assert_allclose(selem.step_initial_value, [0.0])
    npt.assert_allclose(selem.step_initial_value_fm, [0.0])
    assert selem.true_value is None
    assert selem.true_value_fm is None
    assert isinstance(selem.state_mapping, rf.StateMappingLinear)
    assert isinstance(selem.state_mapping_retrieval_to_fm, rf.StateMappingLinear)
    # Perhaps basis can go away? Replaced with state_mapping_retrieval_to_fm?
    npt.assert_allclose(selem.basis_matrix, np.eye(1))
    npt.assert_allclose(selem.map_to_parameter_matrix, np.eye(1))
    # notify_start_retrieval
    # notify_new_step
    # Can update_state go away? Perhaps notify_solution?
