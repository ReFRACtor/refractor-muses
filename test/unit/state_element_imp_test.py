from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementPcloud,
)
import numpy.testing as npt


def test_state_element_pcloud(airs_omi_old_shandle):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier("PCLOUD"))
    sold_value_fm = sold.value_fm
    # This is value_fm before we have cycled through all the strategy
    # steps
    sold_constraint_vector_fm = sold.constraint_vector_fm
    npt.assert_allclose(sold_value_fm, sold_constraint_vector_fm)
    s = StateElementPcloud.create(retrieval_config=rconfig, sounding_metadata=smeta)
    npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Cycle through strategy steps, and check value_fm after that
    # strat.retrieval_initial_fm_from_cycle(s, rconfig)
    # npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check an number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.diag_cov
    assert s.metadata == {}
