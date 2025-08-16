from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementPcloud,
    StateElementFromSingle,
)
import numpy.testing as npt
import pytest


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


@pytest.mark.parametrize("sid", ("SO2", "NH3", "OCS", "HCOOH", "N2"))
def test_state_element_from_single(airs_omi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    # This is value_fm before we have cycled through all the strategy
    # steps
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromSingle.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
    )
    # NH3 has separate logic to override the value in some cases. Skip checking
    # against sold - we don't agree but that is ok because this particular handle
    # doesn't get used in this test case
    if sid != "NH3":
        npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    if sid != "NH3":
        npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.metadata == {}
