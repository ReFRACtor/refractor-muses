from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementEmis,
    StateElementNativeEmis,
    StateElementCloudExt,
)
import numpy.testing as npt


def test_state_element_emis(cris_tropomi_old_shandle):
    h_old, _, rconfig, strat, _, smeta, sinfo = cris_tropomi_old_shandle
    sold = h_old.state_element(StateElementIdentifier("EMIS"))
    sold_value_fm = sold.value_fm
    # This is value_fm before we have cycled through all the strategy
    # steps
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementEmis.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    # Check that we match before cycling through the strategy steps
    npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Also check spectral domain
    npt.assert_allclose(s.spectral_domain.data, sold.spectral_domain.data)
    # And metadata
    assert s.metadata == sold.metadata


def test_state_element_native_emis(cris_tropomi_old_shandle):
    h_old, _, rconfig, strat, _, smeta, sinfo = cris_tropomi_old_shandle
    sold = h_old.state_element(StateElementIdentifier("native_emissivity"))
    sold_value_fm = sold.value_fm
    # This is value_fm before we have cycled through all the strategy
    # steps
    # sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementNativeEmis.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    # Check that we match before cycling through the strategy steps
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Also check spectral domain
    npt.assert_allclose(s.spectral_domain.data, sold.spectral_domain.data)


def test_state_element_cloudext(airs_omi_old_shandle):
    h_old, _, rconfig, strat, _, smeta, sinfo = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier("CLOUDEXT"))
    sold_value_fm = sold.value_fm[0, :]
    # This is value_fm before we have cycled through all the strategy
    # steps
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementCloudExt.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    # Check that we match before cycling through the strategy steps
    npt.assert_allclose(s.value_fm, sold_constraint_vector_fm, atol=1e-20)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.value_fm, sold_value_fm, atol=1e-20)
    # Also check spectral domain
    npt.assert_allclose(s.spectral_domain.data, sold.spectral_domain.data)
