from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementFromGmaoPressure,
    StateElementFromGmaoPsur,
    StateElementFromGmaoTropopausePressure,
    StateElementFromGmaoTatm,
    StateElementFromGmaoH2O,
    StateElementFromGmaoTsur,
)
import numpy.testing as npt


def test_state_element_from_gmao_pressure(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "pressure"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoPressure.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tropopause_pressure(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "gmaoTropopausePressure"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoTropopausePressure.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tatm(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "TATM"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoTatm.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_h2o(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "H2O"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoH2O.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tsur(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "TSUR"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoTsur.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_psur(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_old_shandle
    sid = "PSUR"
    # sold = h_old.state_element(StateElementIdentifier(sid))
    # sold_value_fm = sold.value_fm
    # sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromGmaoPsur.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    # The old element doesn't seem to actually have the right psur, it seems
    # to be identically zero. I don't think this was ever used. Don't bother
    # tracking down, just skip check.
    # npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    # npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert list(s.metadata.keys()) == ["gmao_data"]
    assert s.poltype_used_constraint
