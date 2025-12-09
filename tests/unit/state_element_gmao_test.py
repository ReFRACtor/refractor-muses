from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElement,
    StateElementFromGmaoPressure,
    StateElementFromGmaoPsur,
    StateElementFromGmaoTropopausePressure,
    StateElementFromGmaoTatm,
    StateElementFromGmaoH2O,
    StateElementFromGmaoTsur,
)
import pickle
import numpy.testing as npt
from pathlib import Path


def check_selem(selem: StateElement, fexpect: Path, save: bool = False) -> None:
    # We validated the results against the old state elements from muses-py.
    # Remove that so we don't depend on having muses-py available, but we want to
    # know if the value has changed indicating a possible problem.
    if save:
        pickle.dump(
            {
                "value_fm": selem.value_fm,
                "constraint_vector_fm": selem.constraint_vector_fm,
            },
            open(fexpect, "wb"),
        )
    expected = pickle.load(open(fexpect, "rb"))
    npt.assert_allclose(selem.constraint_vector_fm, expected["constraint_vector_fm"])
    npt.assert_allclose(selem.value_fm, expected["value_fm"])


def test_state_element_from_gmao_pressure(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "pressure"
    s = StateElementFromGmaoPressure.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_gmao" / "pressure_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tropopause_pressure(
    airs_omi_shandle, unit_test_expected_dir
):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "gmaoTropopausePressure"
    s = StateElementFromGmaoTropopausePressure.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir
        / "state_element_gmao"
        / "tropopause_pressure_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tatm(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "TATM"
    s = StateElementFromGmaoTatm.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s, unit_test_expected_dir / "state_element_gmao" / "tatm_expect.pkl", save=False
    )

    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_h2o(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "H2O"
    s = StateElementFromGmaoH2O.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s, unit_test_expected_dir / "state_element_gmao" / "h2o_expect.pkl", save=False
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_tsur(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "TSUR"
    s = StateElementFromGmaoTsur.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s, unit_test_expected_dir / "state_element_gmao" / "tsur_expect.pkl", save=False
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_gmao_psur(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "PSUR"
    s = StateElementFromGmaoPsur.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s, unit_test_expected_dir / "state_element_gmao" / "psur_expect.pkl", save=False
    )

    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert list(s.metadata.keys()) == ["gmao_data"]
    assert s.poltype_used_constraint
