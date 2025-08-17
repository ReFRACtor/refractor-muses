from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementFromClimatology,
    StateElementFromClimatologyHdo,
    StateElementFromClimatologyCh3oh,
    StateInfo,
)
import numpy.testing as npt
from pathlib import Path
import pytest


def test_read_climatology_2022(airs_omi_old_shandle):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    p = h_old.state_element(StateElementIdentifier("pressure"))
    panvmr = h_old.state_element(StateElementIdentifier("PAN"))
    vmr, type_name = StateElementFromClimatology.read_climatology_2022(
        StateElementIdentifier("PAN"),
        p.value_fm,
        False,
        Path(rconfig.osp_dir) / "Climatology/Climatology_files",
        smeta,
    )
    vmr_prior, type_name = StateElementFromClimatology.read_climatology_2022(
        StateElementIdentifier("PAN"),
        p.value_fm,
        True,
        Path(rconfig.osp_dir) / "Climatology/Climatology_files",
        smeta,
    )
    npt.assert_allclose(panvmr.value_fm, vmr)
    npt.assert_allclose(panvmr.constraint_vector_fm, vmr_prior)


@pytest.mark.parametrize(
    "sid",
    [
        "CO",
        "CO2",
        "HNO3",
        "CFC12",
        "CCL4",
        "CFC22",
        "N2O",
        "O3",
        "CH4",
        "SF6",
        "C2H4",
        "PAN",
        "HCN",
        "CFC11",
    ],
)
def test_state_element_from_climatology(airs_omi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    # sold.constraint_vector_fm isn't always the fm contraint vector, sometime is
    # is the fmprime version. This is just an oddity of the old state element, all that
    # actually matters is constraint_vector_ret, but that isn't always defined for the
    # old state element. So we just directly pick out the fm version of the constraint
    # vector in our comparison
    sold_constraint_vector_fm = sold._current_state_old.state_constraint_vector(
        StateElementIdentifier(sid)
    )
    s = StateElementFromClimatology.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
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


# Repeat with cris_tropomi. Most of these act the same, but we shook out
# a few issues by looking at both.
@pytest.mark.parametrize(
    "sid",
    [
        "CO",
        "CO2",
        "HNO3",
        "CFC12",
        "CCL4",
        "CFC22",
        "N2O",
        "O3",
        "CH4",
        "SF6",
        "C2H4",
        "PAN",
        "HCN",
        "CFC11",
    ],
)
def test_state_element_from_climatology2(cris_tropomi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta = cris_tropomi_old_shandle
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    # sold.constraint_vector_fm isn't always the fm contraint vector, sometime is
    # is the fmprime version. This is just an oddity of the old state element, all that
    # actually matters is constraint_vector_ret, but that isn't always defined for the
    # old state element. So we just directly pick out the fm version of the constraint
    # vector in our comparison
    sold_constraint_vector_fm = sold._current_state_old.state_constraint_vector(
        StateElementIdentifier(sid)
    )
    s = StateElementFromClimatology.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
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


def test_state_element_from_climatology_hdo(airs_omi_old_shandle):
    h_old, measurement_id, rconfig, strat, obs_hset, smeta = airs_omi_old_shandle
    state_info = StateInfo()
    state_info.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    sid = "HDO"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromClimatologyHdo.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=state_info,
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


def test_state_element_from_climatology_ch3oh(airs_omi_old_shandle):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    sid = "CH3OH"
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromClimatologyCh3oh.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype == "LAND_ENH"
    assert s.metadata == {"poltype": "LAND_ENH"}
    assert s.poltype_used_constraint
