from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElement,
    StateElementFromClimatology,
    StateElementFromGmaoPressure,
    StateElementFromClimatologyHdo,
    StateElementFromClimatologyCh3oh,
    StateElementFromClimatologyNh3,
    StateElementFromClimatologyHcooh,
)
import numpy.testing as npt
from pathlib import Path
import pytest
import pickle


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


def test_read_climatology_2022(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    sid = "pressure"
    p = StateElementFromGmaoPressure.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    panvmr = StateElementFromClimatology.create(
        sid=StateElementIdentifier("PAN"),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    vmr, type_name = StateElementFromClimatology.read_climatology_2022(
        StateElementIdentifier("PAN"),
        p.value_fm,
        False,
        Path(rconfig.osp_dir) / "Climatology/Climatology_files",
        smeta,
        rconfig.input_file_monitor,
    )
    vmr_prior, type_name = StateElementFromClimatology.read_climatology_2022(
        StateElementIdentifier("PAN"),
        p.value_fm,
        True,
        Path(rconfig.osp_dir) / "Climatology/Climatology_files",
        smeta,
        rconfig.input_file_monitor,
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
def test_state_element_from_climatology(airs_omi_shandle, unit_test_expected_dir, sid):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    s = StateElementFromClimatology.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / f"{sid}_expect.pkl",
        save=False,
    )
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
def test_state_element_from_climatology2(
    cris_tropomi_shandle, unit_test_expected_dir, sid
):
    _, rconfig, strat, _, smeta, sinfo = cris_tropomi_shandle
    s = StateElementFromClimatology.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / f"{sid}_2_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_climatology_hdo(airs_omi_shandle, unit_test_expected_dir):
    measurement_id, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "HDO"
    s = StateElementFromClimatologyHdo.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / "hdo_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.metadata == {}
    assert s.poltype_used_constraint


def test_state_element_from_climatology_ch3oh(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    sid = "CH3OH"
    s = StateElementFromClimatologyCh3oh.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / "ch3oh_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype == "LAND_ENH"
    assert s.metadata == {"poltype": "LAND_ENH"}
    assert s.poltype_used_constraint


def test_state_element_from_climatology_nh3(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, obs_hset, smeta, sinfo = airs_omi_shandle
    sid = "NH3"
    s = StateElementFromClimatologyNh3.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        strategy=strat,
        observation_handle_set=obs_hset,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / "nh3_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype == "CLN"
    assert s.metadata == {"poltype": "CLN"}
    assert s.poltype_used_constraint


def test_state_element_from_climatology_hcooh(tes_shandle, unit_test_expected_dir):
    _, rconfig, strat, obs_hset, smeta, sinfo = tes_shandle
    sid = "HCOOH"
    s = StateElementFromClimatologyHcooh.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        strategy=strat,
        observation_handle_set=obs_hset,
        state_info=sinfo,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    check_selem(
        s,
        unit_test_expected_dir / "state_element_climatology" / "hcooh_expect.pkl",
        save=False,
    )
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype == "CLN"
    assert s.metadata == {"poltype": "CLN"}
    assert not s.poltype_used_constraint
