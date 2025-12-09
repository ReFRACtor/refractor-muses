from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElement,
    StateElementPcloud,
    StateElementFromSingle,
    StateElementFromCalibration,
)
import pytest
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


def test_state_element_pcloud(airs_omi_shandle, unit_test_expected_dir):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    s = StateElementPcloud.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    check_selem(
        s,
        unit_test_expected_dir / "state_element_single" / "pcloud_expect.pkl",
        save=False,
    )

    # Cycle through strategy steps, and check value_fm after that
    # strat.retrieval_initial_fm_from_cycle(s, rconfig)
    # Check an number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.diag_cov
    assert s.metadata == {}


@pytest.mark.parametrize("sid", ("SO2", "NH3", "OCS", "HCOOH", "N2"))
def test_state_element_from_single(airs_omi_shandle, unit_test_expected_dir, sid):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    s = StateElementFromSingle.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    check_selem(
        s,
        unit_test_expected_dir / "state_element_single" / f"{sid}_expect.pkl",
        save=False,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.metadata == {}


# We can't really test this, we don't have test data for it. Can revisit it
# if we ever need to support this
@pytest.mark.skip
@pytest.mark.parametrize(
    "sid",
    (
        "calibrationScale",
        "calibrationOffset",
        "residualScale",
    ),
)
def test_state_element_from_calibration(airs_omi_shandle, unit_test_expected_dir, sid):
    _, rconfig, strat, _, smeta, sinfo = airs_omi_shandle
    # The calibration isn't actually listed in Species_List_From_Single, so go ahead
    # and add it so we can test handling if it was there.
    rconfig["Species_List_From_Single"] = f"{rconfig['Species_List_From_Single']},{sid}"
    s = StateElementFromCalibration.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    check_selem(
        s,
        unit_test_expected_dir / "state_element_single" / f"{sid}_expect.pkl",
        save=False,
    )
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.metadata == {}
