from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementPcloud,
    StateElementFromSingle,
    StateElementFromCalibration,
    StateElementOldInitialValue,
)
import numpy.testing as npt
import pytest


def test_state_element_pcloud(airs_omi_old_shandle_ok_no_muses_py):
    h_old, _, rconfig, strat, _, smeta, sinfo = airs_omi_old_shandle_ok_no_muses_py
    s = StateElementPcloud.create(
        retrieval_config=rconfig, sounding_metadata=smeta, state_info=sinfo
    )
    if h_old is not None:
        sold = h_old.state_element(StateElementIdentifier("PCLOUD"))
        sold_value_fm = sold.value_fm
        # This is value_fm before we have cycled through all the strategy
        # steps
        sold_constraint_vector_fm = sold.constraint_vector_fm
        npt.assert_allclose(sold_value_fm, sold_constraint_vector_fm)
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
def test_state_element_from_single(airs_omi_old_shandle_ok_no_muses_py, sid):
    h_old, _, rconfig, strat, _, smeta, sinfo = airs_omi_old_shandle_ok_no_muses_py
    s = StateElementFromSingle.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    if h_old is not None:
        sold = h_old.state_element(StateElementIdentifier(sid))
        sold_value_fm = sold.value_fm
        # This is value_fm before we have cycled through all the strategy
        # steps
        sold_constraint_vector_fm = sold.constraint_vector_fm
        # NH3 has separate logic to override the value in some cases. Skip checking
        # against sold - we don't agree but that is ok because this particular handle
        # doesn't get used in this test case
        if sid != "NH3":
            npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    if h_old is not None and sid != "NH3":
        npt.assert_allclose(s.value_fm, sold_value_fm)
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
def test_state_element_from_calibration(airs_omi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta, sinfo = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    # The calibration isn't actually listed in Species_List_From_Single, so go ahead
    # and add it so we can test handling if it was there.
    rconfig["Species_List_From_Single"] = f"{rconfig['Species_List_From_Single']},{sid}"
    s = StateElementFromCalibration.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        state_info=sinfo,
    )
    # npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    # Cycle through strategy steps, and check value_fm after that
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check a number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.metadata == {}


@pytest.mark.skip
@pytest.mark.parametrize(
    "sid", ("calibrationScale", "calibrationOffset", "residualScale", "scalePressure")
)
def test_state_element_from_default(airs_omi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta, sinfo = airs_omi_old_shandle
    # sold = h_old.state_element(StateElementIdentifier(sid))
    s = StateElementOldInitialValue.create(
        retrieval_config=rconfig,
        sounding_metadata=smeta,
        sid=StateElementIdentifier(sid),
    )
    print(s.value_fm)
    print(s.value_fm.shape)
    print(s.spectral_domain)
    if s.spectral_domain is not None:
        print(s.spectral_domain.data)
    # residual scale fixed at 40 zeros
    # scale pressure fixed at 0.1
    # calibration offset fixed at 300 zeros
    # calibration scale is zeros, not sure how size is calculated (25)
    # Comes from CalibrationState in StateElementOld
    # calibrationpars num_frequencies
    # From /bigdata/smyth/OSP/OSP/L2_Setup/ops/L2_Setup/State_CalibrationData.asc
    # See get_on_state.py for getting the file name
    # These values could come form the calibration data file, however they seem to be
    # held to 0
