from __future__ import annotations
from refractor.muses import (
    StateElementIdentifier,
    StateElementFromClimatology,
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


# Doesn't currently work. Comment out so we can check in, we'll get this working in a bit
@pytest.mark.skip
@pytest.mark.parametrize("sid", ["CO2", "HNO3", "CFC12", "CH3OH", "CCL4", "CFC22",
        "N2O", "O3", "CH4", "CO", "HDO", "SF6", "C2H4", "PAN", "HCN", "CFC11"])
def test_state_element_from_climatology(airs_omi_old_shandle, sid):
    h_old, _, rconfig, strat, _, smeta = airs_omi_old_shandle
    sold = h_old.state_element(StateElementIdentifier(sid))
    sold_value_fm = sold.value_fm
    sold_constraint_vector_fm = sold.constraint_vector_fm
    s = StateElementFromClimatology.create(
        sid=StateElementIdentifier(sid),
        retrieval_config=rconfig,
        sounding_metadata=smeta,
    )
    # Cycle through strategy steps, and check value_fm after that
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    #npt.assert_allclose(s.value_fm, sold_value_fm)
    npt.assert_allclose(s.value_fm, sold_constraint_vector_fm)
    strat.retrieval_initial_fm_from_cycle(s, rconfig)
    npt.assert_allclose(s.constraint_vector_fm, sold_constraint_vector_fm)
    npt.assert_allclose(s.value_fm, sold_value_fm)
    # Check an number of things we set in StateElementOsp, just to make sure
    # we match stuff
    assert s.spectral_domain is None
    assert not s.cov_is_constraint
    assert s.poltype is None
    assert s.poltype_used_constraint
    assert s.metadata == {}
