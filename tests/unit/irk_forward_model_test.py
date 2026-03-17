from refractor.muses import (
    InstrumentIdentifier,
    StateElementIdentifier,
    CrisFmObjectCreator,
    AirsFmObjectCreator,
)
from refractor.muses_py_fm import (
    MusesCrisForwardModel,
    MusesAirsForwardModel,
)
from fixtures.require_check import require_muses_py_fm
import pytest
import numpy.testing as npt


@require_muses_py_fm
def test_muses_cris_forward_model_irk(joint_tropomi_step_12):
    """We don't normally run IRK for CRIS (only for AIRS), but put this test
    in because it doesn't have the added complication that AIRS uses the
    TES frequencies. This tests the IRK code without including that extra
    piece (which we test separately"""
    rs, rstep, _ = joint_tropomi_step_12
    obs_cris = rs.observation_handle_set.observation(
        InstrumentIdentifier("CRIS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("CRIS")],
        None,
    )
    obs_cris.spectral_window.include_bad_sample = True
    # For the purpose of testing, add some extra jacobians in so we can check
    # that part of the code.
    rs.current_state.testing_add_retrieval_state_element_id(
        StateElementIdentifier("TATM")
    )
    ocreator = CrisFmObjectCreator(rs.current_state, rs.retrieval_config, obs_cris)
    fm = ocreator.forward_model
    rirk = fm.irk(rs.current_state)

    fmcmp = MusesCrisForwardModel(rs.current_state, obs_cris, rs.retrieval_config)
    rirkcmp = fmcmp.irk(rs.current_state)
    # Small round off errors, so increase tolerance slightly.
    assert rirk.flux == pytest.approx(rirkcmp.flux)
    assert rirk.flux_l1b == pytest.approx(rirkcmp.flux_l1b)
    npt.assert_allclose(rirk.fluxSegments, rirkcmp.fluxSegments, rtol=1e-6)
    npt.assert_allclose(rirk.fluxSegments_l1b, rirkcmp.fluxSegments_l1b, rtol=2e-6)
    npt.assert_allclose(rirk.freqSegments_irk, rirkcmp.freqSegments_irk, rtol=1e-6)
    assert rirk.radiances["gi_angle"] == pytest.approx(rirkcmp.radiances["gi_angle"])
    npt.assert_allclose(
        rirk.radiances["radarr_fm"],
        rirkcmp.radiances["radarr_fm"],
        atol=1e-8,
        rtol=1e-6,
    )
    npt.assert_allclose(
        rirk.radiances["freq_fm"], rirkcmp.radiances["freq_fm"], rtol=1e-6
    )
    npt.assert_allclose(
        rirk.radiances["rad_L1b"], rirkcmp.radiances["rad_L1b"], rtol=1e-6
    )
    npt.assert_allclose(
        rirk.radiances["freq_L1b"], rirkcmp.radiances["freq_L1b"], rtol=1e-6
    )


@require_muses_py_fm
def test_muses_airs_forward_model_irk(airs_irk_step_6):
    rs, rstep, _ = airs_irk_step_6
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    obs_airs.spectral_window.include_bad_sample = True
    ocreator = AirsFmObjectCreator(rs.current_state, rs.retrieval_config, obs_airs)
    fm = ocreator.forward_model
    rirk = fm.irk(rs.current_state)
    fmcmp = MusesAirsForwardModel(rs.current_state, obs_airs, rs.retrieval_config)
    rirkcmp = fmcmp.irk(rs.current_state)
    # Small round off errors, so increase tolerance slightly.
    assert rirk.flux == pytest.approx(rirkcmp.flux)
    assert rirk.flux_l1b == pytest.approx(rirkcmp.flux_l1b)
    npt.assert_allclose(rirk.fluxSegments, rirkcmp.fluxSegments, rtol=1e-6)
    npt.assert_allclose(rirk.fluxSegments_l1b, rirkcmp.fluxSegments_l1b, rtol=1e-6)
    npt.assert_allclose(rirk.freqSegments_irk, rirkcmp.freqSegments_irk, rtol=1e-6)
    assert rirk.radiances["gi_angle"] == pytest.approx(rirkcmp.radiances["gi_angle"])
    npt.assert_allclose(
        rirk.radiances["radarr_fm"],
        rirkcmp.radiances["radarr_fm"],
        atol=1e-10,
        rtol=1e-6,
    )
    npt.assert_allclose(
        rirk.radiances["freq_fm"], rirkcmp.radiances["freq_fm"], rtol=1e-6
    )
    npt.assert_allclose(
        rirk.radiances["rad_L1b"], rirkcmp.radiances["rad_L1b"], rtol=1e-6
    )
    npt.assert_allclose(
        rirk.radiances["freq_L1b"], rirkcmp.radiances["freq_L1b"], rtol=1e-6
    )
