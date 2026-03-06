from refractor.muses import (
    InstrumentIdentifier,
    AirsFmObjectCreator,
    MusesAltitude,
    MusesRefractiveIndex,
)
import refractor.framework as rf  # type: ignore
import numpy.testing as npt
import pytest

def test_muses_altitude(joint_omi_step_8_no_run_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    ocreator = AirsFmObjectCreator(rs.current_state, rs.retrieval_config, obs_airs)
    alt = MusesAltitude(
        ocreator.pressure_fm,
        ocreator.temperature,
        ocreator.h2o_vmr,
        rs.current_state.sounding_metadata.latitude,
        rs.current_state.sounding_metadata.surface_altitude,
    )
    rindex = MusesRefractiveIndex(ocreator.pressure_fm, ocreator.temperature, ocreator.h2o_vmr,
                                  alt)
    # Grabbed from old py-retrieve code.
    assert rindex.refractive_index(rf.DoubleWithUnit(1618.063355672948, "m")) == pytest.approx(1.0002328018712983)
    assert rindex.refractive_index(rf.DoubleWithUnit(31149.47757696744, "m")) == pytest.approx(1.0000034326320308)
