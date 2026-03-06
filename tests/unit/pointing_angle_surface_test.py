from refractor.muses import (
    InstrumentIdentifier,
    AirsFmObjectCreator,
    MusesAltitude,
    MusesRefractiveIndex,
    pointing_angle_surface,
)
import refractor.framework as rf  # type: ignore
import numpy.testing as npt
import pytest

def test_pointing_angle_surface(joint_omi_step_8_no_run_dir):
    rs, rstep, _ = joint_omi_step_8_no_run_dir
    obs_airs = rs.observation_handle_set.observation(
        InstrumentIdentifier("AIRS"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("AIRS")],
        None,
    )
    ocreator = AirsFmObjectCreator(rs.current_state, rs.retrieval_config, obs_airs)
    # Grabbed from old py-retrieve code.
    sat_radius = rf.DoubleWithUnit(7080047.110649415, "m")
    pointing_angle = rf.DoubleWithUnit(-17.09299659729004, "deg")
    pangle = pointing_angle_surface(sat_radius, pointing_angle, ocreator.pressure_fm, ocreator.muses_altitude, ocreator.refractive_index)
    assert pangle.convert("deg").value == pytest.approx(-19.058069989055987)
