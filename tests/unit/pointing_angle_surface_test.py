from refractor.muses import (
    InstrumentIdentifier,
    AirsFmObjectCreator,
    PointingAngleSurface,
)
import refractor.framework as rf  # type: ignore
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
    sat_radius = rf.DoubleWithUnit(708698.5473632812, "m")
    earth_radius = rf.DoubleWithUnit(6371348.563286134, "m")
    pointing_angle = rf.DoubleWithUnit(-17.09299659729004, "deg")
    pntsurf = PointingAngleSurface(
        sat_radius,
        earth_radius,
        ocreator.pressure_fm,
        ocreator.muses_altitude,
        ocreator.refractive_index,
    )
    pangle = pntsurf.pointing_angle_surface(pointing_angle)
    assert pangle.convert("deg").value == pytest.approx(-19.058069989055987)
    # For IRK, for some reason a different altitude is used. I'm not sure this was
    # actually intended
    sat_radius = rf.DoubleWithUnit(0, "m")
    pntsurf = PointingAngleSurface(
        sat_radius,
        earth_radius,
        ocreator.pressure_fm,
        ocreator.muses_altitude,
        ocreator.refractive_index,
    )
    pointing_angle = rf.DoubleWithUnit(48.16890311054117, "deg")
    pangle = pntsurf.pointing_angle_surface(pointing_angle)
    assert pangle.convert("deg").value == pytest.approx(48.151082849612976)
    pointing_angle = rf.DoubleWithUnit(59.0983034698522, "deg")
    pangle = pntsurf.pointing_angle_surface(pointing_angle)
    assert pangle.convert("deg").value == pytest.approx(59.07165952959758)
    pointing_angle = rf.DoubleWithUnit(63.6764997367534, "deg")
    pangle = pntsurf.pointing_angle_surface(pointing_angle)
    assert pangle.convert("deg").value == pytest.approx(63.644272237719804)
