from __future__ import annotations
from refractor.muses import (
    CurrentStateDict,
    CurrentStateUip,
    RetrievalStrategy,
    MusesRunDir,
    ProcessLocation,
    StateElementIdentifier,
    SingleSpeciesHandle,
    OmiEofStateElement,
    modify_strategy_table,
)
import pytest
import numpy.testing as npt
import numpy as np
from fixtures.misc_fixture import all_output_disabled
from typing import Any


class RetrievalStrategyStop:
    def notify_update(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        **kwargs: Any,
    ) -> None:
        if location == ProcessLocation("initial set up done"):
            raise StopIteration()


def test_current_state_dict():
    d = {
        StateElementIdentifier("TROPOMISOLARSHIFTBAND3"): 1.0,
        StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"): 2.0,
        StateElementIdentifier("TROPOMIRADSQUEEZEBAND3"): 3.0,
    }
    cs = CurrentStateDict(
        d,
        [
            StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
        ],
    )
    coeff, mp = cs.object_state(
        [
            StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADSQUEEZEBAND3"),
        ]
    )
    npt.assert_allclose(coeff, [1.0, 2.0, 3.0])
    npt.assert_allclose(mp.retrieval_indexes, [0, 1])
    cs = CurrentStateDict(
        d,
        [
            StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADSQUEEZEBAND3"),
        ],
    )
    coeff, mp = cs.object_state(
        [
            StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADSQUEEZEBAND3"),
        ]
    )
    npt.assert_allclose(coeff, [1.0, 2.0, 3.0])
    npt.assert_allclose(mp.retrieval_indexes, [0, 2])


def test_current_state_uip(joint_tropomi_step_12):
    rs, rstep, _ = joint_tropomi_step_12
    rf_uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(None)
    cs = CurrentStateUip(rf_uip)
    print(cs.fm_sv_loc)
    print(cs.fm_state_vector_size)
    coeff, mp = cs.object_state(
        [
            StateElementIdentifier("TROPOMISOLARSHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADIANCESHIFTBAND3"),
            StateElementIdentifier("TROPOMIRADSQUEEZEBAND3"),
        ]
    )
    print(coeff)
    print(mp)
    print(mp.retrieval_indexes)


def test_current_state(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_tropomi_test_in_dir
):
    try:
        with all_output_disabled():
            r = MusesRunDir(
                joint_tropomi_test_in_dir, osp_dir, gmao_dir, path_prefix="."
            )
            rs = RetrievalStrategy(r.run_dir / "Table.asc")
            rs.register_with_muses_py()
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.retrieval_ms()
    except StopIteration:
        pass
    cstate = rs.current_state()
    assert cstate.sounding_metadata.wrong_tai_time == pytest.approx(839312679.58409)
    assert cstate.full_state_value(StateElementIdentifier("emissivity"))[
        0
    ] == pytest.approx(0.98081997)
    assert cstate.full_state_spectral_domain_wavelength(
        StateElementIdentifier("emissivity")
    )[0] == pytest.approx(600)
    assert cstate.sounding_metadata.latitude.value == pytest.approx(62.8646)
    assert cstate.sounding_metadata.longitude.value == pytest.approx(81.0379)
    assert cstate.sounding_metadata.surface_altitude.convert(
        "m"
    ).value == pytest.approx(169.827)
    assert cstate.sounding_metadata.tai_time == pytest.approx(839312683.58409)
    assert cstate.sounding_metadata.sounding_id == "20190807_065_04_08_5"
    assert cstate.sounding_metadata.is_land
    assert cstate.full_state_value(StateElementIdentifier("cloudEffExt"))[
        0
    ] == pytest.approx(1e-29)
    assert cstate.full_state_spectral_domain_wavelength(
        StateElementIdentifier("cloudEffExt")
    )[0] == pytest.approx(600)
    assert cstate.full_state_value(StateElementIdentifier("PCLOUD"))[
        0
    ] == pytest.approx(500.0)
    assert cstate.full_state_value(StateElementIdentifier("PSUR"))[0] == pytest.approx(
        0.0
    )
    assert cstate.sounding_metadata.local_hour == pytest.approx(11.40252685546875)
    assert cstate.sounding_metadata.height.value[0] == 0
    assert cstate.full_state_value(StateElementIdentifier("TATM"))[0] == pytest.approx(
        293.28302002
    )


def test_update_cloudfraction(omi_step_0):
    """Test updating OMICLOUDFRACTION. Nothing particularly special about this
    StateElement, it is just a good simple test case for checking the muses-py
    species handling."""
    rs, _, _ = omi_step_0
    cstate = rs.current_state()
    rs._strategy_executor.restart()
    selement = cstate.full_state_element(StateElementIdentifier("OMICLOUDFRACTION"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 0
    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(selement.value, np.array([0.3533266]))
    npt.assert_allclose(selement.initialGuessList, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    npt.assert_allclose(selement.constraintMatrix, np.array([[4.0]]))

    # Update results, and make sure element gets updated
    rs._strategy_executor.get_initial_guess()
    rinfo = rs.retrieval_info
    results_list = np.zeros((rinfo.n_totalParameters))
    results_list[np.array(rinfo.species_list) == "OMICLOUDFRACTION"] = 0.5
    cstate.update_state(
        rinfo,
        results_list,
        [],
        rs.retrieval_config,
        rs.current_strategy_step.strategy_step.step_number,
    )

    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(selement.value, np.array([0.5]))
    # Note these don't get updated yet, until we update the initial guess
    npt.assert_allclose(selement.initialGuessList, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    npt.assert_allclose(selement.constraintMatrix, np.array([[4.0]]))

    # Go to the next step, and check that the state element is updated
    # Should clean this up
    cstate._state_info.next_state_to_current()
    rs._strategy_executor.next_step()
    selement = cstate.full_state_element(StateElementIdentifier("OMICLOUDFRACTION"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 1, after an update
    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(selement.value, np.array([0.5]))
    npt.assert_allclose(selement.initialGuessList, np.array([0.5]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.5]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    # Note the constraintMatrix has been updated. There was a different
    # species information file read in the first step, this the value for
    # other steps. So the difference here is actually correct
    npt.assert_allclose(selement.constraintMatrix, np.array([[400.0]]))


def test_noupdate_cloudfraction(omi_step_0):
    """Repeat the previous test, but label the update as "do_not_update". This
    tests the handling of that case."""
    rs, _, _ = omi_step_0
    cstate = rs.current_state()
    rs._strategy_executor.restart()
    selement = cstate.full_state_element(StateElementIdentifier("OMICLOUDFRACTION"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 0
    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(selement.value, np.array([0.3533266]))
    npt.assert_allclose(selement.initialGuessList, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    npt.assert_allclose(selement.constraintMatrix, np.array([[4.0]]))

    # Update results, and make sure element gets updated
    rs._strategy_executor.get_initial_guess()
    rinfo = rs.retrieval_info
    results_list = np.zeros((rinfo.n_totalParameters))
    results_list[np.array(rinfo.species_list) == "OMICLOUDFRACTION"] = 0.5
    cstate.update_state(
        rinfo,
        results_list,
        [StateElementIdentifier("OMICLOUDFRACTION")],
        rs.retrieval_config,
        rs.current_strategy_step.strategy_step.step_number,
    )

    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(selement.value, np.array([0.5]))
    # Note these don't get updated yet, until we update the initial guess
    npt.assert_allclose(selement.initialGuessList, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    npt.assert_allclose(selement.constraintMatrix, np.array([[4.0]]))

    # Go to the next step, and check that the state element is updated
    # Should clean this up
    cstate._state_info.next_state_to_current()
    rs._strategy_executor.next_step()
    selement = cstate.full_state_element(StateElementIdentifier("OMICLOUDFRACTION"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 1, after an update
    assert selement.mapType == "linear"
    npt.assert_allclose(
        selement.pressureList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeList,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.pressureListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    npt.assert_allclose(
        selement.altitudeListFM,
        np.array(
            [
                -2,
            ]
        ),
    )
    # These go back to the original values
    npt.assert_allclose(selement.value, np.array([0.3533266]))
    npt.assert_allclose(selement.initialGuessList, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVector, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterList, np.array([0.0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.3533266]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0.3533266]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0.0]))
    npt.assert_allclose(selement.minimum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum, np.array([-999.0]))
    npt.assert_allclose(selement.maximum_change, np.array([-999.0]))
    npt.assert_allclose(selement.mapToState, np.eye(1))
    npt.assert_allclose(selement.mapToParameters, np.eye(1))
    # Note the constraintMatrix still has been updated. This is separate
    # from the "do_not_update" handling, it just depends on what
    # strategy step we are on.
    npt.assert_allclose(selement.constraintMatrix, np.array([[400.0]]))


def test_update_omieof(isolated_dir, osp_dir, gmao_dir, vlidort_cli, omi_test_in_dir):
    """Repeat the tests for OMICLOUDFRACTION for our own ReFRACtor only
    StateElement. This is the OmiEofStateElement, but this should be pretty
    much the same for any other ReFRACtor only StateElement."""
    try:
        with all_output_disabled():
            r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(None)
            # Modify the Table.asc to add a EOF element. This is just a short cut,
            # so we don't need to make a new strategy table. Eventually a new table
            # will be needed in the OSP directory, but it is too early for that.
            modify_strategy_table(
                rs,
                0,
                [
                    StateElementIdentifier("OMICLOUDFRACTION"),
                    StateElementIdentifier("OMIEOFUV1"),
                    StateElementIdentifier("OMIEOFUV2"),
                ],
            )
            rs.register_with_muses_py()
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.state_element_handle_set.add_handle(
                SingleSpeciesHandle(
                    StateElementIdentifier("OMIEOFUV1"),
                    OmiEofStateElement,
                    pass_state=False,
                    name=StateElementIdentifier("OMIEOFUV1"),
                    number_eof=3,
                )
            )
            rs.state_element_handle_set.add_handle(
                SingleSpeciesHandle(
                    StateElementIdentifier("OMIEOFUV2"),
                    OmiEofStateElement,
                    pass_state=False,
                    name=StateElementIdentifier("OMIEOFUV2"),
                    number_eof=3,
                )
            )
            rs.script_retrieval_ms(r.run_dir / "Table.asc")
    except StopIteration:
        pass
    cstate = rs.current_state()
    rs._strategy_executor.restart()
    selement = cstate.full_state_element(StateElementIdentifier("OMIEOFUV1"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 0
    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))

    # Update results, and make sure element gets updated
    rs._strategy_executor.get_initial_guess()
    rinfo = rs.retrieval_info
    results_list = np.zeros((rinfo.n_totalParameters))
    results_list[np.array(rinfo.species_list) == "OMIEOFUV1"] = [0.5, 0.3, 0.2]
    cstate.update_state(
        rinfo,
        results_list,
        [],
        rs.retrieval_config,
        rs.current_strategy_step.strategy_step.step_number,
    )

    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0.5, 0.3, 0.2]))
    npt.assert_allclose(selement.initialGuessList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))

    # Go to the next step, and check that the state element is updated
    # Should clean this up
    cstate._state_info.next_state_to_current()
    rs._strategy_executor.next_step()
    selement = cstate.full_state_element(StateElementIdentifier("OMIEOFUV1"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 1, after an update
    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0.5, 0.3, 0.2]))
    npt.assert_allclose(selement.initialGuessList, np.array([0.5, 0.3, 0.2]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0.5, 0.3, 0.2]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))


def test_noupdate_omieof(isolated_dir, osp_dir, gmao_dir, vlidort_cli, omi_test_in_dir):
    """Repeat the previous test, but label the update as "do_not_update". This
    tests the handling of that case."""
    try:
        with all_output_disabled():
            r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir, path_prefix=".")
            rs = RetrievalStrategy(None)
            # Modify the Table.asc to add a EOF element. This is just a short cut,
            # so we don't need to make a new strategy table. Eventually a new table
            # will be needed in the OSP directory, but it is too early for that.
            modify_strategy_table(
                rs,
                0,
                [
                    StateElementIdentifier("OMICLOUDFRACTION"),
                    StateElementIdentifier("OMIEOFUV1"),
                    StateElementIdentifier("OMIEOFUV2"),
                ],
            )
            rs.register_with_muses_py()
            rs.clear_observers()
            rs.add_observer(RetrievalStrategyStop())
            rs.state_element_handle_set.add_handle(
                SingleSpeciesHandle(
                    StateElementIdentifier("OMIEOFUV1"),
                    OmiEofStateElement,
                    pass_state=False,
                )
            )
            rs.state_element_handle_set.add_handle(
                SingleSpeciesHandle(
                    StateElementIdentifier("OMIEOFUV2"),
                    OmiEofStateElement,
                    pass_state=False,
                )
            )
            rs.script_retrieval_ms(r.run_dir / "Table.asc")
    except StopIteration:
        pass
    cstate = rs.current_state()
    rs._strategy_executor.restart()
    selement = cstate.full_state_element(StateElementIdentifier("OMIEOFUV1"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 0
    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))

    # Update results, and make sure element gets updated
    rs._strategy_executor.get_initial_guess()
    rinfo = rs.retrieval_info
    results_list = np.zeros((rinfo.n_totalParameters))
    results_list[np.array(rinfo.species_list) == "OMIEOFUV1"] = [0.5, 0.3, 0.2]
    cstate.update_state(
        rinfo,
        results_list,
        [StateElementIdentifier("OMIEOFUV1")],
        rs.retrieval_config,
        rs.current_strategy_step.strategy_step.step_number,
    )

    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0.5, 0.3, 0.2]))
    npt.assert_allclose(selement.initialGuessList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))

    # Go to the next step, and check that the state element is updated
    # Should clean this up
    cstate._state_info.next_state_to_current()
    rs._strategy_executor.next_step()
    selement = cstate.full_state_element(StateElementIdentifier("OMIEOFUV1"))
    selement.update_initial_guess(rs.current_strategy_step)
    # Test all the initial values at step 1, after an update
    assert selement.mapType == "linear"
    npt.assert_allclose(selement.pressureList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeList, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.pressureListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.altitudeListFM, np.array([-2, -2, -2]))
    npt.assert_allclose(selement.value, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVector, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterList, np.array([0, 0, 0]))
    npt.assert_allclose(selement.initialGuessListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.constraintVectorFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.trueParameterListFM, np.array([0, 0, 0]))
    npt.assert_allclose(selement.minimum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.maximum_change, np.array([-999, -999, -999]))
    npt.assert_allclose(selement.mapToState, np.eye(3))
    npt.assert_allclose(selement.mapToParameters, np.eye(3))
    npt.assert_allclose(selement.constraintMatrix, np.diag([100, 100, 100]))
