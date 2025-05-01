from __future__ import annotations
from refractor.muses import (
    CurrentStateDict,
    CurrentStateUip,
    RetrievalStrategy,
    MusesRunDir,
    ProcessLocation,
    StateElementIdentifier,
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
    cstate = rs.current_state
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
    cstate = rs.current_state
    rs._strategy_executor.restart()

    # Update results, and make sure element gets updated
    rs.current_state.notify_start_step(
        rs.current_strategy_step, rs.error_analysis, rs.retrieval_config
    )
    results_list = np.zeros((len(rs.current_state.retrieval_state_vector_element_list)))
    msk = (
        np.array([str(i) for i in rs.current_state.retrieval_state_vector_element_list])
        == "OMICLOUDFRACTION"
    )
    results_list[msk] = 0.5
    cstate.notify_step_solution(results_list)
    # Go to the next step, and check that the state element is updated
    rs._strategy_executor.next_step()
    rs._strategy_executor.notify_start_step()
