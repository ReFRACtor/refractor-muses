import os
import pytest
from refractor.muses import (
    MusesRunDir,
    RetrievalStrategy,
    ProcessLocation,
    CurrentState,
    CurrentStateRecordAndPlay,
    CurrentStateStateInfo,
    MusesObservationHandlePickleSave,
    MusesAirsObservation,
    MusesOmiObservation,
    InstrumentIdentifier,
    ObservationHandleSet,
    MusesStrategyStepList,
    MeasurementIdFile,
    RetrievalConfiguration,
)
from refractor.tropomi import TropomiFmObjectCreator, TropomiSwirFmObjectCreator
from refractor.omi import OmiFmObjectCreator
from pathlib import Path
from loguru import logger
import sys

# Fixtures that set up a full RetrievalStrategy at a given retrieval step, for use
# in testing that is hard to do outside of a full retrieval


def load_step(
    rs: RetrievalStrategy, step_number: int, dir: Path, include_ret_state=False
):
    """Load in the state information and optional retrieval results for the given
    step, and jump to that step."""
    rs.load_state_info(
        dir / f"state_info_step_{step_number}.pkl",
        step_number,
        ret_state_file=f"{dir}/retrieval_state_step_{step_number}.json.gz"
        if include_ret_state
        else None,
    )


def set_up_run_to_location(
    dir: str | Path,
    step_number: int,
    location: str | ProcessLocation,
    osp_dir: str | Path,
    gmao_dir: str | Path,
    vlidort_cli: str,
    include_ret_state=True,
):
    """Set up directory and run the given step number to the given location."""
    r = MusesRunDir(dir, osp_dir, gmao_dir)
    # TODO Short term turn off checking values. This is temporary, we will replace the
    # old state info stuff in a bit
    CurrentState.check_old_state_element_value = False
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc", vlidort_cli=vlidort_cli, osp_dir=osp_dir
    )
    rstep, kwargs = run_step_to_location(
        rs, step_number, dir, location, include_ret_state=include_ret_state
    )
    return rs, rstep, kwargs


def run_step_to_location(
    rs: RetrievalStrategy,
    step_number: int,
    dir: str | Path,
    location: str | ProcessLocation,
    include_ret_state=True,
):
    """Load in the given step, and run up to the location we notify at
    (e.g., "retrieval step"). Return the retrieval_strategy_step"""

    class CaptureRs:
        def __init__(self):
            self.retrieval_strategy_step = None

        def notify_update(
            self, retrieval_strategy, loc, retrieval_strategy_step=None, **kwargs
        ):
            if loc != ProcessLocation(location):
                return
            self.retrieval_strategy_step = retrieval_strategy_step
            self.kwargs = kwargs
            raise StopIteration()

    try:
        rcap = CaptureRs()
        rs.add_observer(rcap)
        load_step(rs, step_number, Path(dir), include_ret_state=include_ret_state)
        try:
            rs.continue_retrieval(stop_after_step=step_number)
        except StopIteration:
            return rcap.retrieval_strategy_step, rcap.kwargs
    finally:
        rs.remove_observer(rcap)


@pytest.fixture(scope="function")
def joint_omi_step_8(
    isolated_dir, joint_omi_test_in_dir, osp_dir, gmao_dir, vlidort_cli
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_omi_test_in_dir, 8, "retrieval input", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def omi_step_0(isolated_dir, omi_test_in_dir, osp_dir, gmao_dir, vlidort_cli):
    rs, rstep, kwargs = set_up_run_to_location(
        omi_test_in_dir, 0, "retrieval input", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12(
    isolated_dir,
    joint_tropomi_test_in_dir,
    osp_dir,
    gmao_dir,
    vlidort_cli,
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval input", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12_output(
    isolated_dir,
    joint_tropomi_test_in_dir,
    osp_dir,
    gmao_dir,
    vlidort_cli,
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval step", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def airs_irk_step_6(
    isolated_dir,
    airs_irk_test_in_dir,
    osp_dir,
    gmao_dir,
    vlidort_cli,
):
    rs, rstep, kwargs = set_up_run_to_location(
        airs_irk_test_in_dir, 6, "IRK step", osp_dir, gmao_dir, vlidort_cli
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_0(
    request, isolated_dir, osp_dir, gmao_dir, vlidort_cli, tropomi_test_in_dir
):
    """Fixture for TropomiFmObjectCreator, at the start of step 1"""

    oss_param = getattr(request, "param", {})
    use_oss = oss_param.get("use_oss", False)
    oss_training_data = oss_param.get("oss_training_data", None)
    if oss_training_data is not None:
        oss_training_data = tropomi_test_in_dir / oss_training_data
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir,
        0,
        "retrieval input",
        osp_dir,
        gmao_dir,
        vlidort_cli,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(
        InstrumentIdentifier("TROPOMI")
    )
    res = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("TROPOMI")
            ],
            None,
            osp_dir=osp_dir,
        ),
        use_oss=use_oss,
        oss_training_data=oss_training_data,
        rf_uip_func=lambda instrument: uip,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_swir_step(
    request,
    isolated_dir,
    josh_osp_dir,
    gmao_dir,
    vlidort_cli,
    tropomi_band7_test_in_dir,
):
    """Fixture for TropomiFmObjectCreator, just so we don't need to repeat code
    in multiple tests"""
    # Note this example is pretty hokey, and we don't even complete the first step. So
    # skip the ret_info stuff

    oss_param = getattr(request, "param", {})
    use_oss = oss_param.get("use_oss", False)
    oss_training_data = oss_param.get("oss_training_data", None)
    if oss_training_data is not None:
        oss_training_data = tropomi_band7_test_in_dir / oss_training_data
    rs, rstep, _ = set_up_run_to_location(
        tropomi_band7_test_in_dir,
        0,
        "retrieval input",
        josh_osp_dir,
        gmao_dir,
        vlidort_cli,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    res = TropomiSwirFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("TROPOMI")
            ],
            None,
            osp_dir=josh_osp_dir,
        ),
        use_oss=use_oss,
        oss_training_data=oss_training_data,
        rf_uip_func=rs.strategy_executor.rf_uip_func_cost_function(False, None),
        osp_dir=josh_osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_1(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, tropomi_test_in_dir
):
    """Fixture for TropomiFmObjectCreator, at the start of step 1"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir,
        1,
        "retrieval input",
        osp_dir,
        gmao_dir,
        vlidort_cli,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(
        InstrumentIdentifier("TROPOMI")
    )
    res = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.observation_handle_set.observation(
            InstrumentIdentifier("TROPOMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[
                InstrumentIdentifier("TROPOMI")
            ],
            None,
            osp_dir=osp_dir,
        ),
        rf_uip_func=lambda instrument: uip,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def omi_fm_object_creator_step_0(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, omi_test_in_dir
):
    """Fixture for OmiFmObjectCreator, at the start of step 0"""
    rs, rstep, _ = set_up_run_to_location(
        omi_test_in_dir,
        0,
        "retrieval input",
        osp_dir,
        gmao_dir,
        vlidort_cli,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(
        InstrumentIdentifier("OMI")
    )
    res = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
            None,
            osp_dir=osp_dir,
        ),
        rf_uip_func=lambda instrument: uip,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res

@pytest.fixture(scope="function")
def joint_omi_current_state(osp_dir, gmao_dir, joint_omi_test_in_dir, isolated_dir):
    '''Set up the CurrentState, for the start of the retrieval of a joint air/omi
    target'''
    # The setup is really noisy with the logger. Since we aren't actually testing this,
    # suppress this just so we can see what we actually care about
    logger.remove()
    r = MusesRunDir(
        joint_omi_test_in_dir,
        osp_dir,
        gmao_dir,
    )
    # The old state element stuff assumes it is in the run directory
    os.chdir(r.run_dir)
    tfilename = r.run_dir / "Table.asc"
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        tfilename, osp_dir=osp_dir
    )
    measurement_id = MeasurementIdFile(r.run_dir / "Measurement_ID.asc", rconfig, {})
    strat = MusesStrategyStepList.create_from_strategy_file(tfilename, osp_dir=osp_dir)
    strat.notify_update_target(measurement_id)
    measurement_id.filter_list_dict = strat.filter_list_dict
    obs_hset = ObservationHandleSet()
    obs_hset.add_handle(MusesObservationHandlePickleSave(InstrumentIdentifier("AIRS"), MusesAirsObservation))
    obs_hset.add_handle(MusesObservationHandlePickleSave(InstrumentIdentifier("OMI"), MusesOmiObservation))
    obs_hset.notify_update_target(measurement_id)
    cstate = CurrentStateStateInfo()
    cstate.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    cstate.notify_start_retrieval(strat.current_strategy_step(), rconfig)
    cstate_record = CurrentStateRecordAndPlay(joint_omi_test_in_dir / "current_state_record.pkl")
    logger.add(sys.stderr, level="DEBUG")
    return cstate, cstate_record, strat, rconfig, measurement_id

@pytest.fixture(scope="function")
def omi_fm_object_creator_step_1(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, omi_test_in_dir
):
    """Fixture for OmiFmObjectCreator, at the start of step 0"""
    rs, rstep, _ = set_up_run_to_location(
        omi_test_in_dir,
        1,
        "retrieval input",
        osp_dir,
        gmao_dir,
        vlidort_cli,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    uip = rs.strategy_executor.rf_uip_func_cost_function(False, None)(
        InstrumentIdentifier("OMI")
    )
    res = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        rs.observation_handle_set.observation(
            InstrumentIdentifier("OMI"),
            rs.current_state,
            rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
            None,
            osp_dir=osp_dir,
        ),
        rf_uip_func=lambda instrument: uip,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def tropomi_swir(isolated_dir, gmao_dir, josh_osp_dir, tropomi_band7_test_in_dir2):
    r = MusesRunDir(tropomi_band7_test_in_dir2, josh_osp_dir, gmao_dir)
    return r
