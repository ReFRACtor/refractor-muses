import os
import pytest
from refractor.muses import (
    MusesRunDir,
    RetrievalStrategy,
    ProcessLocation,
    MusesObservationHandlePickleSave,
    MusesOmiObservation,
    MusesTropomiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesTesObservation,
    InstrumentIdentifier,
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
    rs.load_step_info(
        dir / "current_state_record.pkl",
        step_number,
        ret_state_file=dir / f"retrieval_state_step_{step_number}.json.gz"
        if include_ret_state
        else None,
    )


def set_up_run_to_location(
    dir: str | Path,
    step_number: int,
    location: str | ProcessLocation,
    osp_dir: str | Path,
    gmao_dir: str | Path,
    include_ret_state=True,
):
    """Set up directory and run the given step number to the given
    location.

    Note that this returns the full RetrievalStrategy. A particular
    test might not need everything - for example a lot of tests just
    need the CurrentState. However the overall RetrievalStrategy code
    is reasonably stable and it has everything. Returning this seems
    cleaner that having a fixture that returns CurrentState, another
    that returns CurrentState + MusesStrategy etc. We can revisit this
    if needed, but for now the simplicity of "always having
    everything" seems reasonable.

    Also note that it turns out reading the MusesObservation takes a
    significant amount of time. We are only talking about 5-10
    seconds, but since we have lots of unit tests that use these
    fixtures it is worth speeding these up. We of course need to test
    reading the MusesObservation (and have unit tests for that), but
    for other tests that just need a MusesObservation we don't need to
    read these in ever test. So we have pickled versions of these that
    we save in our capture tests, and then use a
    MusesObservationHandlePickleSave to read those pickled
    versions. It is okay if the pickle file isn't there, we then just
    fall back to reading the original files. But it is worth having
    these for most of the tests.
    """
    # The logger is really noisy as we set stuff up. Since we aren't actually
    # interested in debugging the setup, turn this off. We turn this back on
    # at the end, so the log messages are for the things actually in our tests.
    logger.remove()
    r = MusesRunDir(dir, osp_dir, gmao_dir)
    # TODO Short term turn off checking values. This is temporary, we will replace the
    # old state info stuff in a bit
    # CurrentState.check_old_state_element_value = False
    rs = RetrievalStrategy(None, osp_dir=osp_dir)
    obs_hset = rs.observation_handle_set
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("AIRS"), MusesAirsObservation
        ),
        priority_order=2,
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("OMI"), MusesOmiObservation
        ),
        priority_order=2,
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("CRIS"), MusesCrisObservation
        ),
        priority_order=2,
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("TROPOMI"), MusesTropomiObservation
        ),
        priority_order=2,
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("TES"), MusesTesObservation
        ),
        priority_order=2,
    )
    rs.update_target(r.run_dir / "Table.asc")
    rstep, kwargs = run_step_to_location(
        rs, step_number, dir, location, include_ret_state=include_ret_state
    )
    logger.add(sys.stderr, level="DEBUG")
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
    return None, None


@pytest.fixture(scope="function")
def joint_omi_step_8(isolated_dir, joint_omi_test_in_dir, osp_dir, gmao_dir):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_omi_test_in_dir, 8, "retrieval input", osp_dir, gmao_dir
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def omi_step_0(isolated_dir, omi_test_in_dir, osp_dir, gmao_dir):
    rs, rstep, kwargs = set_up_run_to_location(
        omi_test_in_dir, 0, "retrieval input", osp_dir, gmao_dir
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12(
    isolated_dir,
    joint_tropomi_test_in_dir,
    osp_dir,
    gmao_dir,
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval input", osp_dir, gmao_dir
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def joint_tropomi_step_12_output(
    isolated_dir,
    joint_tropomi_test_in_dir,
    osp_dir,
    gmao_dir,
):
    rs, rstep, kwargs = set_up_run_to_location(
        joint_tropomi_test_in_dir, 12, "retrieval step", osp_dir, gmao_dir
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def airs_irk_step_6(
    isolated_dir,
    airs_irk_test_in_dir,
    osp_dir,
    gmao_dir,
):
    rs, rstep, kwargs = set_up_run_to_location(
        airs_irk_test_in_dir, 6, "IRK step", osp_dir, gmao_dir
    )
    os.chdir(rs.run_dir)
    return rs, rstep, kwargs


@pytest.fixture(scope="function")
def tropomi_fm_object_creator_step_0(
    request, isolated_dir, osp_dir, gmao_dir, tropomi_test_in_dir
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
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    obs = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
        osp_dir=osp_dir,
    )
    res = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        obs,
        use_oss=use_oss,
        oss_training_data=oss_training_data,
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
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    obs = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
        osp_dir=josh_osp_dir,
    )
    res = TropomiSwirFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        obs,
        use_oss=use_oss,
        oss_training_data=oss_training_data,
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
    isolated_dir, osp_dir, gmao_dir, tropomi_test_in_dir
):
    """Fixture for TropomiFmObjectCreator, at the start of step 1"""
    rs, rstep, _ = set_up_run_to_location(
        tropomi_test_in_dir,
        1,
        "retrieval input",
        osp_dir,
        gmao_dir,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    obs = rs.observation_handle_set.observation(
        InstrumentIdentifier("TROPOMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("TROPOMI")],
        None,
        osp_dir=osp_dir,
    )

    res = TropomiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        obs,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def omi_fm_object_creator_step_0(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    """Fixture for OmiFmObjectCreator, at the start of step 0"""
    rs, rstep, _ = set_up_run_to_location(
        omi_test_in_dir,
        0,
        "retrieval input",
        osp_dir,
        gmao_dir,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    obs = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
        osp_dir=osp_dir,
    )

    res = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        obs,
        osp_dir=osp_dir,
    )
    # Put RetrievalStrategy and RetrievalStrategyStep into OmiFmObjectCreator,
    # just for use in unit tests. We could set up a different way of passing
    # this one, but shoving into the creator object is the easiest
    res.rs = rs
    res.rstep = rstep
    return res


@pytest.fixture(scope="function")
def omi_fm_object_creator_step_1(isolated_dir, osp_dir, gmao_dir, omi_test_in_dir):
    """Fixture for OmiFmObjectCreator, at the start of step 0"""
    rs, rstep, _ = set_up_run_to_location(
        omi_test_in_dir,
        1,
        "retrieval input",
        osp_dir,
        gmao_dir,
        include_ret_state=False,
    )
    os.chdir(rs.run_dir)
    obs = rs.observation_handle_set.observation(
        InstrumentIdentifier("OMI"),
        rs.current_state,
        rs.current_strategy_step.spectral_window_dict[InstrumentIdentifier("OMI")],
        None,
        osp_dir=osp_dir,
    )
    res = OmiFmObjectCreator(
        rs.current_state,
        rs.measurement_id,
        obs,
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
