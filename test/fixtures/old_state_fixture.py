# Fixtures for setting up the old state elements, used for comparing against our
# new code.
import pytest
import os
from refractor.muses import (
    MusesRunDir,
    MusesObservationHandlePickleSave,
    MusesTropomiObservation,
    MusesOmiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesObservationHandle,
    InstrumentIdentifier,
    ObservationHandleSet,
    MeasurementIdFile,
    SoundingMetadata,
    MusesStrategyStepList,
    RetrievalConfiguration,
    StateInfo,
)
from refractor.old_py_retrieve_wrapper import state_element_old_wrapper_handle
from loguru import logger
import sys


@pytest.fixture(scope="function")
def cris_tropomi_old_shandle(
    osp_dir, gmao_dir, joint_tropomi_test_in_dir, isolated_dir
):
    """Set up the old state info, and return a handle"""
    # The setup is really noisy with the logger. Since we aren't actually testing this,
    # suppress this just so we can see what we actually care about
    logger.remove()
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        osp_dir,
        gmao_dir,
    )
    # The old state element stuff assumes it is in the run directory
    os.chdir(r.run_dir)
    tfilename = r.run_dir / "Table.asc"
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        tfilename, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    measurement_id = MeasurementIdFile(r.run_dir / "Measurement_ID.asc", rconfig, {})
    strat = MusesStrategyStepList.create_from_strategy_file(tfilename, osp_dir=osp_dir)
    strat.notify_update_target(measurement_id)
    measurement_id.filter_list_dict = strat.filter_list_dict
    obs_hset = ObservationHandleSet()
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("CRIS"), MusesCrisObservation
        )
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("TROPOMI"), MusesTropomiObservation
        )
    )
    obs_hset.notify_update_target(measurement_id)
    smeta = SoundingMetadata.create_from_measurement_id(
        measurement_id,
        strat.instrument_name[0],
        obs_hset.observation(
            strat.instrument_name[0], None, None, None, osp_dir=osp_dir
        ),
    )
    state_element_old_wrapper_handle.notify_update_target(
        measurement_id, rconfig, strat, obs_hset, smeta
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    logger.add(sys.stderr, level="DEBUG")
    return (
        state_element_old_wrapper_handle,
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )


@pytest.fixture(scope="function")
def tropomi_swir_old_shandle(
    josh_osp_dir, gmao_dir, tropomi_band7_test_in_dir2, isolated_dir
):
    """Set up the old state info, and return a handle"""
    # The setup is really noisy with the logger. Since we aren't actually testing this,
    # suppress this just so we can see what we actually care about
    logger.remove()
    r = MusesRunDir(
        tropomi_band7_test_in_dir2,
        josh_osp_dir,
        gmao_dir,
    )
    # The old state element stuff assumes it is in the run directory
    os.chdir(r.run_dir)
    tfilename = r.run_dir / "Table.asc"
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        tfilename, osp_dir=josh_osp_dir, gmao_dir=gmao_dir
    )
    measurement_id = MeasurementIdFile(r.run_dir / "Measurement_ID.asc", rconfig, {})
    strat = MusesStrategyStepList.create_from_strategy_file(
        tfilename, osp_dir=josh_osp_dir
    )
    strat.notify_update_target(measurement_id)
    measurement_id.filter_list_dict = strat.filter_list_dict
    obs_hset = ObservationHandleSet()
    obs_hset.add_handle(
        MusesObservationHandle(InstrumentIdentifier("TROPOMI"), MusesTropomiObservation)
    )
    obs_hset.notify_update_target(measurement_id)
    smeta = SoundingMetadata.create_from_measurement_id(
        measurement_id,
        strat.instrument_name[0],
        obs_hset.observation(
            strat.instrument_name[0], None, None, None, osp_dir=josh_osp_dir
        ),
    )
    state_element_old_wrapper_handle.notify_update_target(
        measurement_id, rconfig, strat, obs_hset, smeta
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    logger.add(sys.stderr, level="DEBUG")
    return (
        state_element_old_wrapper_handle,
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )


@pytest.fixture(scope="function")
def airs_omi_old_shandle(osp_dir, gmao_dir, joint_omi_test_in_dir, isolated_dir):
    """Set up the old state info, and return a handle"""
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
        tfilename, osp_dir=osp_dir, gmao_dir=gmao_dir
    )
    measurement_id = MeasurementIdFile(r.run_dir / "Measurement_ID.asc", rconfig, {})
    strat = MusesStrategyStepList.create_from_strategy_file(tfilename, osp_dir=osp_dir)
    strat.notify_update_target(measurement_id)
    measurement_id.filter_list_dict = strat.filter_list_dict
    obs_hset = ObservationHandleSet()
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("AIRS"), MusesAirsObservation
        )
    )
    obs_hset.add_handle(
        MusesObservationHandlePickleSave(
            InstrumentIdentifier("OMI"), MusesOmiObservation
        )
    )
    obs_hset.notify_update_target(measurement_id)
    smeta = SoundingMetadata.create_from_measurement_id(
        measurement_id,
        strat.instrument_name[0],
        obs_hset.observation(
            strat.instrument_name[0], None, None, None, osp_dir=osp_dir
        ),
    )
    state_element_old_wrapper_handle.notify_update_target(
        measurement_id, rconfig, strat, obs_hset, smeta
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    logger.add(sys.stderr, level="DEBUG")
    return (
        state_element_old_wrapper_handle,
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )
