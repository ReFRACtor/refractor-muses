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
    InstrumentIdentifier,
    ObservationHandleSet,
    MeasurementIdFile,
    SoundingMetadata,
    MusesStrategyStepList,
    RetrievalConfiguration,
    h_old,
)
from loguru import logger
import sys


@pytest.fixture(scope="function")
def cris_tropomi_old_shandle(
    osp_dir, gmao_dir, joint_tropomi_test_in_dir, isolated_dir
):
    """Set up the old state info, and return a StateElementOspFileHandleNew"""
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
        tfilename, osp_dir=osp_dir
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
    h_old.notify_update_target(measurement_id, rconfig, strat, obs_hset, smeta)
    logger.add(sys.stderr, level="DEBUG")
    return h_old, measurement_id, rconfig, strat, obs_hset, smeta


@pytest.fixture(scope="function")
def airs_omi_old_shandle(osp_dir, gmao_dir, joint_omi_test_in_dir, isolated_dir):
    """Set up the old state info, and return a StateElementOspFileHandleNew"""
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
    h_old.notify_update_target(measurement_id, rconfig, strat, obs_hset, smeta)
    logger.add(sys.stderr, level="DEBUG")
    return h_old, measurement_id, rconfig, strat, obs_hset, smeta
