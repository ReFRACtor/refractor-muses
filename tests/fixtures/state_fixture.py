import pytest
from refractor.muses import (
    MusesRunDir,
    MusesObservationHandlePickleSave,
    MusesTropomiObservation,
    MusesOmiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesTesObservation,
    MusesObservationHandle,
    InstrumentIdentifier,
    ObservationHandleSet,
    MeasurementIdFile,
    SoundingMetadata,
    MusesStrategyStepList,
    RetrievalConfiguration,
    StateInfo,
)
from loguru import logger
import sys


@pytest.fixture(scope="function")
def cris_tropomi_shandle(osp_dir, gmao_dir, joint_tropomi_test_in_dir, isolated_dir):
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        osp_dir,
        gmao_dir,
    )
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
        obs_hset.observation(strat.instrument_name[0], None, None, None),
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    return (
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )


@pytest.fixture(scope="function")
def tropomi_swir_shandle(
    josh_osp_dir, gmao_dir, tropomi_band7_test_in_dir2, isolated_dir
):
    r = MusesRunDir(
        tropomi_band7_test_in_dir2,
        josh_osp_dir,
        gmao_dir,
    )
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
            strat.instrument_name[0],
            None,
            None,
            None,  # osp_dir=josh_osp_dir
        ),
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    return (
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )


@pytest.fixture(scope="function")
def airs_omi_shandle(osp_dir, gmao_dir, joint_omi_test_in_dir, isolated_dir):
    r = MusesRunDir(
        joint_omi_test_in_dir,
        osp_dir,
        gmao_dir,
    )
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
            strat.instrument_name[0],
            None,
            None,
            None,
        ),
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    return (
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )


@pytest.fixture(scope="function")
def tes_shandle(osp_dir, gmao_dir, tes_test_in_dir, isolated_dir):
    r = MusesRunDir(
        tes_test_in_dir,
        osp_dir,
        gmao_dir,
    )
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
            InstrumentIdentifier("TES"), MusesTesObservation
        )
    )
    obs_hset.notify_update_target(measurement_id)
    smeta = SoundingMetadata.create_from_measurement_id(
        measurement_id,
        strat.instrument_name[0],
        obs_hset.observation(
            strat.instrument_name[0],
            None,
            None,
            None,
        ),
    )
    sinfo = StateInfo()
    sinfo.notify_update_target(measurement_id, rconfig, strat, obs_hset)
    logger.add(sys.stderr, level="DEBUG")
    return (
        measurement_id,
        rconfig,
        strat,
        obs_hset,
        smeta,
        sinfo,
    )
