import pytest
from refractor.muses import (
    MusesRunDir,
    RetrievalStepCaptureObserver,
    RetrievalStrategy,
    CurrentStateRecordAndPlay,
    MusesObservationHandlePickleSave,
    MusesOmiObservation,
    MusesTropomiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesTesObservation,
    InstrumentIdentifier,
)
from refractor.tropomi import TropomiForwardModelHandle, TropomiSwirForwardModelHandle
from refractor.omi import OmiForwardModelHandle
import shutil


def run_capture(rs, run_dir, dir):
    # Most of the tests have a lot in common, so collect that here to give one place
    # to update
    rs.clear_observers()
    rs.strategy_executor.current_state.record = CurrentStateRecordAndPlay()
    rscap2 = RetrievalStepCaptureObserver(dir / "retrieval_state_step")
    rs.add_observer(rscap2)
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
    rs.update_target(run_dir / "Table.asc")
    try:
        rs.retrieval_ms()
    finally:
        rs.strategy_executor.current_state.record.save_pickle(
            dir / "current_state_record.pkl"
        )
        for f in run_dir.glob("*_obs.pkl"):
            shutil.copy(f, dir / f.name)


@pytest.mark.capture_test
def test_capture_tropomi_cris_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, joint_tropomi_test_in_dir
):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    ihandle = TropomiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    run_capture(rs, r.run_dir, joint_tropomi_test_in_dir)


@pytest.mark.capture_test
def test_capture_tropomi_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, tropomi_test_in_dir
):
    r = MusesRunDir(tropomi_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    ihandle = TropomiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    run_capture(rs, r.run_dir, tropomi_test_in_dir)


@pytest.mark.capture_test
def test_capture_tropomi_band7_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, tropomi_band7_test_in_dir
):
    r = MusesRunDir(tropomi_band7_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    ihandle = TropomiSwirForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    run_capture(rs, r.run_dir, tropomi_band7_test_in_dir)


@pytest.mark.capture_test
def test_capture_omi_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, omi_test_in_dir
):
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    ihandle = OmiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    run_capture(rs, r.run_dir, omi_test_in_dir)


@pytest.mark.capture_test
def test_capture_airs_omi_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, joint_omi_test_in_dir
):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    ihandle = OmiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    run_capture(rs, r.run_dir, joint_omi_test_in_dir)


@pytest.mark.capture_test
def test_capture_airs_irk(isolated_dir, osp_dir, gmao_dir, airs_irk_test_in_dir):
    r = MusesRunDir(airs_irk_test_in_dir, osp_dir, gmao_dir, skip_obs_link=True)
    rs = RetrievalStrategy(None)
    run_capture(rs, r.run_dir, airs_irk_test_in_dir)
