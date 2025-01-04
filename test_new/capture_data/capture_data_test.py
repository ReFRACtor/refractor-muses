import pytest
from refractor.muses import (
    MusesRunDir,
    RetrievalStepCaptureObserver,
    RetrievalStrategy,
    StateInfoCaptureObserver,
)
from refractor.tropomi import TropomiForwardModelHandle
from refractor.omi import OmiForwardModelHandle


@pytest.mark.capture_test
def test_capture_tropomi_cris_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_tropomi_test_in_dir
):
    r = MusesRunDir(joint_tropomi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    ihandle = TropomiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        joint_tropomi_test_in_dir / "state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        joint_tropomi_test_in_dir / "retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(r.run_dir / "Table.asc")
    rs.retrieval_ms()


@pytest.mark.capture_test
def test_capture_airs_omi_retrieval_strategy(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, joint_omi_test_in_dir
):
    r = MusesRunDir(joint_omi_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    ihandle = OmiForwardModelHandle(
        use_pca=True, use_lrad=False, lrad_second_order=False
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        joint_omi_test_in_dir / "state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        joint_omi_test_in_dir / "retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(r.run_dir / "Table.asc")
    rs.retrieval_ms()


@pytest.mark.capture_test
def test_capture_airs_irk(
    isolated_dir, osp_dir, gmao_dir, vlidort_cli, airs_irk_test_in_dir
):
    r = MusesRunDir(airs_irk_test_in_dir, osp_dir, gmao_dir)
    rs = RetrievalStrategy(None)
    rs.clear_observers()
    rscap = StateInfoCaptureObserver(
        f"{airs_irk_test_in_dir}/state_info_step", "starting run_step"
    )
    rs.add_observer(rscap)
    rscap2 = RetrievalStepCaptureObserver(
        f"{airs_irk_test_in_dir}/retrieval_state_step"
    )
    rs.add_observer(rscap2)
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()
