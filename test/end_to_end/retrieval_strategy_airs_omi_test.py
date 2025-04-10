import pytest
import subprocess
from fixtures.compare_run import compare_run
from loguru import logger
from refractor.omi import OmiForwardModelHandle
from refractor.muses import (
    RetrievalStrategyCaptureObserver,
    RetrievalStrategy,
    MusesRunDir,
)

# Use refractor forward model, or use py-retrieve.
# Note that there is a separate set of expected results for a refractor run.
# run_refractor = False
run_refractor = True

# Can use the older py_retrieve matching objects
match_py_retrieve = False
# match_py_retrieve = True


@pytest.mark.long_test
def test_retrieval_strategy_airs_omi(
    osp_dir,
    gmao_dir,
    vlidort_cli,
    python_fp_logger,
    end_to_end_run_dir,
    joint_omi_test_in_dir,
    joint_omi_test_refractor_expected_dir,
    joint_omi_test_expected_dir,
):
    """Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises."""
    dir = end_to_end_run_dir / "retrieval_strategy_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir,
        osp_dir,
        gmao_dir,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
        vlidort_cli=vlidort_cli,
        write_omi_radiance_pickle=not run_refractor,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver(
            "retrieval_strategy_retrieval_step", "starting run_step"
        )
        rs.add_observer(rscap)
        rscap2 = RetrievalStrategyCaptureObserver(
            "retrieval_result", "systematic_jacobian"
        )
        rs.add_observer(rscap2)
        compare_dir = joint_omi_test_expected_dir
        if run_refractor:
            # Use refractor forward model.
            ihandle = OmiForwardModelHandle(
                use_pca=True,
                use_lrad=False,
                lrad_second_order=False,
                match_py_retrieve=match_py_retrieve,
            )
            rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
            # Different expected results. Close, but not identical to VLIDORT version
            compare_dir = joint_omi_test_refractor_expected_dir
            rs.update_target(f"{r.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)
