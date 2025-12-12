import pytest
import subprocess
from fixtures.compare_run import compare_run
from loguru import logger
from refractor.muses import (
    RetrievalStrategyCaptureObserver,
    RetrievalStrategy,
    MusesRunDir,
)

# Can use the older py_retrieve matching objects
match_py_retrieve = False
# match_py_retrieve = True


@pytest.mark.long_test
def test_retrieval_strategy_tes(
    osp_dir,
    gmao_dir,
    python_fp_logger,
    end_to_end_run_dir,
    tes_test_in_dir,
    tes_test_expected_dir,
):
    """Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises."""
    dir = end_to_end_run_dir / "retrieval_strategy_tes"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        tes_test_in_dir,
        osp_dir,
        gmao_dir,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver(
            "retrieval_strategy_retrieval_step", "starting run_step"
        )
        rs.add_observer(rscap)
        rscap2 = RetrievalStrategyCaptureObserver("retrieval_result", "retrieval step")
        rs.add_observer(rscap2)
        compare_dir = tes_test_expected_dir
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)
