import pytest
import subprocess
from fixtures.compare_run import compare_run
from loguru import logger
from refractor.tropomi import TropomiForwardModelHandle
from refractor.muses import (
    RetrievalStrategyCaptureObserver,
    RetrievalStrategy,
    MusesRunDir,
    InputFileRecord,
)
from fixtures.require_check import require_muses_py_fm

# Use refractor forward model, or use py-retrieve.
# Note that there is a separate set of expected results for a refractor run.
# run_refractor = False
run_refractor = True

# Can use the older py_retrieve matching objects
match_py_retrieve = False
# match_py_retrieve = True


@pytest.mark.long_test
@require_muses_py_fm
def test_retrieval_strategy_cris_tropomi(
    ifile_hlp,
    python_fp_logger,
    end_to_end_run_dir,
    joint_tropomi_test_in_dir,
    joint_tropomi_test_refractor_expected_dir,
    joint_tropomi_test_expected_dir,
):
    """Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises."""
    dir = end_to_end_run_dir / "retrieval_strategy_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
        writeOutput=True,
        writePlots=True,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        # Record input file, just for our information
        rs.input_file_helper.add_observer(InputFileRecord(dir / "input_list.log"))
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver(
            "retrieval_strategy_retrieval_step", "starting run_step"
        )
        rs.add_observer(rscap)
        rscap2 = RetrievalStrategyCaptureObserver(
            "retrieval_result", "systematic_jacobian"
        )
        rs.add_observer(rscap2)
        compare_dir = joint_tropomi_test_expected_dir
        if run_refractor:
            # Use refractor forward model.
            ihandle = TropomiForwardModelHandle(
                use_pca=True,
                use_lrad=False,
                lrad_second_order=False,
                match_py_retrieve=match_py_retrieve,
            )
            rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
            # Different expected results. Close, but not identical to VLIDORT version
            compare_dir = joint_tropomi_test_refractor_expected_dir
            rs.update_target(f"{r.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


# Don't normally run, we test cris in the cris_tropomi test above. But useful
# to run to diagnose problems - we've run into issue before with cris only
# retrievals that don't show up in cris_tropomi
@pytest.mark.skip
@pytest.mark.long_test
@require_muses_py_fm
def test_retrieval_cris(
    ifile_hlp,
    python_fp_logger,
    end_to_end_run_dir,
    cris_test_in_dir,
):
    """Full run, that we then compare the output files to expected results.
    This is not really a unit test, but for convenience we have it here.
    Note that a "failure" in the comparison might not actually indicate a problem, just
    that the output changed. You may need to look into detail and decide that the
    run was successful and we just want to update the expected results.

    Data goes in the local directory, rather than an isolated one. We can change this
    in the future if desired, but  for now it is useful to be able to look into the directory
    if some kind of a problem arises."""
    dir = end_to_end_run_dir / "retrieval_strategy_cris"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        cris_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
        writeOutput=True,
        writePlots=True,
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
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    # Don't bother checking results, we just want to make sure we can run
