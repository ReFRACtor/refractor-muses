import pytest
import subprocess
from fixtures.compare_run import compare_run
from fixtures.require_check import require_muses_py_fm
from loguru import logger
from refractor.omi import OmiForwardModelHandle
from refractor.muses import (
    RetrievalStrategyCaptureObserver,
    RetrievalStrategy,
    MusesRunDir,
    MusesObservationHandlePickleSave,
    MusesOmiObservation,
    MusesTropomiObservation,
    MusesAirsObservation,
    MusesCrisObservation,
    MusesTesObservation,
    InstrumentIdentifier,
    InputFileRecord,
)
import sys

# Use refractor forward model, or use py-retrieve.
# Note that there is a separate set of expected results for a refractor run.
# run_refractor = False
run_refractor = True

# Can use the older py_retrieve matching objects
match_py_retrieve = False
# match_py_retrieve = True


@require_muses_py_fm
@pytest.mark.long_test
def test_retrieval_strategy_airs_omi(
    ifile_hlp,
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
        ifile_hlp,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
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
        rscap2 = RetrievalStrategyCaptureObserver("retrieval_result", "retrieval step")
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


@require_muses_py_fm
def test_load_step(ifile_hlp, joint_omi_test_in_dir, isolated_dir):
    logger.remove()
    r = MusesRunDir(joint_omi_test_in_dir, ifile_hlp)
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
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
    step_number = 8
    dir = joint_omi_test_in_dir
    rs.load_step_info(
        dir / "current_state_record.pkl",
        step_number,
        ret_state_file=dir / f"retrieval_state_step_{step_number}.json.gz",
    )
    # rs.continue_retrieval(stop_after_step=step_number)
    logger.add(sys.stderr, level="DEBUG")
    # breakpoint()


@require_muses_py_fm
def test_run_pickle_1(end_to_end_run_dir, ifile_hlp):
    """Test loading the pickle files saved in test_retrieval_strategy_airs_omi, rerunning
    step 5. This is an example of how you might run to debug a problem. This is similar
    to the current_state_record.pkl / retrieval_state_step_*.json.gz test above. However
    this is more direct, and something you are more likely to run into while debugging.
    The rs.load_step_info is more robust way of handling unit test data (where we
    want to allow changes in code as we modify things), but a little more complicated.
    """
    # We need to have had test_retrieval_strategy_airs_omi run before this step. Check
    # for the pickle file. If it isn't there, then just skip this test
    pfile = (
        end_to_end_run_dir
        / "retrieval_strategy_airs_omi/20160401_231_049_29/retrieval_strategy_retrieval_step_5.pkl"
    )
    if not pfile.exists():
        pytest.skip("Need to run retrieval_strategy_airs_omi before running this test")

    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        pfile,
        end_to_end_run_dir / "retrieval_strategy_airs_omi_pickle",
        ifile_hlp=ifile_hlp,
    )
    # Run step 5, and then stop
    rs.continue_retrieval(stop_after_step=6)


def test_run_pickle_2(end_to_end_run_dir, ifile_hlp):
    """Test loading the pickle files saved in test_retrieval_strategy_airs_omi, at the
    end of step 5. Examine the results.
    """
    # We need to have had test_retrieval_strategy_airs_omi run before this step. Check
    # for the pickle file. If it isn't there, then just skip this test
    pfile = (
        end_to_end_run_dir
        / "retrieval_strategy_airs_omi/20160401_231_049_29/retrieval_result_5.pkl"
    )
    if not pfile.exists():
        pytest.skip("Need to run retrieval_strategy_airs_omi before running this test")

    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        pfile,
        end_to_end_run_dir / "retrieval_strategy_airs_omi_pickle_2",
        ifile_hlp=ifile_hlp,
    )
    # This get passed on by the notify when we pickle this step. This is the same message
    # that goes to RetrievalOutput. Just check here that we can access this, but you can
    # do whatever here when debugging
    rstep = kwargs["retrieval_strategy_step"]
    print(rstep.results.tropopause_pressure)
