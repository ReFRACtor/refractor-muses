from refractor.muses import (
    ProcessLocation,
    InstrumentIdentifier,
    MusesTesObservation,
    MusesObservationHandlePickleSave,
    MusesTropomiObservation,
    MusesCrisObservation,
    MusesOmiObservation,
    MusesAirsObservation,
    RetrievalStrategy,
    MusesRunDir,
    RetrievalStepCaptureObserver,
)
from refractor.tropomi import TropomiForwardModelHandle
from refractor.omi import OmiForwardModelHandle
import subprocess
from fixtures.compare_run import compare_run

import pytest
from pathlib import Path

# ---------------------------------------------------------------------
# Note that it is hard to fully test the output generation, the code is
# very coupled. It would be good to rewrite this at some point, but for
# now we really can only fully test everything in full runs.
#
# These tests run through full retrievals, except we use our saved results
# to skip actually do the retrieval step. These run much faster, although
# these still take a minute or so to run so we have these marked as long
# tests.
#
# Note that this checks pretty much all the observers attached to a
# RetrievalStrategy, these are full retrievals except we "cheat" and
# grab our saved retrieval results.
#
# Note that the saved retrieval results can get a little out of date with
# a real retrieval. This is perfectly fine for these tests, since we are
# just checking that everything flows through correctly. So we have
# a *separate* saved expected results relative to things like
# retrieval_strategy_compare_test. These should be pretty similar, but
# aren't necessarily identical.
# ---------------------------------------------------------------------


class RsSetupRetState:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.step_number = 0

    def notify_update(self, retrieval_strategy, loc, **kwargs):
        if loc != ProcessLocation("starting run_step"):
            return
        t = RetrievalStepCaptureObserver.load_retrieval_state(
            self.directory / f"retrieval_state_step_{self.step_number}.json.gz"
        )
        retrieval_strategy.strategy_executor.kwargs["ret_state"] = t
        self.step_number += 1


def run_canned_results(directory: Path, input_directory: Path, ifile_hlp) -> None:
    r = MusesRunDir(input_directory, ifile_hlp, path_prefix=directory)
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
    # Use refractor forward model.
    ihandle = TropomiForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
        match_py_retrieve=False,
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    ihandle = OmiForwardModelHandle(
        use_pca=True,
        use_lrad=False,
        lrad_second_order=False,
        match_py_retrieve=False,
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.add_observer(RsSetupRetState(input_directory))
    rs.update_target(r.run_dir / "Table.asc")
    rs.retrieval_ms()


@pytest.mark.compare_test
def test_compare_retrieval_output_cris_tropomi(
    ifile_hlp,
    end_to_end_run_dir,
    joint_tropomi_test_expected_retrieval_output_dir,
):
    """The test_retrieval_output_cris_tropomi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_output_cris_tropomi already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_output_cris_tropomi"
    compare_dir = joint_tropomi_test_expected_retrieval_output_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_output_airs_omi(
    ifile_hlp,
    end_to_end_run_dir,
    joint_omi_test_expected_retrieval_output_dir,
):
    """The test_retrieval_output_airs_omi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_output_airs_omi already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_output_airs_omi"
    compare_dir = joint_omi_test_expected_retrieval_output_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_output_airs_irk(
    ifile_hlp,
    end_to_end_run_dir,
    airs_irk_test_expected_retrieval_output_dir,
):
    """The test_retrieval_output_airs_irk already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_output_airs_irk already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_output_airs_irk"
    compare_dir = airs_irk_test_expected_retrieval_output_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_output_tes(
    ifile_hlp,
    end_to_end_run_dir,
    tes_test_expected_retrieval_output_dir,
):
    """The test_retrieval_output_tes already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_output_tes already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_output_tes"
    compare_dir = tes_test_expected_retrieval_output_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.long_test
def test_retrieval_output_cris_tropomi(
    end_to_end_run_dir,
    joint_tropomi_test_in_dir,
    joint_tropomi_test_expected_retrieval_output_dir,
    ifile_hlp,
):
    dir = end_to_end_run_dir / "retrieval_output_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    run_canned_results(dir, joint_tropomi_test_in_dir, ifile_hlp)
    compare_dir = joint_tropomi_test_expected_retrieval_output_dir
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.long_test
def test_retrieval_output_airs_omi(
    end_to_end_run_dir,
    joint_omi_test_in_dir,
    joint_omi_test_expected_retrieval_output_dir,
    ifile_hlp,
):
    dir = end_to_end_run_dir / "retrieval_output_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    run_canned_results(dir, joint_omi_test_in_dir, ifile_hlp)
    compare_dir = joint_omi_test_expected_retrieval_output_dir
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.long_test
def test_retrieval_output_airs_irk(
    end_to_end_run_dir,
    airs_irk_test_in_dir,
    airs_irk_test_expected_retrieval_output_dir,
    ifile_hlp,
):
    dir = end_to_end_run_dir / "retrieval_output_airs_irk"
    subprocess.run(["rm", "-r", str(dir)])
    run_canned_results(dir, airs_irk_test_in_dir, ifile_hlp)
    compare_dir = airs_irk_test_expected_retrieval_output_dir
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.long_test
def test_retrieval_output_tes(
    end_to_end_run_dir,
    tes_test_in_dir,
    tes_test_expected_retrieval_output_dir,
    ifile_hlp,
):
    dir = end_to_end_run_dir / "retrieval_output_tes"
    subprocess.run(["rm", "-r", str(dir)])
    run_canned_results(dir, tes_test_in_dir, ifile_hlp)
    compare_dir = tes_test_expected_retrieval_output_dir
    diff_is_error = True
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)
