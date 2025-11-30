# This compares the output from the end to end tests with the expected results. You
# don't normally need to run these, the tests already check there output. But when
# updating expected results for looking at differences in detail, it can be useful to
# run these checks without regenerating the data. These tests supply that.
import pytest
from fixtures.compare_run import compare_run

# Use refractor forward model, or use py-retrieve.
# Note that there is a separate set of expected results for a refractor run.
run_refractor = False
# run_refractor = True


@pytest.mark.compare_test
def test_compare_retrieval_cris_tropomi(
    osp_dir,
    gmao_dir,
    end_to_end_run_dir,
    joint_tropomi_test_expected_dir,
    joint_tropomi_test_refractor_expected_dir,
):
    """The test_retrieval_strategy_cris_tropomi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_strategy_cris_tropomi already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    # diff_is_error = True
    dir = end_to_end_run_dir / "retrieval_strategy_cris_tropomi"
    diff_is_error = False
    compare_dir = joint_tropomi_test_expected_dir
    if run_refractor:
        compare_dir = joint_tropomi_test_refractor_expected_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_airs_omi(
    osp_dir,
    gmao_dir,
    end_to_end_run_dir,
    joint_omi_test_expected_dir,
    joint_omi_test_refractor_expected_dir,
):
    """The test_retrieval_strategy_airs_omi already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_strategy_airs_omi already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    # diff_is_error = True
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_strategy_airs_omi"
    compare_dir = joint_omi_test_expected_dir
    if run_refractor:
        compare_dir = joint_omi_test_refractor_expected_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_tes(
    osp_dir, gmao_dir, end_to_end_run_dir, tes_test_expected_dir
):
    """The test_retrieval_strategy_tes already checks the results, but it is nice
    to have a stand alone run that just checks the results. Note that this depends on
    test_retrieval_strategy_tes already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    # diff_is_error = True
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_strategy_tes"
    compare_dir = tes_test_expected_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)


@pytest.mark.compare_test
def test_compare_retrieval_airs_irk(
    osp_dir, gmao_dir, airs_irk_test_expected_dir, end_to_end_run_dir
):
    """The test_retrieval_strategy_airs_irk already checks the results, but it is
    nice to have a stand alone run that just checks the results. Note that this
    depends on
    test_retrieval_strategy_airs_irk already having been run."""
    # Either error if we have any differences if this is True, or if this is False
    # just report differences
    # diff_is_error = True
    diff_is_error = False
    dir = end_to_end_run_dir / "retrieval_strategy_airs_irk"
    compare_dir = airs_irk_test_expected_dir
    compare_run(compare_dir, dir, diff_is_error=diff_is_error)
