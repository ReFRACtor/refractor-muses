# These tests were run to generate the original test data we compared against, by
# running py-retrieve. We've moved away from this data, there are differences in
# ReFRACtor which have been examined and found to be ok. So we don't normally run
# these tests. But go ahead and save the original runs in case we need to come back to this
# for some reason
import pytest
import subprocess
from refractor.muses import MusesRunDir


# This test was used to generate the original test data using py-retrieve. We have
# tweaked the expected output slightly for test_retrieval_strategy_cris_tropomi (so
# minor round off differeneces). But leave this code around, it can still be useful to
# have the full original run if we need to dig into and issue with
# test_retrieval_strategy_cris_tropomi. But note the output isn't identical, just pretty
# close.
@pytest.mark.skip
@pytest.mark.long_test
def test_original_retrieval_cris_tropomi(
    ifile_hlp, end_to_end_run_dir, joint_tropomi_test_in_dir
):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our RefractorMusesIntegration, but muses-py version of
    script_retrieval_ms. There are pretty minor differences (which we have checked out
    separately) with the original muses-py run - we just pull these out here so
    we can focus on any difference with our RetrievalStrategy"""
    dir = end_to_end_run_dir / "original_retrieval_cris_tropomi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_tropomi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    # rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    # rmi.register_with_muses_py()
    r.run_retrieval(debug=True, plots=True)


# This test was used to generate the original test data using py-retrieve. We have
# tweaked the expected output slightly for test_retrieval_strategy_airs_omi (so
# minor round off differeneces). But leave this code around, it can still be useful to
# have the full original run if we need to dig into and issue with
# test_retrieval_strategy_airs_omi. But note the output isn't identical, just pretty
# close.
@pytest.mark.skip
@pytest.mark.long_test
def test_original_retrieval_airs_omi(
    ifile_hlp, end_to_end_run_dir, joint_omi_test_in_dir
):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This uses our RefractorMusesIntegration, but muses-py version of
    script_retrieval_ms. There are pretty minor differences (which we have checked out
    separately) with the original muses-py run - we just pull these out here so
    we can focus on any difference with our RetrievalStrategy"""
    dir = end_to_end_run_dir / "original_retrieval_airs_omi"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    # rmi = RefractorMusesIntegration(vlidort_cli=vlidort_cli, save_debug_data=True)
    # rmi.register_with_muses_py()
    r.run_retrieval()


@pytest.mark.skip
@pytest.mark.long_test
def test_original_retrieval_tes(ifile_hlp, end_to_end_run_dir, test_base_path):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.
    """
    dir = end_to_end_run_dir / "original_retrieval_tes"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        test_base_path / "tes/in/sounding_1",
        ifile_hlp,
        path_prefix=dir,
    )
    r.run_retrieval()


@pytest.mark.skip
@pytest.mark.long_test
def test_original_retrieval_airs_irk(ifile_hlp, end_to_end_run_dir, test_base_path):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    This is a version 8 IRK ocean retrieval.

    Data goes in the local directory, rather than an isolated one.
    """
    dir = end_to_end_run_dir / "original_retrieval_airs_irk"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        test_base_path / "airs_omi/in/sounding_1_irk",
        ifile_hlp,
        path_prefix=dir,
    )
    r.run_retrieval()
