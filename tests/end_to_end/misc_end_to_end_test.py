# Tests that we ran at one time to diagnose problems etc. Leave these in place as
# examples, although these aren't actually run now.

import pytest
import subprocess
from loguru import logger
from refractor.muses import RetrievalStrategy, MusesRunDir


# Sample of how to load a saved step (by RetrievalStrategyCaptureObserver) and
# start running it again. This  test depends on a specific run, and a hard coded path,
# so we don't normally run. But it is a basic example of how to do this.
@pytest.mark.skip
def test_run_fabiano_refractor(isolated_dir, ifile_hlp, end_to_end_run_dir):
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        end_to_end_run_dir
        / "compare_fabiano_refractor/20200701_204_05_29_0/retrieval_step_10.pkl",
        ifile_hlp=ifile_hlp,
    )
    rs.continue_retrieval()


@pytest.mark.skip
def test_run_fabiano_vlidort(isolated_dir, ifile_hlp, end_to_end_run_dir):
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        end_to_end_run_dir
        / "compare_fabiano_refractor_vlidort/20200701_204_05_29_0/retrieval_step_10.pkl",
        ifile_hlp=ifile_hlp,
    )
    rs.continue_retrieval()


# Used to diagnose a few problems with TES. Leave here in case we need
# to do this again in the future
@pytest.mark.skip
def test_failed_tes(ifile_hlp, python_fp_logger, end_to_end_run_dir):
    """Quick turn around at failing step for tes"""
    dir = end_to_end_run_dir / "failed_tes"
    subprocess.run(["rm", "-r", str(dir)])
    step_number = 12
    rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
        end_to_end_run_dir
        / f"retrieval_strategy_tes/20040920_02147_388_02/retrieval_strategy_retrieval_step_{step_number}.pkl",
        path=dir,
        ifile_hlp=ifile_hlp,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        rs.continue_retrieval(stop_after_step=5)
    finally:
        logger.remove(lognum)


@pytest.mark.skip
@pytest.mark.long_test
def test_original_retrieval_oco2(ifile_hlp, test_base_path, end_to_end_run_dir):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.
    """
    raise RuntimeError(
        """This doesn't currently work. I'm guessing Susan has her own
        branch of py-retrieve to run this. No reason to spend time now
        debugging this, we were just adding OCO-2 to make sure
        refractor works the same with it. But since py-retrieve on the
        develop branch doesn't work with OCO-2, no reason to try to
        get refractor working. refractor already duplicates
        py-retrieve behavior of failing ;->"""
    )
    dir = end_to_end_run_dir / "original_retrieval_oco2"
    subprocess.run(["rm", "-r", str(dir)])
    subprocess.run("rm -r original_retrieval_oco2", shell=True)
    r = MusesRunDir(
        test_base_path / "oco2/in/sounding_1",
        ifile_hlp,
        path_prefix=dir,
    )
    r.run_retrieval()
