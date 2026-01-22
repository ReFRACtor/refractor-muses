# Top level executables. Don't need to normally run these, we check all the calculations
# at a lower level. But useful every once in a while to make sure this isn't a problem with
# the cli or something like that.
#
# Note, unlike most of our tests these shouldn't be run in parallel (e.g., using -n 10 or
# whatever). Since these are full runs, each one uses a lot of resources. We were regularly
# seeing load of > 150. Instead, just run this like
#
# pytest -s tests/executable --run-long-executable
#
# or something like that
import subprocess
import pytest
from refractor.muses import MusesRunDir
from pathlib import Path
from loguru import logger
import os


@pytest.mark.long_executable_test
def test_refractor_retrieve_airs_omi_modify(
    ifile_hlp,
    joint_omi_test_in_dir,
    end_to_end_run_dir,
):
    """Full run of the refractor-retrieve code, using the
    ReFRACtor forward models. used to make sure we didn't break
    anything. We just check that we can run, we don't bother checking
    the output.  That is done at lower levels of testing.

    """
    dir = end_to_end_run_dir / "refractor_retrieve_airs_omi_modify"
    # Sample configuration file that we have modified the NH3 initial guess
    config_file = (
        Path(os.path.dirname(__file__)).parent
        / "sample_config"
        / "refractor_config_modify_nh3.py"
    )
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    logger.info(
        f'Running "refractor-retrieve --refractor-config {config_file} --targets {r.run_dir}"'
    )
    subprocess.run(
        f"refractor-retrieve --refractor-config {config_file} --targets {r.run_dir}",
        shell=True,
        check=True,
    )
