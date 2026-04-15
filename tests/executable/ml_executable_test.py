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

import pytest
from loguru import logger
import subprocess


@pytest.mark.long_executable_test
def test_refractor_retrieve_ml(ifile_hlp, cris_ml_test_in_dir, end_to_end_run_dir):
    """This is a top level running of the machine learning retrieval.
    Note that this runs fairly different than our OE retrievals, rather
    than running on a single sounding to runs on all of them for an input
    file. It isn't exactly clear what the interface should be here. We've
    tried to put this into something similar to the OE retrievals as a
    starting point, but this might change.
    """
    dir = end_to_end_run_dir / "refractor_retrieve_ml"
    subprocess.run(["rm", "-r", str(dir)])
    subprocess.run(["mkdir", "-p", str(dir)])
    logger.info(
        f'Running "refractor-retrieve stac {cris_ml_test_in_dir}/retrieval_config.yaml {cris_ml_test_in_dir}/strategy.yaml {cris_ml_test_in_dir}/catalog.json {dir}"'
    )
    subprocess.run(
        f"refractor-retrieve stac {cris_ml_test_in_dir}/retrieval_config.yaml {cris_ml_test_in_dir}/strategy.yaml {cris_ml_test_in_dir}/catalog.json {dir}",
        shell=True,
        check=True,
    )
