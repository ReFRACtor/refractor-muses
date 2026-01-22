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
from refractor.muses import MusesRunDir
from loguru import logger
import os
from pathlib import Path
import subprocess
import importlib.util
import sys


@pytest.mark.long_executable_test
def test_refractor_retrieve_ml(
    ifile_hlp, cris_ml_dir, cris_ml_test_in_dir, end_to_end_run_dir
):
    """This is a top level running of the machine learning retrieval.
    Note that this runs fairly different than our OE retrievals, rather
    than running on a single sounding to runs on all of them for an input
    file. It isn't exactly clear what the interface should be here. We've
    tried to put this into something similar to the OE retrievals as a
    starting point, but this might change.
    """
    dir = end_to_end_run_dir / "refractor_retrieve_ml"
    # Isn't clear how to handle the ML files. This will perhaps end up in the OSP
    # directory. For now, we pass in an environment variable so this is kind of
    # like MUSES_OSP_PATH
    os.environ["MUSES_ML_PATH"] = str(cris_ml_dir)
    config_file = (
        Path(os.path.dirname(__file__)).parent
        / "sample_config"
        / "refractor_config_ml.py"
    )
    r = MusesRunDir(
        cris_ml_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    # Temp, so we can directly look at stuff. We might just make a separate unit
    # test for this
    if False:
        logger.info(
            f'Running "refractor-retrieve --refractor-config {config_file} --targets {r.run_dir}"'
        )
        subprocess.run(
            f"refractor-retrieve --refractor-config {config_file} --targets {r.run_dir}",
            shell=True,
            check=True,
        )
    else:
        spec = importlib.util.spec_from_file_location("refractor_config", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["refractor_config"] = module
        rs = module.rs
        rs.update_target(f"{r.run_dir}/Table.asc")
        try:
            lognum = logger.add(dir / "retrieve.log")
            rs.retrieval_ms()
        finally:
            logger.remove(lognum)
