from __future__ import annotations
import subprocess
from refractor.muses import (
    MusesRunDir,
    RetrievalStrategy,
    InputFileRecord,
)
from loguru import logger
import pytest


@pytest.mark.skip
def test_retrieval_cris_co_ml(
    ml_cris_test_in_dir, cris_ml_dir, end_to_end_run_dir, ifile_hlp
):
    """Full run of CRIS ML for CO column."""
    dir = end_to_end_run_dir / "retrieval_strategy_cris_ml"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        ml_cris_test_in_dir,
        ifile_hlp,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        r.run_dir / "Table.asc",
        ifile_hlp=ifile_hlp,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        # Record input file, just for our information
        rs.input_file_helper.add_observer(InputFileRecord(dir / "input_list.log"))
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
