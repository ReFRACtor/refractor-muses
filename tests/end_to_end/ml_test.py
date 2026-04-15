from refractor.muses import (
    RetrievalStrategy,
    RetrievalConfiguration,
    InputFileRecord,
)

# Import to grab the registration of objects needed
import refractor.osr_ml  # noqa: F401
import subprocess
from loguru import logger
import pystac


def test_refractor_retrieve_ml_end_to_end(
    ifile_hlp, cris_ml_test_in_dir, end_to_end_run_dir
):
    dir = end_to_end_run_dir / "refractor_retrieve_ml"
    subprocess.run(["rm", "-r", dir])
    subprocess.run(["mkdir", "-p", dir])
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    rs.input_file_helper.add_observer(InputFileRecord(dir / "input_list.log"))
    rconfig = RetrievalConfiguration.create_from_yaml(
        cris_ml_test_in_dir / "retrieval_config.yaml",
        ifile_hlp=rs.input_file_helper,
        output_directory=dir,
    )
    stac_catalog = pystac.Catalog.from_file(cris_ml_test_in_dir / "catalog.json")
    rs.strategy_context.update_strategy_context(
        creator_dict=rs.creator_dict,
        stac_catalog=stac_catalog,
        retrieval_config=rconfig,
        strategy_table_filename=cris_ml_test_in_dir / "strategy.yaml",
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
