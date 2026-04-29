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
import pytest


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


# We don't normally run this. Probably turn this into a capture test at some point, but
# for now just skip
@pytest.mark.skip
def test_cris_for_ml_end_to_end(
    ifile_hlp,
    cris_ml_test_in_dir,
    end_to_end_run_dir,
    test_base_path,
    gmao_real_dir,
    osp_real_dir,
):
    """Run amuse-me to generate CRIS output, so we can compare various fields to things
    we populate in ML run."""
    dir = end_to_end_run_dir / "cris_run"
    subprocess.run(["rm", "-r", dir])
    test_pipeline_config = dir / "test_pipeline_config"
    subprocess.run(["amuse-config", "export", "-p", test_pipeline_config], check=True)
    subprocess.run(
        [
            "sed",
            "-i",
            f"s#/project/muses/input#{test_base_path}/fake_input#g",
            test_pipeline_config / "sensor_config.yml",
        ],
        check=True,
    )
    subprocess.run(
        [
            "sed",
            "-i",
            f"s#/project/muses/input/geos_fp_it#{gmao_real_dir}#g",
            test_pipeline_config / "setup_targets.yml",
        ],
        check=True,
    )
    subprocess.run(
        [
            "sed",
            "-i",
            "s#/py-geolocate#/py-geolocate\\n  skip: true\\n#g",
            test_pipeline_config / "geolocate.yml",
        ],
        check=True,
    )
    subprocess.run(
        [
            "amuse-me",
            "--pipeline-config",
            test_pipeline_config / "config.yml",
            "--output",
            dir,
            "--OSP",
            osp_real_dir,
            "--clear-output",
            "--hosts",
            "localhost",
            "--tasks-per-host",
            "8",
            "--start-step",
            "geolocate",
            "--end-step",
            "retrieve",
            "--sensor-set",
            "CRIS",
            "--profile",
            "Global_Survey",
            "--date",
            "2019-08-07",
        ],
        check=True,
    )
