from refractor.muses import (
    RetrievalStrategy,
    RetrievalConfiguration,
    MusesSpectralWindowDict,
    RetrievalStrategyStep,
    RetrievalType,
)
from refractor.osr_ml import (
    DummySpectralWindowHandle,
    RetrievalStrategyStepMl,
    RetrievalStrategyStepMlHandle,
    RetrievalMlOutput,
)
from loguru import logger
import os
import pystac


def test_refractor_retrieve_ml_end_to_end(
    ifile_hlp, cris_ml_dir, cris_ml_test_in_dir, end_to_end_run_dir
):
    dir = end_to_end_run_dir / "refractor_retrieve_ml"
    # Isn't clear how to handle the ML files. This will perhaps end up in the OSP
    # directory. For now, we pass in an environment variable so this is kind of
    # like MUSES_OSP_PATH
    os.environ["MUSES_ML_PATH"] = str(cris_ml_dir)
    rs = RetrievalStrategy(None, ifile_hlp=ifile_hlp)
    # TODO I believe if we rework current step, this can go away
    rs.creator_dict[MusesSpectralWindowDict].add_handle(
        DummySpectralWindowHandle(), priority_order=1
    )
    rs.creator_dict[RetrievalStrategyStep].add_handle(
        RetrievalStrategyStepMlHandle(
            RetrievalStrategyStepMl, {RetrievalType("ml")}, "CRIS-JPSS-1", "CO"
        ),
        priority_order=1,
    )
    rs.add_observer(RetrievalMlOutput())

    # We want to get this into retrieval_strategy, but have here as we figure this out
    rs._filename = cris_ml_test_in_dir / "Table.asc"  # noqa: SLF001
    rs._capture_directory.rundir = dir.absolute()  # noqa: SLF001
    stac = pystac.Catalog.from_file(
        cris_ml_test_in_dir / "catalog.json",
    )
    rconf = RetrievalConfiguration.create_from_strategy_file(
        cris_ml_test_in_dir / "Table.asc",
        ifile_hlp,
    )
    rconf["output_directory"] = dir
    rconf["LMDelta"] = "-999 -999"
    rconf["ConvTolerance_CostThresh"] = "-999.0"
    rconf["ConvTolerance_pThresh"] = "-999.0"
    rconf["ConvTolerance_JacThresh"] = "-999.0"
    rs.strategy_context.update_strategy_context(
        stac_catalog=stac,
        retrieval_config=rconf,
        strategy_table_filename=cris_ml_test_in_dir / "Table.asc",
        creator_dict=rs.creator_dict,
    )
    try:
        lognum = logger.add(dir / "retrieve.log")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
