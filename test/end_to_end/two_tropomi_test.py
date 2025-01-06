import pytest
from refractor.tropomi import TropomiForwardModelHandle
from loguru import logger
from refractor.muses import (
    RetrievalStrategy,
    MusesRunDir,
)

# Use refractor forward model, or use py-retrieve.
# Note that there is a separate set of expected results for a refractor run.
# run_refractor = False
run_refractor = True

# Can use the older py_retrieve matching objects
match_py_retrieve = False
# match_py_retrieve = True


@pytest.mark.long_test
def test_two_tropomi(
    isolated_dir,
    osp_dir,
    gmao_dir,
    vlidort_cli,
    tropomi_test_in_dir,
    tropomi_test_in_dir3,
):
    """Run two soundings, using a single RetrievalStrategy. This is how the
    MPI version of py-retrieve works, and we need to make sure any caching
    etc. gets cleared out from the first sounding to the second."""
    r = MusesRunDir(tropomi_test_in_dir, osp_dir, gmao_dir)
    r2 = MusesRunDir(tropomi_test_in_dir3, osp_dir, gmao_dir, skip_sym_link=True)

    rs = RetrievalStrategy(
        None, writeOutput=True, writePlots=True, vlidort_cli=vlidort_cli
    )
    rs.forward_model_handle_set.add_handle(
        TropomiForwardModelHandle(
            use_pca=True, use_lrad=False, lrad_second_order=False
        ),
        priority_order=100,
    )
    try:
        lognum = logger.add("retrieve.log")
        rs.script_retrieval_ms(r.run_dir / "Table.asc")
        rs.script_retrieval_ms(r2.run_dir / "Table.asc")
    finally:
        logger.remove(lognum)
