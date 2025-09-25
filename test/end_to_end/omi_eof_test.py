from refractor.omi import OmiForwardModelHandle
from refractor.muses import (
    RetrievalStrategy,
    MusesRunDir,
    RetrievalStrategyCaptureObserver,
    OmiEofStateElementHandle,
    RetrievalStrategyMemoryUse,
    StateElementIdentifier,
    modify_strategy_table,
)
import subprocess
import pytest
from loguru import logger


@pytest.mark.long_test
def test_eof_omi(osp_dir, gmao_dir, omi_test_in_dir, end_to_end_run_dir):
    """Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one."""
    dir = end_to_end_run_dir / "omi_eof"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir, path_prefix=dir)
    rs = RetrievalStrategy(None)
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    # Also, set number of iterations to 1, just to run faster
    modify_strategy_table(
        rs,
        1,
        [
            StateElementIdentifier("O3"),
            StateElementIdentifier("OMICLOUDFRACTION"),
            StateElementIdentifier("OMINRADWAVUV1"),
            StateElementIdentifier("OMIEOFUV1"),
            StateElementIdentifier("OMIEOFUV2"),
            StateElementIdentifier("OMINRADWAVUV2"),
            StateElementIdentifier("OMIODWAVUV1"),
            StateElementIdentifier("OMIODWAVUV2"),
            StateElementIdentifier("OMISURFACEALBEDOUV1"),
            StateElementIdentifier("OMISURFACEALBEDOUV2"),
            StateElementIdentifier("OMISURFACEALBEDOSLOPEUV2"),
        ],
        max_iter=1,
    )
    ihandle = OmiForwardModelHandle(
        use_pca=False, use_lrad=False, lrad_second_order=False, use_eof=True
    )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.state_element_handle_set.add_handle(
        OmiEofStateElementHandle(StateElementIdentifier("OMIEOFUV1"))
    )
    rs.state_element_handle_set.add_handle(
        OmiEofStateElementHandle(StateElementIdentifier("OMIEOFUV2"))
    )
    rs.update_target(r.run_dir / "Table.asc")
    try:
        lognum = logger.add(dir / "retrieve.log")
        rscap = RetrievalStrategyCaptureObserver(
            "retrieval_strategy_retrieval_step", "starting run_step"
        )
        rs.add_observer(rscap)
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    if True:
        # Print out output of EOF, just so we have something to see
        subprocess.run(
            "h5dump -d OMI_EOF_UV1 -A 0 "
            + str(dir / "20160414_23_394_11_23/Products/Products_L2-O3-0.nc"),
            shell=True,
        )
        subprocess.run(
            "h5dump -d OMI_EOF_UV2 -A 0 "
            + str(dir / "20160414_23_394_11_23/Products/Products_L2-O3-0.nc"),
            shell=True,
        )


# Still failing, issue with microwindows. We'll continue working on this in a bit        
@pytest.mark.skip        
@pytest.mark.long_test
def test_eof_airs_omi(osp_dir, gmao_dir, end_to_end_run_dir, joint_omi_eof_test_in_dir):
    """Full run of AIRS/OMI that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This currently depends on a test case Sebastian has, we can either switch
    this to our existing AIRS/OMI test case or pull his data in. But for now,
    just use hard coded paths"""
    dir = end_to_end_run_dir / "airs_omi_eof"
    subprocess.run(["rm", "-r", str(dir)])
    r = MusesRunDir(
        joint_omi_eof_test_in_dir,
        osp_dir,
        gmao_dir,
        path_prefix=dir,
    )
    rs = RetrievalStrategy(
        None,
    )
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    modify_strategy_table(
        rs,
        1,
        [
            StateElementIdentifier("O3"),
            StateElementIdentifier("OMICLOUDFRACTION"),
            StateElementIdentifier("OMINRADWAVUV1"),
            StateElementIdentifier("OMIEOFUV1"),
            StateElementIdentifier("OMIEOFUV2"),
            StateElementIdentifier("OMINRADWAVUV2"),
            StateElementIdentifier("OMIODWAVUV1"),
            StateElementIdentifier("OMIODWAVUV2"),
            StateElementIdentifier("OMISURFACEALBEDOUV1"),
            StateElementIdentifier("OMISURFACEALBEDOUV2"),
            StateElementIdentifier("OMISURFACEALBEDOSLOPEUV2"),
        ],
    )
    # Save data so we can work on getting output in isolation
    rscap = RetrievalStrategyCaptureObserver("retrieval_step", "retrieval step")
    rs.add_observer(rscap)
    # Watch memory usage.
    rsmem = RetrievalStrategyMemoryUse()
    rs.add_observer(rsmem)
    ihandle = OmiForwardModelHandle(
        use_pca=False,
        use_lrad=False,
        lrad_second_order=False,
        use_eof=True,
        eof_dir=joint_omi_eof_test_in_dir / "eof_dir",
    )
    # eof_dir = "./eof_stuff/EOFout")
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.state_element_handle_set.add_handle(
        OmiEofStateElementHandle(StateElementIdentifier("OMIEOFUV1"))
    )
    rs.state_element_handle_set.add_handle(
        OmiEofStateElementHandle(StateElementIdentifier("OMIEOFUV2"))
    )
    rs.update_target(r.run_dir / "Table.asc")
    try:
        lognum = logger.add(dir / "retrieve.log")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)
    if True:
        # Print out output of EOF, just so we have something to see
        subprocess.run(
            "h5dump -d OMI_EOF_UV1 -A 0 "
            + str(dir / "20160414_23_394_11_23/Products/Products_L2-O3-0.nc"),
            shell=True,
        )
        subprocess.run(
            "h5dump -d OMI_EOF_UV2 -A 0 "
            + str(dir / "20160414_23_394_11_23/Products/Products_L2-O3-0.nc"),
            shell=True,
        )
