from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy, RetrievalStrategyCaptureObserver,
                             CurrentStateUip)
from refractor.tropomi import TropomiForwardModelHandle
import refractor.muses.muses_py as mpy
import subprocess
import pprint
import glob
import shutil
import copy
from loguru import logger

# Probably move this into test_support later, but for now keep here until
# we have everything worked out
@pytest.fixture(scope="function")
def tropomi_swir(isolated_dir, gmao_dir, josh_osp_dir):
    r = MusesRunDir(tropomi_band7_test_in_dir2, josh_osp_dir, gmao_dir)
    return r

# Not ready yet
@skip
def test_retrieval(tropomi_swir):
    try:
        lognum = logger.add(f"{tropomi_swir.run_dir}/retrieve.log")
        rs = RetrievalStrategy(None, writeOutput=True,
                               writePlots=True)
        # Grab each step so we can separately test output
        rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
        rs.add_observer(rscap)
        ihandle = TropomiForwardModelHandle(use_pca=False, use_lrad=True,
                                            lrad_second_order=True)
        rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
        rs.update_target(f"{tropomi_swir.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)

    
