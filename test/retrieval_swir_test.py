from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy,
                             RetrievalStrategyCaptureObserver,
                             RetrievableStateElement,
                             SingleSpeciesHandle,
                             StateInfo,
                             RetrievalInfo,
                             CurrentStateUip)
from refractor.tropomi import TropomiForwardModelHandle
import refractor.muses.muses_py as mpy
import refractor.framework as rf
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

@long_test
@require_muses_py
def test_retrieval(tropomi_swir):
    subprocess.run(f'sed -i -e "s/CO,CH4,H2O,HDO,TROPOMISOLARSHIFTBAND7,TROPOMIRADIANCESHIFTBAND7,TROPOMISURFACEALBEDOBAND7,TROPOMISURFACEALBEDOSLOPEBAND7,TROPOMISURFACEALBEDOSLOPEORDER2BAND7/CO                                                                                                                                                           /" {tropomi_swir.run_dir}/Table.asc', shell=True)
    try:
        lognum = logger.add(f"{tropomi_swir.run_dir}/retrieve.log")
        rs = RetrievalStrategy(None, writeOutput=True,
                               writePlots=True)
        # Grab each step so we can separately test output
        #rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
        #rs.add_observer(rscap)
        ihandle = TropomiForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False)
        rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
        rs.update_target(f"{tropomi_swir.run_dir}/Table.asc")
        rs.retrieval_ms()
    finally:
        logger.remove(lognum)

    
