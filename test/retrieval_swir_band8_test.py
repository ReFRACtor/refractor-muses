from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy,
                             RetrievalStrategyCaptureObserver,
                             RetrievableStateElement,
                             ForwardModelHandle,
                             SingleSpeciesHandle,
                             SimulatedObservation,
                             SimulatedObservationHandle,
                             StateInfo,
                             RetrievalInfo,
                             CurrentStateUip)
from refractor.tropomi import (TropomiSwirForwardModelHandle,
                               TropomiSwirFmObjectCreator)
import refractor.muses.muses_py as mpy
import refractor.framework as rf
from functools import cached_property
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

@pytest.fixture(scope="function")
def tropomi_co_step(tropomi_swir):
    subprocess.run(f'sed -i -e "s/CO,CH4,H2O,HDO,TROPOMISOLARSHIFTBAND7,TROPOMIRADIANCESHIFTBAND7,TROPOMISURFACEALBEDOBAND7,TROPOMISURFACEALBEDOSLOPEBAND7,TROPOMISURFACEALBEDOSLOPEORDER2BAND7/CO                                                                                                                                                           /" {tropomi_swir.run_dir}/Table.asc', shell=True)
    return tropomi_swir

@long_test
def test_band8_retrieval(tropomi_co_step, josh_osp_dir):
    '''Work through issues to do a band 8 retrieval, without making
    any py-retrieve modifications'''
    rs = RetrievalStrategy(None, osp_dir=josh_osp_dir)
    # Grab each step so we can separately test output
    #rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    #rs.add_observer(rscap)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False,
                                            osp_dir=josh_osp_dir)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_co_step.run_dir}/Table.asc")
    # This doesn't execute yet for band 8. We'll work through issues here by
    # debugging, and put the first problems in the next section to work through
    # them
    if False:
        rs.retrieval_ms()

    # Do all the setup etc., but stop the retrieval at step 0 (i.e., before we
    # do the first retrieval step). We then grab things to check stuff out
    rs.strategy_executor.execute_retrieval(stop_at_step=0)
    flist_dict = rs.strategy_executor.filter_list_dict
    assert flist_dict == {"TROPOMI" : ['BAND8']}
    

