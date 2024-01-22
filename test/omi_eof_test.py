import numpy as np
import numpy.testing as npt
from refractor.omi import (OmiFmObjectCreator, RefractorOmiFm,
                           OmiInstrumentHandle)
from test_support import *
import refractor.framework as rf
import glob
from refractor.muses import (RetrievalStrategy, MusesRunDir,
                             RetrievalStrategyCaptureObserver,
                             RefractorMusesIntegration,
                             RetrievalInfo,
                             RetrievalL2Output,
                             StrategyTable,
                             OmiEofStateElement,
                             SingleSpeciesHandle,
                             FmObsCreator, RetrievableStateElement)
import subprocess

class RetrievalStrategyStop:
    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location == "initial set up done"):
            raise StopIteration()
        
@long_test
@require_muses_py
def test_eof_omi(osp_dir, gmao_dir, vlidort_cli,
                 clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r omi_eof", shell=True)
    r = MusesRunDir(omi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="omi_eof")
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    subprocess.run(f'sed -i -e "s/OMINRADWAVUV1/OMINRADWAVUV1,OMIEOFUV1,OMIEOFUV2/" {r.run_dir}/Table.asc', shell=True)
    # For faster turn around time, set number of iterations to 1. We can test
    # everything, even though the final residual will be pretty high
    subprocess.run(f'sed -i -e "s/15/1 /" {r.run_dir}/Table.asc', shell=True)
    
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rs.register_with_muses_py()
    # Save data so we can work on getting output in isolation
    rscap = RetrievalStrategyCaptureObserver("retrieval_step", "retrieval step")
    rs.add_observer(rscap)
    ihandle = OmiInstrumentHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False, use_eof=True)
    rs.instrument_handle_set.add_handle(ihandle, priority_order=100)
    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("OMIEOFUV1", OmiEofStateElement, pass_state=False, name="OMIEOFUV1", number_eof=3))
    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("OMIEOFUV2", OmiEofStateElement, pass_state=False, name="OMIEOFUV2", number_eof=3))
    if(False):
        # We can use CLI and call this like py-retrieve. But this absorbed
        # errors, so it is a bit harder to debug
        r.run_retrieval(vlidort_cli=vlidort_cli)
    else:
        # Instead we can just directly call the function that run_retrieval
        # ends up running, and occasionally test the other interface to make sure
        # nothing weird happens
        rs.retrieval_ms()
    if True:
        # Print out output of EOF, just so we have something to see
        subprocess.run("h5dump -d OMI_EOF_UV1 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
        subprocess.run("h5dump -d OMI_EOF_UV2 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)

@long_test
@require_muses_py
def test_eof_airs_omi(osp_dir, gmao_dir, vlidort_cli,
                 clean_up_replacement_function):
    '''Full run of AIRS/OMI that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.

    This currently depends on a test case Sebastian has, we can either switch
    this to our existing AIRS/OMI test case or pull his data in. But for now,
    just use hard coded paths'''
    subprocess.run("rm -r airs_omi_eof", shell=True)
    r = MusesRunDir('/tb/sandbox17/sval/muses_output_eof_application_single/airs_omi/2022-11-01/setup-targets/Global_Survey_Grid_4.0/20221101_028_009_22',
                    osp_dir, gmao_dir, path_prefix="airs_omi_eof")
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    subprocess.run(f'sed -i -e "s/OMINRADWAVUV1/OMINRADWAVUV1,OMIEOFUV1,OMIEOFUV2/" {r.run_dir}/Table.asc', shell=True)
    
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    rs.register_with_muses_py()
    # Save data so we can work on getting output in isolation
    rscap = RetrievalStrategyCaptureObserver("retrieval_step", "retrieval step")
    rs.add_observer(rscap)
    ihandle = OmiInstrumentHandle(use_pca=False, use_lrad=False,
                                  lrad_second_order=False, use_eof=True,
                                  eof_dir = "/tb/sandbox21/lkuai/muses/output_py/airs_omi/plot/no_softCal_EOF/EOFout")
    rs.instrument_handle_set.add_handle(ihandle, priority_order=100)
    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("OMIEOFUV1", OmiEofStateElement, pass_state=False, name="OMIEOFUV1", number_eof=3))
    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("OMIEOFUV2", OmiEofStateElement, pass_state=False, name="OMIEOFUV2", number_eof=3))
    if(False):
        # We can use CLI and call this like py-retrieve. But this absorbed
        # errors, so it is a bit harder to debug
        r.run_retrieval(vlidort_cli=vlidort_cli)
    else:
        # Instead we can just directly call the function that run_retrieval
        # ends up running, and occasionally test the other interface to make sure
        # nothing weird happens
        rs.retrieval_ms()
    if True:
        # Print out output of EOF, just so we have something to see
        subprocess.run("h5dump -d OMI_EOF_UV1 -A 0 airs_omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
        subprocess.run("h5dump -d OMI_EOF_UV2 -A 0 airs_omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)


# This depends on the previous test. We can skip this test in the future, this
# is useful when we work out the initial output here
@skip
@require_muses_py
def test_eof_l2_output(isolated_dir, vlidort_cli, osp_dir, gmao_dir):
    step_number = 1
    r = MusesRunDir(omi_test_in_dir, osp_dir, gmao_dir)
    pname = f"/home/smyth/Local/omi/omi_eof/20160414_23_394_11_23/retrieval_step_{step_number}.pkl"

    rs, kwarg = RetrievalStrategy.load_retrieval_strategy(pname, vlidort_cli=vlidort_cli)
    rs.clear_observers()
    jout = RetrievalL2Output()
    rs.add_observer(jout)
    rs.notify_update("retrieval step", **kwarg)
    cmd = f"h5ls -r 20160414_23_394_11_23/Products/Products_L2-O3-0.nc"
    subprocess.run(cmd, shell=True)
    cmd = f"h5ls -r 20160414_23_394_11_23/Products/Lite_Products_L2-O3-0.nc"
    subprocess.run(cmd, shell=True)
    
