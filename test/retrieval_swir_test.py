from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy,
                             RetrievalStrategyCaptureObserver,
                             RetrievableStateElement,
                             SingleSpeciesHandle,
                             StateInfo,
                             RetrievalInfo,
                             CurrentStateUip)
from refractor.tropomi import TropomiSwirForwardModelHandle
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

@pytest.fixture(scope="function")
def tropomi_co_step(tropomi_swir):
    subprocess.run(f'sed -i -e "s/CO,CH4,H2O,HDO,TROPOMISOLARSHIFTBAND7,TROPOMIRADIANCESHIFTBAND7,TROPOMISURFACEALBEDOBAND7,TROPOMISURFACEALBEDOSLOPEBAND7,TROPOMISURFACEALBEDOSLOPEORDER2BAND7/CO                                                                                                                                                           /" {tropomi_swir.run_dir}/Table.asc', shell=True)
    return tropomi_swir

@long_test
@require_muses_py
def test_retrieval(tropomi_co_step):
    rs = RetrievalStrategy(None)
    # Grab each step so we can separately test output
    #rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    #rs.add_observer(rscap)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_co_step.run_dir}/Table.asc")
    rs.retrieval_ms()

class CostFunctionCapture:
    '''Grab cost function, and then raise a exception to break out of retrieval.'''
    def __init__(self):
        self.location_to_capture = "create_cost_function"

    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        if(location != self.location_to_capture):
            return
        self.cost_function = retrieval_strategy_step.cfunc
        raise StopIteration()

class PrintSpectrum(rf.ObserverPtrNamedSpectrum):

    def notify_update(self, o):
        print("---------")
        print(o.name)
        print(o.spectral_domain.wavelength("nm"))
        print(o.spectral_range.data)
        print("---------")
    
# Look just at the forward model        
@long_test
@require_muses_py
def test_co_fm(tropomi_co_step):
    '''Look just at the forward model'''
    # This is slightly convoluted, but we want to make sure we have the cost
    # function/ForwardModel that is being used in the retrieval. So we 
    # start running the retrieval, and then stop when have the cost function.
    rs = RetrievalStrategy(None)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_co_step.run_dir}/Table.asc")
    cfcap = CostFunctionCapture()
    rs.add_observer(cfcap)
    try:
        rs.retrieval_ms()
    except StopIteration:
        pass
    cfunc = cfcap.cost_function
    fm_sv = cfunc.fm_sv
    fm = cfunc.fm_list[0]
    fm.underlying_forward_model.add_observer_and_keep_reference(PrintSpectrum())
    obs = cfunc.obs_list[0]
    #spec = fm.radiance_all()
    p = cfunc.parameters
    print(p)
    absorber = fm.underlying_forward_model.radiative_transfer.lidort.atmosphere.absorber
    # VMR values, we'll want to make sure this actually changes
    vmr_val = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    residual = copy.copy(cfunc.residual)
    print(vmr_val)
    coeff = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print(absorber.absorber_vmr("CO").coefficient.value)
    
    # After the first step in levmar_nllsq, this is the parameter. We just got
    # this by setting a breakpoint in levmar_nllsq and looking. But this is enough
    # for us to try to figure out what is going on
    p2 = [-15.74241805, -16.20146017, -16.16204205, -16.18824768, -16.24048363,
          -16.3996463,  -16.59970071, -16.79074649, -16.9677597,  -17.49288014,
          -17.79474739, -17.46974549, -17.36741773]
    cfunc.parameters = p2
    residual2 = copy.copy(cfunc.residual)
    print(p-p2)
    vmr_val2 = absorber.absorber_vmr("CO").vmr_profile
    # Very small change
    print(vmr_val-vmr_val2)
    coeff2 = absorber.absorber_vmr("CO").coefficient.value
    print(coeff2-coeff)
    # This is the portion that isn't the parameter constraint
    print((residual2-residual)[:112])

    # Force stale cache
    absorber.notify_update(absorber.absorber_vmr("CO"))
    residual3 = copy.copy(cfunc.residual)
    print((residual3-residual)[:112])

    # All zero. Why?
    breakpoint()

    

    
