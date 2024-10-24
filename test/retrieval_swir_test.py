from test_support import *
from refractor.muses import (MusesRunDir, RetrievalStrategy,
                             RetrievalStrategyCaptureObserver,
                             RetrievableStateElement,
                             SingleSpeciesHandle,
                             SimulatedObservation,
                             SimulatedObservationHandle,
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
def test_retrieval(tropomi_co_step, josh_osp_dir):
    rs = RetrievalStrategy(None, osp_dir=josh_osp_dir)
    # Grab each step so we can separately test output
    #rscap = RetrievalStrategyCaptureObserver("retrieval_step", "starting run_step")
    #rs.add_observer(rscap)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False,
                                            osp_dir=josh_osp_dir)
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
    def __init__(self):
        super().__init__()
        self.data = {}
        
    def notify_update(self, o):
        if(o.name not in self.data):
            self.data[o.name] = []
        self.data[o.name].append([o.spectral_domain.wavelength("nm"), o.spectral_range.data])
        print("---------")
        print(o.name)
        print(o.spectral_domain.wavelength("nm"))
        print(o.spectral_range.data)
        print("---------")
    
# Look just at the forward model        
@long_test
@require_muses_py
def test_co_fm(tropomi_co_step, josh_osp_dir):
    '''Look just at the forward model'''
    # This is slightly convoluted, but we want to make sure we have the cost
    # function/ForwardModel that is being used in the retrieval. So we 
    # start running the retrieval, and then stop when have the cost function.
    rs = RetrievalStrategy(None, osp_dir=josh_osp_dir)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False,
                                            osp_dir=josh_osp_dir,
                                            # absorption_gases=["CO",]
                                            )
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{tropomi_co_step.run_dir}/Table.asc")
    cfcap = CostFunctionCapture()
    rs.add_observer(cfcap)
    try:
        rs.retrieval_ms()
    except StopIteration:
        pass
    cfunc = cfcap.cost_function
    # Save in case we want to access directly
    pickle.dump(cfunc,open("cfunc.pkl", "wb"))    
    fm_sv = cfunc.fm_sv
    fm = cfunc.fm_list[0]
    pspec = PrintSpectrum()
    fm.underlying_forward_model.add_observer(pspec)
    obs = cfunc.obs_list[0]
    #spec = fm.radiance_all()
    p = cfunc.parameters
    print("p: ", p)
    atmosphere = fm.underlying_forward_model.radiative_transfer.lidort.atmosphere
    absorber = atmosphere.absorber
    # VMR values, we'll want to make sure this actually changes
    vmr_val = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    #residual = copy.copy(cfunc.residual)
    print("vmr_val: ", vmr_val)
    coeff = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff: ", coeff)
    hresgrid_wn = fm.underlying_forward_model.spectral_grid.high_resolution_grid(0).wavenumber()
    od = np.vstack([np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis,:,:] for wn in hresgrid_wn])
    tod = np.vstack([atmosphere.optical_properties(wn,0).total_optical_depth().value for wn in hresgrid_wn])
    
    # After the first step in levmar_nllsq, this is the parameter. We just got
    # this by setting a breakpoint in levmar_nllsq and looking. But this is enough
    # for us to try to figure out what is going on
    p2 = [-15.74241805, -16.20146017, -16.16204205, -16.18824768, -16.24048363,
          -16.3996463,  -16.59970071, -16.79074649, -16.9677597,  -17.49288014,
          -17.79474739, -17.46974549, -17.36741773]
    cfunc.parameters = p2
    #residual2 = copy.copy(cfunc.residual)
    print("p-p2: ", p-p2)
    vmr_val2 = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    # Very small change
    print("vmr-vmr_val2: ", vmr_val-vmr_val2)
    coeff2 = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff2-coeff: ", coeff2-coeff)
    od2 = np.vstack([np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis,:,:] for wn in hresgrid_wn])
    tod2 = np.vstack([atmosphere.optical_properties(wn,0).total_optical_depth().value for wn in hresgrid_wn])
    print("od-od2: ", np.abs(od-od2)[:,:,1].max())

    # Make a big change
    p3 = np.array(p2)*0.75
    cfunc.parameters=p3
    coeff3 = copy.copy(absorber.absorber_vmr("CO").coefficient.value)
    print("coeff3 - coeff2: ", coeff3- coeff2)
    vmr_val3 = copy.copy(absorber.absorber_vmr("CO").vmr_profile)
    print("vmr_val3 - vmr_val2: ", vmr_val3 - vmr_val2)
    od3 = np.vstack([np.vstack(absorber.optical_depth_each_layer(wn, 0).value)[np.newaxis,:,:] for wn in hresgrid_wn])
    tod3 = np.vstack([atmosphere.optical_properties(wn,0).total_optical_depth().value for wn in hresgrid_wn])
    print("od3-od2: ", np.abs(od3-od2)[:,:,1].max())
    #residual3 = copy.copy(cfunc.residual)
    #print("residual3-residual: ", residual3-residual)
    print("tod3-tod: ", np.abs(tod3-tod).max())
    

    # This is the portion that isn't the parameter constraint
    #print((residual2-residual)[:112])

    # Difference high resolution clear spectrum
    #print(np.abs(pspec.data['high_res_rt'][2][1] - pspec.data['high_res_rt'][0][1]).max())
    # All zero. Why?
    breakpoint()

@long_test
@require_muses_py
def test_simulated_retrieval(gmao_dir, josh_osp_dir):
    '''Do a simulation, and then a retrieval to get this result'''
    subprocess.run("rm -f -r swir_simulation", shell=True)
    mrdir = MusesRunDir(tropomi_band7_test_in_dir2, josh_osp_dir, gmao_dir,
                        path_prefix = "./swir_simulation")
    subprocess.run(f'sed -i -e "s/CO,CH4,H2O,HDO,TROPOMISOLARSHIFTBAND7,TROPOMIRADIANCESHIFTBAND7,TROPOMISURFACEALBEDOBAND7,TROPOMISURFACEALBEDOSLOPEBAND7,TROPOMISURFACEALBEDOSLOPEORDER2BAND7/CO                                                                                                                                                           /" {mrdir.run_dir}/Table.asc', shell=True)
    rs = RetrievalStrategy(None,osp_dir=josh_osp_dir)
    ihandle = TropomiSwirForwardModelHandle(use_pca=True, use_lrad=False,
                                            lrad_second_order=False,
                                            osp_dir=josh_osp_dir)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.update_target(f"{mrdir.run_dir}/Table.asc")
    
    # Do all the setup etc., but stop the retrieval at step 0 (i.e., before we
    # do the first retrieval step). We then grab the CostFunction for that step,
    # which we can use for simulation purposes.
    rs.strategy_executor.execute_retrieval(stop_at_step=0)
    cfunc = rs.strategy_executor.create_cost_function()
    pickle.dump(cfunc,open("swir_simulation/cfunc_initial_guess.pkl","wb"))

    # Get the log vmr values set in the state vector. This is the initial guess.
    # For purposes of a simulation, we will say the "right" answer is to reduce the
    # VMR by 25%. So calculate the "true" log vmr and update the cost function with
    # this set of parameters.
    vmr_log_initial = cfunc.parameters
    vmr_initial = np.exp(vmr_log_initial)
    vmr_true = 0.75 * vmr_initial
    vmr_log_true = np.log(vmr_true)
    cfunc.parameters = vmr_log_true

    # Run forward model and get "true" radiance.
    rad_true = [cfunc.fm_list[0].radiance(0, True).spectral_range.data,]
    obs_sim = SimulatedObservation(cfunc.obs_list[0], rad_true)
    pickle.dump(obs_sim, open("swir_simulation/obs_sim.pkl", "wb"))

    # Have simulated observation, and do retrieval
    ohandle = SimulatedObservationHandle(
        "TROPOMI", pickle.load(open("swir_simulation/obs_sim.pkl", "rb")))
    rs.observation_handle_set.add_handle(ohandle, priority_order=100)
    rs.update_target(f"{mrdir.run_dir}/Table.asc")
    rs.retrieval_ms()
    
    
    
    
    
    







    

    
