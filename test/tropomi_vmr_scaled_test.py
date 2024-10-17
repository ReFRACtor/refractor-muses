from functools import cached_property
import numpy as np
import numpy.testing as npt
from refractor.tropomi import TropomiFmObjectCreator
from test_support import *
import refractor.framework as rf
import glob
from refractor.muses import (RetrievalStrategy, MusesRunDir,
                             RetrievalStrategyCaptureObserver,
                             ForwardModelHandle,
                             SingleSpeciesHandle, O3Absorber,
                             RetrievableStateElement, StateInfo,
                             RetrievalInfo)
import subprocess
from loguru import logger

class O3ScaledStateElement(RetrievableStateElement):
    '''Note that we may rework this. Not sure how much we need specific
    StateElement vs. handling a class of them. But for now, we have
    the O3 scaled as a separate StateElement as we work out what exactly we
    want to do with new ReFRACtor only StateElement.

    We can use the SingleSpeciesHandle to add this in, e.g.,

    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("O3_SCALED", O3SCaledStateElement, pass_state=False))
    '''
    def __init__(self, state_info : StateInfo, name="O3_SCALED"):
        super().__init__(state_info, name)
        self._value = np.array([1.0,])
        self._constraint = self._value.copy()

    def sa_covariance(self):
        '''Return sa covariance matrix, and also pressure. This is what
        ErrorAnalysis needs.'''
        # TODO, Double check this. Not sure of the connection between this
        # and the constraintMatrix. Note the pressure is right, this
        # indicates we aren't on levels so we don't need a pressure
        return np.diag([10*10.0] * 1), [-999.0] * 1

    @property
    def value(self):
        return self._value

    def should_write_to_l2_product(self, instruments):
        breakpoint()
        if "TROPOMI" in instruments:
            return True
        return False
    
    def net_cdf_variable_name(self):
        # Want names like OMI_EOF_UV1
        return self.name
    
    def net_cdf_struct_units(self):
        '''Returns the attributes attached to a netCDF write out of this
        StateElement.'''
        return {'Longname': "O3 VMR scale factor", 'Units': '', 'FillValue': '',
                'MisingValue': ''}

    def update_state_element(self, 
                             retrieval_info: RetrievalInfo,
                             results_list:np.array,
                             update_next: bool,
                             retrieval_config : 'RetrievalConfiguration',
                             step : int,
                             do_update_fm : np.array):
        # If we are requested not to update the next step, then save a copy
        # of this to reset the value
        if(not update_next):
            self.state_info.next_state[self.name] = self.clone_for_other_state()
        self._value = results_list[retrieval_info.species_list==self._name]

    def update_initial_guess(self, current_strategy_step : 'CurrentStrategyStep',
                             swin : 'dict(str,MusesSpectralWindow)'):
        self.mapType = 'linear'
        self.pressureList = np.full((1,), -2.0)
        self.altitudeList  = np.full((1,), -2.0)
        self.pressureListFM = self.pressureList
        self.altitudeListFM = self.altitudeList
        # Apriori
        self.constraintVector = self._constraint.copy()
        # Normally the same as apriori, but doesn't have to be
        self.initialGuessList = self.value.copy()
        self.trueParameterList = np.zeros((1))
        self.constraintVectorFM = self.constraintVector
        self.initialGuessListFM = self.initialGuessList
        self.trueParameterListFM = self.trueParameterList
        self.minimum = np.full((1), -999.0)
        self.maximum = np.full((1), -999.0)
        self.maximum_change = np.full((1), -999.0)
        self.mapToState = np.eye(1)
        self.mapToParameters = np.eye(1)
        # Not sure if the is covariance, or sqrt covariance. Note this
        # does not seem to the be the same as the Sa used in the error
        # analysis. I think muses-py uses the constraintMatrix sort of
        # like a weighting that is independent of apriori covariance.
        self.constraintMatrix = np.diag(np.full((1,),10*10.0))

class ScaledO3Absorber(O3Absorber):
    '''We can put this into O3Absorber as an option, but for now
    just have a new class to handle this.'''
    @cached_property
    def absorber_vmr(self):
        vmrs = []
        # Get the VMR profile. This will remain at the initial guess
        vmr_profile, _ = self.current_state.object_state(["O3",])
        # And get the scaling
        selem = ["O3_SCALED",]
        coeff, mp = self.current_state.object_state(selem)
        vmr_o3 = rf.AbsorberVmrLevelScaled(self._parent.pressure_fm,
                                           vmr_profile, coeff[0], "O3")
        self.current_state.add_fm_state_vector_if_needed(
            self.fm_sv, selem, [vmr_o3,])
        vmrs.append(vmr_o3)
        return vmrs

class ScaledTropomiFmObjectionCreator(TropomiFmObjectCreator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_absorber = ScaledO3Absorber(self)
    
class ScaledTropomiForwardModelHandle(ForwardModelHandle):
    def __init__(self, **creator_kwargs):
        self.creator_kwargs = creator_kwargs
        self.measurement_id = None
        
    def notify_update_target(self, measurement_id : 'MeasurementId'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        self.measurement_id = measurement_id
        
    def forward_model(self, instrument_name : str,
                      current_state : 'CurrentState',
                      obs : 'MusesObservation',
                      fm_sv: rf.StateVector,
                      rf_uip_func,
                      **kwargs):
        if(instrument_name != "TROPOMI"):
            return None
        obj_creator = ScaledTropomiFmObjectionCreator(
            current_state, self.measurement_id, obs,
            rf_uip=rf_uip_func(),
            fm_sv=fm_sv,
            **self.creator_kwargs)
        fm = obj_creator.forward_model
        logger.info(f"Scaled Tropomi Forward model\n{fm}")
        return fm
    

@long_test
@require_muses_py
def test_tropomi_vrm_scaled(osp_dir, gmao_dir, vlidort_cli,
                            clean_up_replacement_function):
    '''Full run, that we can compare the output files. This is not
    really a unit test, but for convenience we have it here. We don't
    actually do anything with the data, other than make it available.

    Data goes in the local directory, rather than an isolated one.'''
    subprocess.run("rm -r tropomi_vmr_scaled", shell=True)
    r = MusesRunDir(tropomi_test_in_dir,
                    osp_dir, gmao_dir, path_prefix="tropomi_vmr_scaled")
    lognum = logger.add("tropomi_vmr_scaled/retrieve.log")
    # Modify the Table.asc to add a EOF element. This is just a short cut,
    # so we don't need to make a new strategy table. Eventually a new table
    # will be needed in the OSP directory, but it is too early for that.
    subprocess.run(f'sed -i -e "s/O3,/O3_SCALED,/" {r.run_dir}/Table.asc', shell=True)
    # For faster turn around time, set number of iterations to 1. We can test
    # everything, even though the final residual will be pretty high
    subprocess.run(f'sed -i -e "s/15/1 /" {r.run_dir}/Table.asc', shell=True)
    
    rs = RetrievalStrategy(f"{r.run_dir}/Table.asc", vlidort_cli=vlidort_cli)
    # Save data so we can work on getting output in isolation
    rscap = RetrievalStrategyCaptureObserver("retrieval_step", "retrieval step")
    rs.add_observer(rscap)
    ihandle = ScaledTropomiForwardModelHandle(use_pca=False, use_lrad=False,
                                              lrad_second_order=False)
    rs.forward_model_handle_set.add_handle(ihandle, priority_order=100)
    rs.state_element_handle_set.add_handle(SingleSpeciesHandle("O3_SCALED", O3ScaledStateElement, pass_state=False, name="O3_SCALED"))
    rs.update_target(f"{r.run_dir}/Table.asc")
    rs.retrieval_ms()
    if True:
        # The L2-O3 product doesn't get generated, since "O3-SCALED" isn't the "O3"
        # looked for in the code. Fixing this looks a bit involved, and we really should
        # just rework the output anyways. So for now just skip this.
        pass
        # Print out output of EOF, just so we have something to see
        #subprocess.run("h5dump -d OMI_EOF_UV1 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
        #subprocess.run("h5dump -d OMI_EOF_UV2 -A 0 omi_eof/20160414_23_394_11_23/Products/Products_L2-O3-0.nc", shell=True)
    logger.remove(lognum)
