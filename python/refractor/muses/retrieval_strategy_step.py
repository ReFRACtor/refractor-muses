import abc
import logging
import refractor.muses.muses_py as mpy
from pathlib import Path
import copy
from .priority_handle_set import PriorityHandleSet
from pprint import pprint, pformat
from .refractor_uip import RefractorUip
from .cost_function import CostFunction
from .muses_levmar_solver import MusesLevmarSolver
from .current_state import CurrentState, CurrentStateUip, CurrentStateStateInfo
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_result import PropagatedQA, RetrievalResult
from .muses_spectral_window import MusesSpectralWindow
from functools import partial
import numpy as np

logger = logging.getLogger("py-retrieve")

# TODO clean up the usage for various internal objects of RetrievalStrategy, we want to rework
# this anyways as we introduce the MusesStrategyExecutor.

def struct_compare(s1, s2):
    for k in s1.keys():
        print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

class RetrievalStrategyStepSet(PriorityHandleSet):
    '''This takes the retrieval_type and determines a RetrievalStrategyStep
    to handle this. It then does the retrieval step.

    Note RetrievalStrategyStep can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def retrieval_step(self, retrieval_type : str, rs : 'RetrievalStrategy') -> None:
        self.handle(retrieval_type.lower(), rs)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(rs)
        
    def handle_h(self, h : 'RetrievalStrategyStep', retrieval_type : str,
                 rs : 'RetrievalStrategy') -> (bool, None):
        return h.retrieval_step(retrieval_type, rs)

class RetrievalStrategyStep(object, metaclass=abc.ABCMeta):
    '''Do the retrieval step indicated by retrieval_type
    
    Note RetrievalStrategyStep can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''

    def __init__(self):
        self._uip = None
        
    def notify_update_target(self, rs : 'RetrievalStrategy'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        # Default is to do nothing
        pass
    
    @abc.abstractmethod
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        '''Returns (True, None) if we handle the retrieval step, (False, None)
        otherwise'''
        raise NotImplementedError

    def uip_func(self, rs, do_systematic, jacobian_speciesIn):
        '''To support the old muses-py ForwardModel, we pass a uip_func to our
        CostFunctionCreator that gets used if needed. This should only be used
        by the old muses-py code, we shouldn't use this for ReFRACtor ForwardModel.
        '''
        if(self._uip is None):
            if(do_systematic):
                retrieval_info = rs.retrieval_info.retrieval_info_obj
                rinfo = mpy.ObjectView({
                    'parameterStartFM': retrieval_info.parameterStartSys,
                    'parameterEndFM' : retrieval_info.parameterEndSys,
                    'species': retrieval_info.speciesSys,
                    'n_species': retrieval_info.n_speciesSys,
                    'speciesList': retrieval_info.speciesListSys,
                    'speciesListFM': retrieval_info.speciesListSys,
                    'mapTypeListFM': mpy.constraint_get_maptype(rs._error_analysis.error_current, retrieval_info.speciesListSys),
                    'initialGuessListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                    'constraintVectorListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                    'initialGuessList': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                    'n_totalParametersFM': retrieval_info.n_totalParametersSys
                })
            else:
                rinfo = rs.retrieval_info
            o_xxx = {"AIRS" : None, "TES" : None, "CRIS" : None, "OMI" : None,
                     "TROPOMI" : None, "OCO2" : None}
            o_airs = None
            o_tes = None
            o_cris = None
            o_omi = None
            o_tropomi = None
            o_oco2 = None
            for iname in rs._strategy_table.instrument_name():
                if iname in o_xxx:
                    obs = rs.observation_handle_set.observation(iname, None,
                       MusesSpectralWindow(rs._strategy_table.spectral_window(iname), None),
                       None)
                    if hasattr(obs, "muses_py_dict"):
                        o_xxx[iname] = obs.muses_py_dict
            self._uip = RefractorUip.create_uip(rs._state_info, rs._strategy_table,
                                                rs._strategy_table.microwindows(), rinfo,
                                                o_xxx["AIRS"], o_xxx["TES"], o_xxx["CRIS"],
                                                o_xxx["OMI"], o_xxx["TROPOMI"], o_xxx["OCO2"],
                                                jacobian_speciesIn=jacobian_speciesIn)
        return self._uip

    def create_cost_function(self, rs : 'RetrievalStrategy',
                             do_systematic=False,
                             include_bad_sample=False,
                             fix_apriori_size=False,
                             jacobian_speciesIn=None):
        '''Create a CostFunction, for use either in retrieval or just for running
        the forward model (the CostFunction is a little overkill for just a
        forward model run, but it has all the pieces needed so no reason not to
        just generate everything).

        If do_systematic is True, then we use the systematic species list. '''
        self._uip = None
        self.cstate = CurrentStateStateInfo(rs._state_info, rs.retrieval_info,
                                            do_systematic=do_systematic,
                                            retrieval_state_element_override=jacobian_speciesIn)
        # Temp, until we get this sorted out
        self.cstate.apriori_cov = rs.retrieval_info.apriori_cov
        self.cstate.sqrt_constraint = (mpy.sqrt_matrix(self.cstate.apriori_cov)).transpose()
        self.cstate.apriori = rs.retrieval_info.apriori
        # TODO Would probably be good to remove include_bad_sample, it isn't clear that
        # we ever want to run the forward model for bad samples. But right now the existing
        # py-retrieve code requires this is a few places.a
        return rs._cost_function_creator.cost_function(
            rs._strategy_table.instrument_name(),
            self.cstate,
            rs._swin_dict,
            partial(self.uip_func, rs, do_systematic, jacobian_speciesIn),
            include_bad_sample=include_bad_sample,
            fix_apriori_size=fix_apriori_size, **rs._kwargs)

    def radiance_step(self):
        '''We have a few places that need the old py-retrieve dict version of our
        observation data. This function calculates that - it is just a reformatting
        of our observation data.'''
        return mpy_radiance_from_observation_list(self.cfunc.obs_list, include_bad_sample=True)

    def radiance_full(self, rs):
        '''The full set of radiance, for all instruments and full band.'''
        olist = [rs.observation_handle_set.observation(iname, None, None,None)
                 for iname in rs._strategy_table.instrument_name(all_step=True)]
        return mpy_radiance_from_observation_list(olist, full_band=True)
        
class RetrievalStrategyStepNotImplemented(RetrievalStrategyStep):
    '''There seems to be a few retrieval types that aren't implemented in
    py-retrieve. It might also be that we just don't have this implemented
    right in ReFRACtor, but in any case we don't have a test case for this.

    Throw and exception to indicate this, we can look at implementing this in the
    future - particularly if we have a test case to validate the code.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type not in ("forwardmodel", "omi_radiance_calibration"):
            return (False,  None)
        raise RuntimeError(f"We don't currently support retrieval_type {retrieval_type}")

class RetrievalStrategyStepBT(RetrievalStrategyStep):
    '''Brightness Temperature strategy step.'''
    def __init__(self):
        super().__init__()
        self.notify_update_target(None)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        self.BTstruct = [{'diff':0.0, 'obs':0.0, 'fit':0.0} for i in range(100)]
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "bt":
            return (False,  None)
        jacobian_speciesNames = ['H2O']
        jacobian_specieslist = ['H2O']
        jacobianOut = None
        mytiming = None
        logger.info("Running run_forward_model ...")
        self.cfunc = self.create_cost_function(rs, include_bad_sample=True,
                                               fix_apriori_size=True,
                                               jacobian_speciesIn=jacobian_speciesNames)
        radiance_fm = self.cfunc.max_a_posteriori.model
        freq_fm = np.concatenate([fm.spectral_domain_all().data
                                  for fm in self.cfunc.max_a_posteriori.forward_model])
        # Put into structure expected by modify_from_bt
        radiance_res = {"radiance" : radiance_fm,
                        "frequency" : freq_fm }
        (rs._strategy_table.strategy_table_dict, rs._state_info.state_info_dict) = mpy.modify_from_bt(
            mpy.ObjectView(rs._strategy_table.strategy_table_dict), rs.step_number,
            self.radiance_step(),
            radiance_res,
            {},
            rs._state_info.state_info_dict,
            self.BTstruct,
            writeOutputFlag=False)
        rs._strategy_table.strategy_table_dict = rs._strategy_table.strategy_table_dict.__dict__
        logger.info(f"Step: {rs.step_number},  Total Steps (after modify_from_bt): {rs.number_retrieval_step}")
        rs._state_info.next_state_dict = copy.deepcopy(rs._state_info.state_info_dict["current"])
        return (True, None)

class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    '''IRK strategy step.

    NOTE - This hasn't been tested. I haven't found a test case that uses this.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "irk":
            return (False,  None)
        jacobian_speciesNames = rs.retrieval_info.species[0:rs.retrieval_info.n_species]
        jacobian_specieslist = rs.retrieval_info.speciesListFM[0:rs.retrieval_info.n_totalParametersFM]
        jacobianOut = None
        mytiming = None
        uip=tes = cris = omi = tropomi = None 
        logger.info("Running run_irk ...")
        self.cfunc = self.create_cost_function(rs)
        (resultsIRK, jacobianOut) = mpy.run_irk(
            rs._strategy_table.strategy_table_dict,
            rs._state_info, rs._strategy_table.microwindows(), rs.retrieval_info,
            jacobian_speciesNames, 
            jacobian_specieslist, 
            self.radiance_step(),
            uip, 
            rs.o_airs, tes, cris, omi, tropomi,
            mytiming, 
            writeOutput=None)
        return (True, None)

class RetrievalStrategyStepRetrieve(RetrievalStrategyStep):
    '''Strategy step that does a retrieval (e.g., the default strategy step).'''
    def __init__(self):
        super().__init__()
        self.notify_update_target(None)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        self.propagated_qa = PropagatedQA()
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        rs.notify_update("retrieval input")
        rs.retrieval_info.stepNumber = rs.step_number
        rs.retrieval_info.stepName = rs.step_name
        logger.info("Running run_retrieval ...")
        
        # SSK 2023.  I find I get failures from glitches like reading
        # L1B files, if there are many processes running.  I re-run
        # several times to give every obs a chance to complete chance
        # to run, because I am running the same set with different
        # strategies and want to compare.  write out a token if it
        # gets here, indicating this obs already got a chance.  I then
        # copy all completed runs to either 00good/ or 00bad/
        # depending on the success flag, and re-run anything remaining
        # in the main directory.
        
        Path(f"{rs.run_dir}/-run_token.asc").touch()

        ret_res = self.run_retrieval(rs)

        self.results = RetrievalResult(ret_res, rs._strategy_table, rs.retrieval_info,
                                       rs._state_info, self.cfunc.obs_list,
                                       self.radiance_full(rs),
                                       self.propagated_qa)
        logger.info('\n---')
        logger.info(f"Step: {rs.step_number}, Step Name: {rs.step_name}")
        logger.info(f"Best iteration {self.results.best_iteration} out of {self.results.num_iterations}")
        logger.info('---\n')
        
        do_not_update = rs._strategy_table.table_entry("donotupdate").lower()
        if do_not_update != '-':
            do_not_update = [x.upper() for x in do_not_update.split(',')]
        else:
            do_not_update = []

        rs._state_info.update_state(rs.retrieval_info, self.results.results_list,
                                    do_not_update, rs.retrieval_config,
                                    rs.step_number)
        if 'OCO2' in rs._strategy_table.instrument_name():
            # set table.pressurefm to stateConstraint.pressure because OCO-2
            # is on sigma levels
            rs._strategy_table.strategy_table_dict['pressureFM'] = rs._state_info.next_state_dict.pressure
        self.extra_after_run_retrieval_step(rs)
        rs.notify_update("run_retrieval_step")

        # TODO jacobian_sys is only used in error_analysis_wrapper and error_analysis.
        # I think we can leave bad sample out, although I'm not positive. Would be
        # nice not to have special handling to add bad samples if we turn around and
        # weed them out. For right now, these are required, we would need to update
        # the error analysis to work without bad samples
        if(rs.retrieval_info.n_speciesSys > 0):
            cfunc_sys = self.create_cost_function(rs, do_systematic=True,
                                     include_bad_sample=True, fix_apriori_size=True)
            logger.info("Running run_forward_model for systematic jacobians ...")
            self.results.update_jacobian_sys(cfunc_sys)

        self.results.update_error_analysis(rs._error_analysis)
        self.propagated_qa.update(rs._strategy_table.retrieval_elements(),
                                  self.results.master_quality)
        
        # The solver can't be pickled, because a few pieces of the cost function
        # can't be pickled. We could sort that out if it becomes an issue, but for
        # now just delete the slv before we call notify_update. Note that it isn't
        # actually required that things calling notify_update can be pickled, it is
        # just convenient for testing. If clearing this becomes a problem, we can
        # come back here to figure out what to do - probably just have a explicit
        # pickle function for this class.
        self.slv = None
        rs.notify_update("retrieval step", retrieval_strategy_step=self)
        
        return (True, None)

    def run_retrieval(self, rs):
        '''run_retrieval'''
        self.cfunc = self.create_cost_function(rs)
        maxIter = int(rs._strategy_table.table_entry("maxNumIterations"))
        
        # Various thresholds from the input table
        ConvTolerance_CostThresh = float(rs._strategy_table.preferences["ConvTolerance_CostThresh"])
        ConvTolerance_pThresh = float(rs._strategy_table.preferences["ConvTolerance_pThresh"])
        ConvTolerance_JacThresh = float(rs._strategy_table.preferences["ConvTolerance_JacThresh"])
        r = self.radiance_step()["NESR"]
        Chi2Tolerance = 2.0 / len(r) # theoretical value for tolerance
        if rs.retrieval_type == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = rs._strategy_table.preferences['LMDelta'] # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc

        self.slv = MusesLevmarSolver(self.cfunc,
                                     maxIter,
                                     delta_value,
                                     ConvTolerance,   
                                     Chi2Tolerance)
        if(maxIter > 0):
            self.slv.solve()
        return self.slv.retrieval_results()
    

    def extra_after_run_retrieval_step(self, rs):
        '''We have a couple of steps that just do some extra adjustments before
        we go into the systematic_jacobian/error_analysis stuff. This is just a hook
        for putting this in place.'''
        pass

class RetrievalStrategyStep_omicloud_ig_refine(RetrievalStrategyStepRetrieve):
    '''This is a retreival, followed by using the results to update the
    OMI cloud fraction.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "omicloud_ig_refine":
            return (False,  None)
        return super().retrieval_step(retrieval_type, rs)

    def extra_after_run_retrieval_step(self, rs):
        rs._state_info.state_info_dict["constraint"]['omi']['cloud_fraction'] = \
            rs._state_info.state_info_dict["current"]['omi']['cloud_fraction']
        

class RetrievalStrategyStep_tropomicloud_ig_refine(RetrievalStrategyStepRetrieve):
    '''This is a retreival, followed by using the results to update the
    TROPOMI cloud fraction.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "tropomicloud_ig_refine":
            return (False,  None)
        return super().retrieval_step(retrieval_type, rs)

    def extra_after_run_retrieval_step(self, rs):
        rs._state_info.state_info_dict["constraint"]['tropomi']['cloud_fraction'] = \
            rs._state_info.state_info_dict["current"]['tropomi']['cloud_fraction']
    
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepNotImplemented())
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepBT())
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepIRK())
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStep_omicloud_ig_refine())
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStep_tropomicloud_ig_refine())
# Anything that isn't one of the special types is a generic retrieval, so
# fall back to this as the lowest priority fall back
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepRetrieve(),
                                            priority_order = -1)

__all__ = ["RetrievalStrategyStepSet", "RetrievalStrategyStep",
           "RetrievalStrategyStepNotImplemented",
           "RetrievalStrategyStepBT",
           "RetrievalStrategyStepIRK",
           "RetrievalStrategyStepRetrieve",
           "RetrievalStrategyStep_omicloud_ig_refine",
           "RetrievalStrategyStep_tropomicloud_ig_refine",
           ]
           
           
