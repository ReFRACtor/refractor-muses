import abc
from loguru import logger
import refractor.muses.muses_py as mpy
from pathlib import Path
import copy
from .priority_handle_set import PriorityHandleSet
from pprint import pprint, pformat
from .muses_levmar_solver import MusesLevmarSolver
from .observation_handle import mpy_radiance_from_observation_list
from .retrieval_result import PropagatedQA, RetrievalResult
import numpy as np
import subprocess

# TODO clean up the usage for various internal objects of RetrievalStrategy, we want to rework
# this anyways as we introduce the MusesStrategyExecutor.

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

    def radiance_step(self):
        '''We have a few places that need the old py-retrieve dict version of our
        observation data. This function calculates that - it is just a reformatting
        of our observation data.'''
        return mpy_radiance_from_observation_list(self.cfunc.obs_list, include_bad_sample=True)

    def radiance_full(self, rs):
        '''The full set of radiance, for all instruments and full band.'''
        olist = [rs.observation_handle_set.observation(iname, None, None,None)
                 for iname in rs.instrument_name_all_step]
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
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        self.BTstruct = [{'diff':0.0, 'obs':0.0, 'fit':0.0} for i in range(100)]
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "bt":
            return (False,  None)
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        jacobian_speciesNames = ['H2O']
        jacobian_specieslist = ['H2O']
        jacobianOut = None
        mytiming = None
        logger.info("Running run_forward_model ...")
        self.cfunc = rs.create_cost_function(
            include_bad_sample=True, fix_apriori_size=True,
            jacobian_speciesIn=jacobian_speciesNames)
        radiance_fm = self.cfunc.max_a_posteriori.model
        freq_fm = np.concatenate([fm.spectral_domain_all().data
                                  for fm in self.cfunc.max_a_posteriori.forward_model])
        # Put into structure expected by modify_from_bt
        radiance_res = {"radiance" : radiance_fm,
                        "frequency" : freq_fm }
        (rs._strategy_executor.strategy._stable.strategy_table_dict, rs._state_info.state_info_dict) = mpy.modify_from_bt(
            mpy.ObjectView(rs._strategy_executor.strategy._stable.strategy_table_dict), rs.step_number,
            self.radiance_step(),
            radiance_res,
            {},
            rs._state_info.state_info_dict,
            self.BTstruct,
            writeOutputFlag=False)
        rs._strategy_executor.strategy._stable.strategy_table_dict = rs._strategy_executor.strategy._stable.strategy_table_dict.__dict__
        logger.info(f"Step: {rs.step_number},  Total Steps (after modify_from_bt): {rs.number_retrieval_step}")
        rs.state_info.next_state_dict = copy.deepcopy(rs.state_info.state_info_dict["current"])
        return (True, None)

class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    '''IRK strategy step.

    NOTE - This hasn't been tested. I haven't found a test case that uses this.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "irk":
            return (False,  None)
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        o_xxx = {"AIRS" : None, "TES" : None, "CRIS" : None, "OMI" : None,
                 "TROPOMI" : None, "OCO2" : None}
        cstep = rs.current_strategy_step
        for iname in cstep.instrument_name:
            if iname in o_xxx:
                obs = rs.observation_handle_set.observation(
                    iname, None, cstep.spectral_window_dict[iname],None)
                if hasattr(obs, "muses_py_dict"):
                    o_xxx[iname] = obs.muses_py_dict
        logger.info("Running run_irk ...")
        self.cfunc = rs.create_cost_function()
        (self.results_irk, self.jacobian_out) = mpy.run_irk(
            rs._strategy_executor.strategy._stable.strategy_table_dict,
            rs.state_info.state_info_dict,
            rs._strategy_executor.strategy._stable.microwindows(),
            rs.retrieval_info.retrieval_info_obj,
            rs.retrieval_info.species_names, 
            rs.retrieval_info.species_list_fm, 
            self.radiance_step(),
            airs=o_xxx["AIRS"], tes_struct=o_xxx["TES"], cris=o_xxx["CRIS"],
            omi=o_xxx["OMI"],
            oco2=o_xxx["OCO2"])
        rs.notify_update("IRK step", retrieval_strategy_step=self)
        return (True, None)

class RetrievalStrategyStepRetrieve(RetrievalStrategyStep):
    '''Strategy step that does a retrieval (e.g., the default strategy step).'''
    def __init__(self):
        super().__init__()
        self.notify_update_target(None)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        # Nothing currently needed
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
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

        self.results = RetrievalResult(
            ret_res, rs.current_strategy_step, rs.retrieval_info, rs.state_info,
            self.cfunc.obs_list, self.radiance_full(rs), rs.propagated_qa)
        logger.info('\n---')
        logger.info(f"Step: {rs.step_number}, Step Name: {rs.step_name}")
        logger.info(f"Best iteration {self.results.best_iteration} out of {self.results.num_iterations}")
        logger.info('---\n')
        
        rs.state_info.update_state(rs.retrieval_info, self.results.results_list,
                                   rs.current_strategy_step.do_not_update_list,
                                   rs.retrieval_config,
                                   rs.step_number)
        # I don't think we actually want this in here. 1) we don't currently
        # support OCO2 and 2) we would just use a direct PressureSigma object
        # along with a new state element name if we did. But leave this commented
        # here to document that py-retrieve did this by we aren't
        #if 'OCO2' in rs.current_strategy_step.instrument_name:
        #    rs._strategy_executor.stable.strategy_table_dict['pressureFM'] = rs._state_info.next_state_dict.pressure
        self.extra_after_run_retrieval_step(rs)
        rs.notify_update("run_retrieval_step", retrieval_strategy_step=self)

        # TODO jacobian_sys is only used in error_analysis_wrapper and error_analysis.
        # I think we can leave bad sample out, although I'm not positive. Would be
        # nice not to have special handling to add bad samples if we turn around and
        # weed them out. For right now, these are required, we would need to update
        # the error analysis to work without bad samples
        if(rs.retrieval_info.n_speciesSys > 0):
            cfunc_sys = rs.create_cost_function(
                do_systematic=True, include_bad_sample=True, fix_apriori_size=True)
            logger.info("Running run_forward_model for systematic jacobians ...")
            self.results.update_jacobian_sys(cfunc_sys)
        rs.notify_update("systematic_jacobian", retrieval_strategy_step=self)
        rs.error_analysis.update_retrieval_result(self.results)
        rs.qa_data_handle_set.qa_update_retrieval_result(self.results)
        rs.propagated_qa.update(rs.current_strategy_step.retrieval_elements,
                                self.results.master_quality)
        
        # The solver can't be pickled, because a few pieces of the cost function
        # can't be pickled. We could sort that out if it becomes an issue, but for
        # now just delete the slv before we call notify_update. Note that it isn't
        # actually required that things calling notify_update can be pickled, it is
        # just convenient for testing. If clearing this becomes a problem, we can
        # come back here to figure out what to do - probably just have a explicit
        # pickle function for this class.
        self.slv = None
        self.cfunc = None
        rs.notify_update("retrieval step", retrieval_strategy_step=self)
        
        return (True, None)

    def run_retrieval(self, rs):
        '''run_retrieval'''
        self.cfunc = rs.create_cost_function()
        rs.notify_update("create_cost_function", retrieval_strategy_step=self)
        maxIter = rs.current_strategy_step.max_num_iterations
        
        # Various thresholds from the input table
        ConvTolerance_CostThresh = float(rs.retrieval_config["ConvTolerance_CostThresh"])
        ConvTolerance_pThresh = float(rs.retrieval_config["ConvTolerance_pThresh"])
        ConvTolerance_JacThresh = float(rs.retrieval_config["ConvTolerance_JacThresh"])
        r = self.radiance_step()["NESR"]
        Chi2Tolerance = 2.0 / len(r) # theoretical value for tolerance
        if rs.retrieval_type == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = rs.retrieval_config['LMDelta'] # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc
        if rs.write_output:
            levmar_log_file = f"{rs.run_dir}/Step{rs.step_number:02d}_{rs.step_name}/LevmarSolver-{rs.step_name}.log"
        else:
            levmar_log_file = None
        logger.info(f"Initial State vector:\n{self.cfunc.fm_sv}")
        self.slv = MusesLevmarSolver(self.cfunc,
                                     maxIter,
                                     delta_value,
                                     ConvTolerance,   
                                     Chi2Tolerance,
                                     verbose=True,
                                     log_file=levmar_log_file)
        if(maxIter > 0):
            self.slv.solve()
        logger.info(f"Solved State vector:\n{self.cfunc.fm_sv}")
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
        rs.state_info.state_info_dict["constraint"]['omi']['cloud_fraction'] = \
            rs.state_info.state_info_dict["current"]['omi']['cloud_fraction']
        

class RetrievalStrategyStep_tropomicloud_ig_refine(RetrievalStrategyStepRetrieve):
    '''This is a retreival, followed by using the results to update the
    TROPOMI cloud fraction.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "tropomicloud_ig_refine":
            return (False,  None)
        return super().retrieval_step(retrieval_type, rs)

    def extra_after_run_retrieval_step(self, rs):
        rs.state_info.state_info_dict["constraint"]['tropomi']['cloud_fraction'] = \
            rs.state_info.state_info_dict["current"]['tropomi']['cloud_fraction']
    
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
           
           
