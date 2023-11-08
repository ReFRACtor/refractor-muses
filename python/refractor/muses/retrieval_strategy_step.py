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
import numpy as np

logger = logging.getLogger("py-retrieve")

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
    to handle this. It then does the retrieval step'''
    def retrieval_step(self, retrieval_type : str, rs : 'RetrievalStrategy') -> None:
        self.handle(retrieval_type, rs)
        
    def handle_h(self, h : 'RetrievalStrategyStep', retrieval_type : str,
                 rs : 'RetrievalStrategy') -> (bool, None):
        return h.retrieval_step(retrieval_type, rs)

class RetrievalStrategyStep(object, metaclass=abc.ABCMeta):
    '''Do the retrieval step indicated by retrieval_type'''
    @abc.abstractmethod
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        '''Returns (True, None) if we handle the retrieval step, (False, None)
        otherwise'''
        raise NotImplementedError

    def create_cost_function(self, rs : 'RetrievalStrategy',
                             do_systematic=False,
                             include_bad_sample=False,
                             fix_apriori_size=False,
                             jacobian_speciesIn=None):
        '''Create a CostFunction, for use either in retrieval or just for running
        the forward model (the CostFunction is a little overkill for just a
        forward model run, but it has all the pieces needed so no reason not to
        just generate everything.

        If do_systematic is True, then we use the systematic species list. Not
        exactly sure how this is different then just subsetting the full retrieval,
        but at least for now duplicate what muses-py does.'''

        args, rf_uip, radianceStepIn = rs.fm_obs_creator.fm_and_obs_rs(
                    do_systematic=do_systematic, include_bad_sample=include_bad_sample,
                    fix_apriori_size=fix_apriori_size,
                    jacobian_speciesIn=jacobian_speciesIn, **rs.kwargs)
        cfunc = CostFunction(*args)
        cfunc.parameters = rf_uip.current_state_x
        return (rf_uip, cfunc, radianceStepIn)
        

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
    '''Branch Type strategy step.'''
    def __init__(self):
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
        _,cfunc,radianceStepIn = self.create_cost_function(rs, include_bad_sample=True,
                                          fix_apriori_size=True,
                                          jacobian_speciesIn=jacobian_speciesNames)
        radiance_fm = cfunc.max_a_posteriori.model
        freq_fm = np.concatenate([fm.spectral_domain_all().data
                                  for fm in cfunc.max_a_posteriori.forward_model])
        # Put into structure expected by modify_from_bt
        radiance_res = {"radiance" : radiance_fm,
                        "frequency" : freq_fm }
        (rs.strategy_table.strategy_table_dict, rs.state_info.state_info_dict) = mpy.modify_from_bt(
            mpy.ObjectView(rs.strategy_table.strategy_table_dict), rs.table_step,
            radianceStepIn,
            radiance_res, rs.windows, rs.state_info.state_info_dict,
            self.BTstruct,
            writeOutputFlag=False)
        rs.strategy_table.strategy_table_dict = rs.strategy_table.strategy_table_dict.__dict__
        logger.info(f"Step: {rs.table_step},  Total Steps (after modify_from_bt): {rs.number_table_step}")
        rs.state_info.next_state_dict = copy.deepcopy(rs.state_info.state_info_dict["current"])
        return (True, None)

class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    '''IRK strategy step.

    NOTE - This hasn't been tested. I haven't found a test case that uses this.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "irk":
            return (False,  None)
        jacobian_speciesNames = rs.retrievalInfo.species[0:rs.retrievalInfo.n_species]
        jacobian_specieslist = rs.retrievalInfo.speciesListFM[0:rs.retrievalInfo.n_totalParametersFM]
        jacobianOut = None
        mytiming = None
        uip=tes = cris = omi = tropomi = None 
        logger.info("Running run_irk ...")
        (resultsIRK, jacobianOut) = mpy.run_irk(
            rs.strategy_table.strategy_table_dict,
            rs.state_info, rs.windows, rs.retrievalInfo,
            jacobian_speciesNames, 
            jacobian_specieslist, 
            rs.radianceStepIn,
            uip, 
            rs.o_airs, tes, cris, omi, tropomi,
            mytiming, 
            writeOutput=None)
        return (True, None)

class RetrievalStrategyStepRetrieve(RetrievalStrategyStep):
    '''Strategy step that does a retrieval (e.g., the default strategy step).'''
    def __init__(self):
        self.propagatedTATMQA = 1
        self.propagatedH2OQA = 1
        self.propagatedO3QA = 1
        
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        rs.notify_update("retrieval input")
        rs.retrievalInfo.stepNumber = rs.table_step
        rs.retrievalInfo.stepName = rs.step_name
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

        retrievalResults = self.run_retrieval(rs)

        self.results = mpy.set_retrieval_results(
            rs.strategy_table.strategy_table_dict, rs.windows, retrievalResults,
            rs.retrievalInfo, rs.radianceStep, rs.state_info.state_info_obj,
            {"currentGuessListFM" : retrievalResults["xretFM"]})
        logger.info('\n---')
        logger.info(f"Step: {rs.table_step}, Step Name: {rs.step_name}")
        logger.info(f"Best iteration {self.results.bestIteration} out of {retrievalResults['num_iterations']}")
        logger.info('---\n')
        self.results = mpy.set_retrieval_results_derived(self.results,
                      rs.radianceStep, self.propagatedTATMQA,
                      self.propagatedO3QA, self.propagatedH2OQA)
        
        do_not_update = rs.strategy_table.table_entry("donotupdate").lower()
        if do_not_update != '-':
            do_not_update = [x.upper() for x in do_not_update.split(',')]
        else:
            do_not_update = []

        rs.state_info.update_state(rs.retrievalInfo, self.results.resultsList,
                                   do_not_update, rs.cloud_prefs, rs.table_step)
        if 'OCO2' in rs.instruments:
            # set table.pressurefm to stateConstraint.pressure because OCO-2
            # is on sigma levels
            rs.strategy_table.strategy_table_dict['pressureFM'] = rs.state_info.next_state_dict.pressure
        self.extra_after_run_retrieval_step(rs)
        rs.notify_update("run_retrieval_step")

        # Systematic jacobians. Not sure how this is different then just a
        # subset of the full jacobian, but for now duplicate what muses-py does.
        # TODO jacobianSys is only used in error_analysis_wrapper and error_analysis.
        # I think we can leave bad sample out, although I'm not positive. Would be
        # nice not to have special handling to add bad samples if we turn around and
        # weed them out.
        if(rs.retrievalInfo.n_speciesSys > 0):
            _,cfunc_sys,_ = self.create_cost_function(rs, do_systematic=True,
                                     include_bad_sample=True, fix_apriori_size=True)
            logger.info("Running run_forward_model for systematic jacobians ...")
            self.results.jacobianSys = cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[np.newaxis,:,:]

        mpy.set_retrieval_results_derived(self.results, rs.radianceStep,
                                          self.propagatedTATMQA, self.propagatedO3QA,
                                          self.propagatedH2OQA)
        # Temp, skip
        #self.results = rs.error_analysis.error_analysis(rs, self.results)
        self.update_retrieval_summary(rs)
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

    def update_retrieval_summary(self, rs):
        '''Calculate various summary statistics for retrieval'''
        self.results = mpy.write_retrieval_summary(
            rs.strategy_table.analysis_directory,
            rs.retrievalInfo.retrieval_info_obj,
            rs.state_info.state_info_obj,
            None,
            self.results,
            rs.windows,
            rs.press_list,
            rs.quality_name, 
            rs.table_step, 
            rs.error_analysis.error_current, 
            writeOutputFlag=False, 
            errorInitial=rs.error_analysis.error_initial
        )
        if 'TATM' in rs.retrievalInfo.species_names:
            self.propagatedTATMQA = self.results.masterQuality
        if 'O3' in rs.retrievalInfo.species_names:
            self.propagatedO3QA = self.results.masterQuality
        if 'H2O' in rs.retrievalInfo.species_names:
            self.propagatedH2OQA = self.results.masterQuality
        

    def run_retrieval(self, rs):
        '''run_retrieval'''
        rf_uip, cfunc,radianceStepIn = self.create_cost_function(rs)
        maxIter = int(rs.strategy_table.table_entry("maxNumIterations"))
        
        # Various thresholds from the input table
        ConvTolerance_CostThresh = np.float(rs.strategy_table.preferences["ConvTolerance_CostThresh"])
        ConvTolerance_pThresh = np.float(rs.strategy_table.preferences["ConvTolerance_pThresh"])
        ConvTolerance_JacThresh = np.float(rs.strategy_table.preferences["ConvTolerance_JacThresh"])
        Chi2Tolerance = 2.0 / len(radianceStepIn["NESR"]) # theoretical value for tolerance
        if rs.retrieval_type == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = rs.strategy_table.preferences['LMDelta'] # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc

        self.slv = MusesLevmarSolver(cfunc,
                                     rf_uip,
                                     maxIter,
                                     delta_value,
                                     ConvTolerance,   
                                     Chi2Tolerance)
        if(maxIter > 0):
            self.slv.solve()
        # TODO Hopefully this can go away
        rs.radianceStep = cfunc.radianceStep(radianceStepIn)
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
           
           
