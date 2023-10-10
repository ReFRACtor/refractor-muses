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
                             do_systematic=False):
        '''Create a CostFunction, for use either in retrieval or just for running
        the forward model (the CostFunction is a little overkill for just a
        forward model run, but it has all the pieces needed so no reason not to
        just generate everything.

        If do_systematic is True, then we use the systematic species list. Not
        exactly sure how this is different then just subsetting the full retrieval,
        but at least for now duplicate what muses-py does.'''

        # TODO We would like to get away from using the UIP and ret_info at this
        # level of processing. But for now generate and return these, we'll
        # try to get this cleaned up later
        if(do_systematic):
            retrieval_info = rs.retrievalInfo.retrieval_info_obj
            rinfo = mpy.ObjectView({
                'parameterStartFM': retrieval_info.parameterStartSys,
                'parameterEndFM' : retrieval_info.parameterEndSys,
                'species': retrieval_info.speciesSys,
                'n_species': retrieval_info.n_speciesSys,
                'speciesList': retrieval_info.speciesListSys,
                'speciesListFM': retrieval_info.speciesListSys,
                'mapTypeListFM': mpy.constraint_get_maptype(rs.errorCurrent, retrieval_info.speciesListSys),
                'initialGuessListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'constraintVectorListFM': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'initialGuessList': np.zeros(shape=(retrieval_info.n_totalParametersSys,), dtype=np.float32),
                'n_totalParametersFM': retrieval_info.n_totalParametersSys
            })
        else:
            rinfo = rs.retrievalInfo
        rf_uip = RefractorUip.create_uip(rs.stateInfo, rs.strategy_table,
                                         rs.windows, rinfo,
                                         rs.o_airs, rs.o_tes,
                                         rs.o_cris, rs.o_omi, rs.o_tropomi,
                                         rs.o_oco2)
        if(do_systematic == True):
            cfunc = CostFunction(*rs.fm_obs_creator.fm_and_fake_obs(rf_uip,
                             **rs.kwargs, use_full_state_vector=True,
                             include_bad_sample=True))
            cfunc.parameters = rf_uip.current_state_x_fm
        else:
            ret_info = { 
                'obs_rad': rs.radianceStepIn["radiance"],
                'meas_err': rs.radianceStepIn["NESR"],            
                'sqrt_constraint': (mpy.sqrt_matrix(rs.retrievalInfo.apriori_cov)).transpose(),
                'const_vec': rs.retrievalInfo.apriori,
            }
            cfunc = CostFunction(*rs.fm_obs_creator.fm_and_obs(rf_uip,
                                ret_info, **rs.kwargs))
            cfunc.parameters = rf_uip.current_state_x
        return (rf_uip, cfunc)
        

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
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "bt":
            return (False,  None)
        jacobian_speciesNames = ['H2O']
        jacobian_specieslist = ['H2O']
        jacobianOut = None
        mytiming = None
        logger.info("Running run_forward_model ...")
        # TODO Add back in writeOutput stuff to run_forward_model.
        # Also, should be able to use forward model w/o jacobian
        rs.radianceResults, _ = rs.run_forward_model(
                rs.strategy_table, rs.stateInfo, rs.windows, rs.retrievalInfo,
                jacobian_speciesNames,
                jacobian_specieslist,
                rs.radianceStepIn,
                rs.o_airs, rs.o_cris, rs.o_tes, rs.o_omi, rs.o_tropomi,
                rs.oco2_step, None)
        rs.stateOneNext = copy.deepcopy(rs.stateInfo.state_info_dict["current"])
        (rs.strategy_table, rs.stateInfo.state_info_dict) = mpy.modify_from_bt(
            mpy.ObjectView(rs.strategy_table), rs.table_step,
            rs.radianceStepIn,
            rs.radianceResults, rs.windows, rs.stateInfo.state_info_dict,
            rs.BTstruct,
            writeOutputFlag=False)
        rs.strategy_table = rs.strategy_table.__dict__
        logger.info(f"Step: {rs.table_step},  Total Steps (after modify_from_bt): {rs.number_table_step}")
        rs.stateOneNext = mpy.ObjectView(copy.deepcopy(rs.stateInfo.state_info_dict["current"]))
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
            rs.strategy_table, rs.stateInfo, rs.windows, rs.retrievalInfo,
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

        self.results = mpy.set_retrieval_results(rs.strategy_table, rs.windows, retrievalResults, rs.retrievalInfo, rs.radianceStep, rs.stateInfo.state_info_obj,
                             {"currentGuessListFM" : retrievalResults["xretFM"]})
        logger.info('\n---')
        logger.info(f"Step: {rs.table_step}, Step Name: {rs.step_name}")
        logger.info(f"Best iteration {self.results.bestIteration} out of {retrievalResults['num_iterations']}")
        logger.info('---\n')
        self.results = mpy.set_retrieval_results_derived(self.results,
                      rs.radianceStep, self.propagatedTATMQA,
                      self.propagatedO3QA, self.propagatedH2OQA)
        
        rs.stateOneNext = copy.deepcopy(rs.stateInfo.state_info_dict["current"])
        donotupdate = mpy.table_get_entry(rs.strategy_table, rs.table_step,
                                          "donotupdate").lower()
        if donotupdate != '-':
            donotupdate = [x.upper() for x in donotupdate.split(',')]
        else:
            donotupdate = []

        # stateOneNext is not updated when it says "do not update"
        # state.current is updated for all results
        (rs.stateInfo.state_info_dict, _, rs.stateOneNext) = \
            mpy.update_state(rs.stateInfo.state_info_dict,
                             rs.retrievalInfo.retrieval_info_obj,
                             self.results.resultsList, rs.cloud_prefs,
                             rs.table_step, donotupdate, rs.stateOneNext)
        rs.stateInfo.state_info_dict = rs.stateInfo.state_info_dict.__dict__
        if 'OCO2' in rs.instruments:
            # set table.pressurefm to stateConstraint.pressure because OCO-2
            # is on sigma levels
            rs.strategy_table['pressureFM'] = rs.stateOneNext.pressure
        self.extra_after_run_retrieval_step(rs)
        rs.notify_update("run_retrieval_step")

        # Systematic jacobians. Not sure how this is different then just a
        # subset of the full jacobian, but for now duplicate what muses-py does.
        # TODO jacobianSys is only used in error_analysis_wrapper and error_analysis.
        # I think we can leave bad sample out, although I'm not positive. Would be
        # nice not to have special handling to add bad samples if we turn around and
        # weed them out.
        if(rs.retrievalInfo.n_speciesSys > 0):
            _,cfunc_sys = self.create_cost_function(rs, do_systematic=True)
            logger.info("Running run_forward_model for systematic jacobians ...")
            self.results.jacobianSys = cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[np.newaxis,:,:]

        # TODO Move this to the one spot it is used, no reason to have here
        # This is an odd interface, but it is currently what is required by
        # write_products_one_radiance. We should perhaps change that function, but
        # currently this is how if get the updates for omi or tropomi
        for inst in ("OMI", "TROPOMI"):
            if(inst in rs.radianceStep["instrumentNames"]):
                i = rs.radianceStep["instrumentNames"].index(inst)
                istart = sum(rs.radianceStep["instrumentSizes"][:i])
                iend = istart + rs.radianceStep["instrumentSizes"][i]
                r = range(istart, iend)
                rs.myobsrad = {"instrumentNames" : [inst],
                               "frequency" : rs.radianceStep["frequency"][r],
                               "radiance" : rs.radianceStep["radiance"][r],
                               "NESR" : rs.radianceStep["NESR"][r]}
        mpy.set_retrieval_results_derived(self.results, rs.radianceStep,
                                          self.propagatedTATMQA, self.propagatedO3QA,
                                          self.propagatedH2OQA)
        self.error_analysis(rs)
        self.update_retrieval_summary(rs)
        rs.notify_update("retrieval step", retrieval_strategy_step=self)
        
        return (True, None)


    def error_analysis(self, rs):
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radianceNoise = {"radiance" : np.zeros_like(rs.radianceStep["radiance"]) }
        (self.results, rs.errorCurrent) = mpy.error_analysis_wrapper(
            rs.table_step,
            rs.strategy_table["dirAnalysis"],
            rs.radianceStep,
            radianceNoise,
            rs.retrievalInfo.retrieval_info_obj,
            rs.stateInfo.state_info_obj,
            rs.errorInitial,
            rs.errorCurrent,
            rs.windows,
            self.results
            )

    def update_retrieval_summary(self, rs):
        '''Calculate various summary statistics for retrieval'''
        self.results = mpy.write_retrieval_summary(
            rs.strategy_table["dirAnalysis"],
            rs.retrievalInfo.retrieval_info_obj,
            rs.stateInfo.state_info_obj,
            None,
            self.results,
            rs.windows,
            rs.press_list,
            rs.quality_name, 
            rs.table_step, 
            rs.errorCurrent, 
            writeOutputFlag=False, 
            errorInitial=rs.errorInitial
        )
        if 'TATM' in rs.retrievalInfo.species_names:
            self.propagatedTATMQA = self.results.masterQuality
        if 'O3' in rs.retrievalInfo.species_names:
            self.propagatedO3QA = self.results.masterQuality
        if 'H2O' in rs.retrievalInfo.species_names:
            self.propagatedH2OQA = self.results.masterQuality
        

    def run_retrieval(self, rs):
        '''run_retrieval'''
        rf_uip, cfunc = self.create_cost_function(rs)
        maxIter = int(mpy.table_get_entry(rs.strategy_table, rs.strategy_table["step"], "maxNumIterations"))
        
        # Various thresholds from the input table
        ConvTolerance_CostThresh = np.float32(mpy.table_get_pref(rs.strategy_table, "ConvTolerance_CostThresh"))
        ConvTolerance_pThresh = np.float32(mpy.table_get_pref(rs.strategy_table, "ConvTolerance_pThresh"))     
        ConvTolerance_JacThresh = np.float32(mpy.table_get_pref(rs.strategy_table, "ConvTolerance_JacThresh"))
        Chi2Tolerance = 2.0 / len(rs.radianceStepIn["NESR"]) # theoretical value for tolerance
        if rs.retrieval_type == "bt_ig_refine":
            ConvTolerance_CostThresh = 0.00001
            ConvTolerance_pThresh = 0.00001
            ConvTolerance_JacThresh = 0.00001
            Chi2Tolerance = 0.00001
        ConvTolerance = [ConvTolerance_CostThresh, ConvTolerance_pThresh, ConvTolerance_JacThresh]
        delta_str = mpy.table_get_pref(rs.strategy_table, 'LMDelta') # 100 // original LM step size
        delta_value = int(delta_str.split()[0])  # We only need the first token sinc

        self.slv = MusesLevmarSolver(cfunc,
                                     rs.table_step,
                                     rf_uip,
                                     maxIter,
                                     delta_value,
                                     ConvTolerance,   
                                     Chi2Tolerance)
        if(maxIter > 0):
            self.slv.solve()
        # TODO Hopefully this can go away
        rs.radianceStep = cfunc.radianceStep(rs.radianceStepIn)
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
        rs.stateInfo.state_info_dict["constraint"]['omi']['cloud_fraction'] = \
            rs.stateInfo.state_info_dict["current"]['omi']['cloud_fraction']
        

class RetrievalStrategyStep_tropomicloud_ig_refine(RetrievalStrategyStepRetrieve):
    '''This is a retreival, followed by using the results to update the
    TROPOMI cloud fraction.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "tropomicloud_ig_refine":
            return (False,  None)
        return super().retrieval_step(retrieval_type, rs)

    def extra_after_run_retrieval_step(self, rs):
        rs.stateInfo.state_info_dict["constraint"]['tropomi']['cloud_fraction'] = \
            rs.stateInfo.state_info_dict["current"]['tropomi']['cloud_fraction']
    
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
           
           
