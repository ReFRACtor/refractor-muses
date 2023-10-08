import abc
import logging
import refractor.muses.muses_py as mpy
from pathlib import Path
import copy
from .priority_handle_set import PriorityHandleSet
from pprint import pprint, pformat

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

        # TODO Put back in writeOutput, we don't have this in our version
        # of run_retrieval
        retrievalResults = rs.run_retrieval()

        rs.results = mpy.set_retrieval_results(rs.strategy_table, rs.windows, retrievalResults, rs.retrievalInfo, rs.radianceStep, rs.stateInfo.state_info_obj,
                             {"currentGuessListFM" : retrievalResults["xretFM"]})
        logger.info('\n---')
        logger.info(f"Step: {rs.table_step}, Step Name: {rs.step_name}")
        logger.info(f"Best iteration {rs.results.bestIteration} out of {retrievalResults['num_iterations']}")
        logger.info('---\n')
        rs.results = mpy.set_retrieval_results_derived(rs.results,
                      rs.radianceStep, rs.propagatedTATMQA,
                      rs.propagatedO3QA, rs.propagatedH2OQA)
        
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
                             rs.results.resultsList, rs.cloud_prefs,
                             rs.table_step, donotupdate, rs.stateOneNext)
        rs.stateInfo.state_info_dict = rs.stateInfo.state_info_dict.__dict__
        if 'OCO2' in rs.instruments:
            # set table.pressurefm to stateConstraint.pressure because OCO-2
            # is on sigma levels
            rs.strategy_table['pressureFM'] = rs.stateOneNext.pressure
        self.extra_after_run_retrieval_step(rs)
        rs.notify_update("run_retrieval_step")

        # We should pull some of this over, but for now call existing code
        rs.systematic_jacobian()

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
        mpy.set_retrieval_results_derived(rs.results, rs.radianceStep,
                                          rs.propagatedTATMQA, rs.propagatedO3QA,
                                          rs.propagatedH2OQA)        
        rs.error_analysis()
        rs.update_retrieval_summary()
        rs.notify_update("retrieval step")
        
        return (True, None)

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
           
           
