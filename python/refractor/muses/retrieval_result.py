import refractor.muses.muses_py as mpy
from .observation_handle import mpy_radiance_from_observation_list
import numpy as np
import os
import logging
logger = logging.getLogger("py-retrieve")

class PropagatedQA:
    '''There are a few parameters that get propagated from one step to the next. Not
    sure exactly what this gets looked for, it look just like flags copied from one
    step to the next. But pull this together into one place so we can track this.
    '''
    def __init__(self):
        self.propagated_qa = {'TATM' : 1, 'H2O' : 1, 'O3' : 1}

    @property
    def tatm_qa(self):
        return self.propagated_qa['TATM']
    
    @property
    def h2o_qa(self):
        return self.propagated_qa['H2O']

    @property
    def o3_qa(self):
        return self.propagated_qa['O3']

    def update(self, retrieval_state_element : 'list[str]', qa_flag : int):
        '''Update the QA flags for items that we retrieved.'''
        for state_element_name in retrieval_state_element:
            if(state_element_name in self.propagated_qa):
                self.propagated_qa[state_element_name] = qa_flag
               
class RetrievalResult:
    '''There are a few top level functions that work with a structure called
    retrieval_results. Pull all this together into an object so we can clearly
    see the interface and possibly change things.

    Unlike a number of things that we want to elevate to a class, this really does
    look like just a structure of various calculated things that then get reported in
    the output files - so I think this is probably little more than wrapping up stuff in
    one place.'''
    def __init__(self, ret_res : dict, strategy_table : 'StrategyTable',
                 retrieval_info : 'RetrievalInfo', state_info : 'StateInfo',
                 obs_list : 'list(MusesObservation)',
                 propagated_qa : PropagatedQA):
        '''ret_res is what we get returned from MusesLevmarSolver'''
        self.rstep = mpy_radiance_from_observation_list(obs_list, include_bad_sample=True)
        self.retrieval_info = retrieval_info
        self.state_info = state_info
        self.strategy_table = strategy_table
        self.retrieval_result_dict = mpy.set_retrieval_results(
            strategy_table.strategy_table_dict, {}, ret_res, retrieval_info,
            self.rstep, state_info.state_info_obj,
            {"currentGuessListFM" : ret_res["xretFM"]})
        self.retrieval_result_dict = mpy.set_retrieval_results_derived(
            self.retrieval_result_dict, self.rstep, propagated_qa.tatm_qa,
            propagated_qa.o3_qa, propagated_qa.h2o_qa)
        self.retrieval_result_dict = self.retrieval_result_dict.__dict__

    def update_jacobian_sys(self, cfunc_sys : 'CostFunction'):
        '''Run the forward model in cfunc to get the jacobian_sys set.'''
        self.retrieval_result_dict['jacobianSys'] = \
            cfunc_sys.max_a_posteriori.model_measure_diff_jacobian.transpose()[np.newaxis,:,:]

    def update_error_analysis(self, error_analysis : 'ErrorAnalysis'):
        '''Run the error analysis and calculate various summary statistics for retrieval'''
        res = error_analysis.error_analysis(self.rstep, self.retrieval_info, self.state_info,
                                            self)
        res = mpy.write_retrieval_summary(
            self.strategy_table.analysis_directory,
            self.retrieval_info.retrieval_info_obj,
            self.state_info.state_info_obj,
            None,
            res,
            {},
            None,
            self.quality_name, 
            None,
            error_analysis.error_current, 
            writeOutputFlag=False, 
            errorInitial=error_analysis.error_initial
        )
        self.retrieval_result_dict = res.__dict__
                              
    @property
    def retrieval_result_obj(self):
        return mpy.ObjectView(self.retrieval_result_dict)

    @retrieval_result_obj.setter
    def retrieval_result_obj(self, v):
        self.retrieval_result_dict = v.__dict__
        
    @property
    def best_iteration(self):
        return self.retrieval_result_dict['bestIteration']

    @property
    def num_iterations(self):
        return self.retrieval_result_dict['num_iterations']
    
    @property
    def results_list(self):
        return self.retrieval_result_dict['resultsList']

    @property
    def master_quality(self):
        return self.retrieval_result_dict['masterQuality']

    @property
    def jacobian_sys(self):
        return self.retrieval_result_dict['jacobianSys']

    @property
    def press_list(self):
        return [float(self.strategy_table.preferences["plotMaximumPressure"]),
                float(self.strategy_table.preferences["plotMinimumPressure"])]

    @property
    def quality_name(self):
        with self.strategy_table.chdir_run_dir():
            res = os.path.basename(self.strategy_table.spectral_filename)
            res = res.replace("Microwindows_", "QualityFlag_Spec_")
            res = res.replace("Windows_", "QualityFlag_Spec_")
            res = self.strategy_table.preferences["QualityFlagDirectory"] + res
            
            # if this does not exist use generic nadir / limb quality flag
            if not os.path.isfile(res):
                logger.warning(f'Could not find quality flag file: {res}')
                viewingMode = self.strategy_table.preferences["viewingMode"]
                viewMode = viewingMode.lower().capitalize()

                res = f"{os.path.dirname(res)}/QualityFlag_Spec_{viewMode}.asc"
                logger.warning(f"Using generic quality flag file: {res}")
                # One last check.
                if not os.path.isfile(res):
                    raise RuntimeError(f"Quality flag filename not found: {res}")
            return os.path.abspath(res)


__all__ = ["PropagatedQA", "RetrievalResult"]    
