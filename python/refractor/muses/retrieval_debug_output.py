import logging
import refractor.muses.muses_py as mpy
from .retrieval_output import RetrievalOutput
import os
import pickle

# We don't have all this in place yet, but put a few samples in place for output
# triggered by having "writeOutput" which is controlled by the --debug flag set

class RetrievalInputOutput(RetrievalOutput):
    '''Write out the retrieval inputs'''
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        os.makedirs(f"{self.step_directory}/ELANORInput", exist_ok=True)
        # May need to extend this logic here
        detectorsUse = [1]
        mpy.write_retrieval_inputs(self.strategy_table.strategy_table_dict,
                                   self.stateInfo.state_info_obj,
                                   self.windows,
                                   self.retrievalInfo.retrieval_info_obj,
                                   self.table_step,
                                   self.errorCurrent.__dict__,
                                   detectorsUse)
        mpy.cdf_write_dict(self.retrievalInfo.retrieval_info_obj.__dict__,
                           f"{self.input_directory}/retrieval.nc")

class RetrievalPickleResult(RetrievalOutput):
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        os.makedirs(self.elanor_directory, exist_ok=True)
        with open(f"{self.elanor_directory}/results.pkl", "wb") as fh:
            pickle.dump(self.results.__dict__, fh)

class RetrievalPlotResult(RetrievalOutput):
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        os.makedirs(self.step_directory, exist_ok=True)
        mpy.plot_results(f"{self.step_directory}/", self.results,
                         self.retrievalInfo.retrieval_info_obj,
                         self.stateInfo.state_info_obj)

class RetrievalPlotRadiance(RetrievalOutput):
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "retrieval step"):
            return
        os.makedirs(self.analysis_directory, exist_ok=True)
        mpy.plot_radiance(self.analysis_directory, self.results,
                          self.radianceStep.__dict__, self.windows)
        
        
__all__ = ["RetrievalInputOutput", "RetrievalPickleResult", "RetrievalPlotResult",
           "RetrievalPlotRadiance"]
