import refractor.muses.muses_py as mpy
from .retrieval_strategy_step import (RetrievalStrategyStep,
                                      RetrievalStrategyStepSet)
from .observation_handle import mpy_radiance_from_observation_list
import refractor.framework as rf
from loguru import logger
import os
import numpy as np

class RetrievalStrategyStepIRK(RetrievalStrategyStep):
    '''IRK strategy step.'''
    def retrieval_step(self, retrieval_type : str,
                       rs : 'RetrievalStrategy') -> (bool, None):
        if retrieval_type != "irk":
            return (False,  None)
        logger.debug(f"Call to {self.__class__.__name__}::retrieval_step")
        logger.info("Running run_irk ...")
        fm = rs._strategy_executor.create_forward_model()
        if(not hasattr(fm, "irk")):
            raise RuntimeError(f"The forward model {fm.__class__.__name__} does not support calculating the irk")
        self.results_irk = fm.irk(rs.retrieval_info,
                                  rs._strategy_executor.rf_uip_irk)
        rs.notify_update("IRK step", retrieval_strategy_step=self)
        return (True, None)

    def observation(self, rs : 'RetrievalStrategy') -> 'MusesObservation':
        if(len(rs.current_strategy_step.instrument_name) != 1):
            raise RuntimeError("RetrievalStrategyStepIrk can only work with one instrument, we don't have handling for multiple.")
        iname = rs.current_strategy_step.instrument_name[0]
        obs = rs.observation_handle_set.observation(
            iname, None, rs.current_strategy_step.spectral_window_dict[iname], None)
        return obs
    
    def forward_model(self, rs : 'RetrievalStrategy',
                      obs : 'MusesObservation') -> rf.ForwardModel:
        if(len(rs.current_strategy_step.instrument_name) != 1):
            raise RuntimeError("RetrievalStrategyStepIrk can only work with one instrument, we don't have handling for multiple.")
        iname = rs.current_strategy_step.instrument_name[0]
        fm_sv = rf.StateVector()
        return rs.forward_model_handle_set.forward_model(
            iname, rs.current_state, obs, fm_sv, rs._strategy_executor.rf_uip_func_cost_function())
        

    
RetrievalStrategyStepSet.add_default_handle(RetrievalStrategyStepIRK())
    
__all__ = [ "RetrievalStrategyStepIRK",]
