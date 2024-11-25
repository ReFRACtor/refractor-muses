from glob import glob
from loguru import logger
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy
from .retrieval_output import RetrievalOutput, CdfWriteTes
import numpy as np

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class RetrievalIrkOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_IRK files.'''
    
    def __reduce__(self):
        return (_new_from_init, (self.__class__,))

    @property
    def retrieval_info(self) -> 'RetrievalInfo':
        return self.retrieval_strategy.retrieval_info

    @property
    def propagated_qa(self) -> 'PropagatedQa':
        return self.retrieval_strategy.propagated_qa

    @property
    def results_irk(self) -> 'ObjectView':
        return mpy.ObjectView(self.retrieval_strategy_step.results_irk)
    
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        if(location != "IRK step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        logger.info("fake output for IRK")
        self.out_fname = f"{self.output_directory}/Products/Products_IRK"
        os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
        mpy.write_products_irk_one(
            self.out_fname,
            self.results_irk,
            self.retrieval_info.retrieval_info_obj,
            self.state_info.state_info_obj,
            self.propagated_qa.tatm_qa,
            self.propagated_qa.o3_qa,
            self.step_number)

__all__ = ["RetrievalIrkOutput", ] 
        
