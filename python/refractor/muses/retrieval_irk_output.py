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
    
    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy = retrieval_strategy
        self.retrieval_strategy_step = retrieval_strategy_step
        # Save these, used in later lite files. Note these actually get
        # saved between steps, so we initialize these for the first step but
        # then leave them alone
        if(location == "retrieval step" and "dataTATM" not in self.__dict__):
            self.dataTATM = None
            self.dataH2O = None
            self.dataN2O = None
        if(location != "IRK step"):
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        logger.info("fake output for IRK")

__all__ = ["RetrievalIrkOutput", ] 
        
