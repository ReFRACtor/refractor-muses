from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy

logger = logging.getLogger("py-retrieve")

def _new_from_init(cls, *args):
    '''For use with pickle, covers common case where we just store the
    arguments needed to create an object.'''
    inst = cls.__new__(cls)
    inst.__init__(*args)
    return inst

class RetrievalOutput:
    '''Observer of RetrievalStrategy, common behavior for Products files.'''
    def notify_add(self, retrieval_strategy):
        self.retrieval_strategy = retrieval_strategy

    def notify_update(self, retrieval_strategy, location, retrieval_strategy_step=None,
                      **kwargs):
        self.retrieval_strategy_step = retrieval_strategy_step

    @property
    def strategy_table(self):
        return self.retrieval_strategy.strategy_table

    @property
    def step_directory(self):
        return self.strategy_table.step_directory

    @property
    def input_directory(self):
        return self.strategy_table.input_directory

    @property
    def analysis_directory(self):
        return self.strategy_table.analysis_directory

    @property
    def elanor_directory(self):
        return self.strategy_table.elanor_directory
    
    @property
    def windows(self):
        return self.retrieval_strategy.windows

    @property
    def errorCurrent(self):
        return self.retrieval_strategy.errorCurrent
    
    @property
    def special_tag(self):
        if self.retrieval_strategy.retrieval_type != 'default':
            return f"-{self.retrieval_strategy.retrieval_type}"
        return ""

    @property
    def species_tag(self):
        res = self.retrieval_strategy.step_name
        res = res.rstrip(', ')
        if 'EMIS' in res and res.index('EMIS') > 0:
            res = res.replace('EMIS', '')
        if res.endswith(',_OMI'):
            res = res.replace(',_OMI', '_OMI')  #  Change "H2O,O3,_OMI" to "H2O,O3_OMI"
        res = res.rstrip(', ')
        return res

    @property
    def quality_name(self):
        return self.retrieval_strategy.quality_name
    
    @property
    def table_step(self):
        return self.retrieval_strategy.table_step

    @property
    def number_table_step(self):
        return self.retrieval_strategy.number_table_step

    @property
    def results(self):
        return self.retrieval_strategy_step.results

    @property
    def state_info(self):
        return self.retrieval_strategy.state_info

    @property
    def radiance_full(self):
        return self.retrieval_strategy.fm_obs_creator.radiance()
    
    @property
    def radianceStep(self):
        return mpy.ObjectView(self.retrieval_strategy.radianceStep)

    @property
    def retrievalInfo(self):
        return self.retrieval_strategy.retrievalInfo
    
    @property
    def instruments(self):
        return self.retrieval_strategy.instruments
    

__all__ = ["RetrievalOutput",] 
