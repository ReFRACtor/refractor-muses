from .strategy_table import StrategyTable
from .refractor_state_info import RefractorStateInfo
import refractor.muses.muses_py as mpy
import copy
import numpy as np

class ErrorAnalysis:
    '''This just groups together some of the error analysis stuff together,
    to put this together. Nothing more than a shuffling around of stuff already
    in muses-py
    '''
    def __init__(self, strategy_table: StrategyTable, state_info: RefractorStateInfo):
        smeta = state_info.sounding_metadata()
        self.error_initial = mpy.get_prior_error(
            strategy_table.error_species, strategy_table.error_map_type,
            state_info.pressure, smeta.latitude.value, state_info.nh3type,
            state_info.hcoohtype, state_info.ch3ohtype,
            "OCEAN" if smeta.is_ocean else "LAND")
        self.error_current = copy.deepcopy(self.error_initial)
        # Code seems to assume these are object view.
        self.error_initial = mpy.ObjectView(self.error_initial)
        self.error_current = mpy.ObjectView(self.error_current)

    def error_analysis(self, rs: "RetrievalStrategy", results):
        '''Update results and error_current'''
        # Doesn't seem to be used for anything, but we need to pass in. I think
        # this might have been something that was used in the past?
        radianceNoise = {"radiance" : np.zeros_like(rs.radianceStep["radiance"]) }
        (results, self.error_current) = mpy.error_analysis_wrapper(
            rs.table_step,
            rs.strategy_table.analysis_directory,
            rs.radianceStep,
            radianceNoise,
            rs.retrievalInfo.retrieval_info_obj,
            rs.state_info.state_info_obj,
            self.error_initial,
            self.error_current,
            rs.windows,
            results
            )
        return results

        
