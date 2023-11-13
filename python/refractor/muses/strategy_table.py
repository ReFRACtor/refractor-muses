import refractor.muses.muses_py as mpy
from contextlib import contextmanager
import os
from .order_species import order_species

class StrategyTable:
    '''This wraps the existing muses-py routines working with the
    strategy table into a python object. This is just syntactic sugar,
    we could do the same thing calling the existing routines. But nice
    to treat as any other python object, plus it gives us a good place to
    unit test stuff.'''
    def __init__(self, filename : str):
        '''Read the given strategy table.'''
        self.filename = os.path.abspath(filename)
        self._table_step = -1
        with self.chdir_run_dir():
            self.strategy_table_dict = mpy.table_read(self.filename)[1].__dict__
            

    @property
    def analysis_directory(self):
        return self.strategy_table_dict["dirAnalysis"]

    @property
    def elanor_directory(self):
        return self.strategy_table_dict["dirELANOR"]
    
    @property
    def step_directory(self):
        return self.strategy_table_dict["stepDirectory"]

    @property
    def input_directory(self):
        return self.strategy_table_dict["dirInput"]

    @property
    def pressure_fm(self):
        return self.strategy_table_dict["pressureFM"]
    
    @property
    def preferences(self) -> dict:
        '''Preferences found in the strategy table'''
        return self.strategy_table_dict["preferences"]

    @property
    def spectral_filename(self):
        with self.chdir_run_dir():
            return os.path.abspath(mpy.table_get_spectral_filename(self.strategy_table_dict, self.table_step))

    @property
    def cloud_parameters_filename(self):
        return self.abs_filename(self.preferences['CloudParameterFilename'])
    
    def abs_filename(self, rel_filename):
        '''Translate a relative path found in the StrategyTable to a
        absolute path.'''
        with self.chdir_run_dir():
            return os.path.abspath(rel_filename)
    
    @contextmanager
    def chdir_run_dir(self):
        '''A number of muses-py routines assume they are in the directory
        that the strategy table lives in. This gives a nice way to ensure
        that is the case. Uses this as a context manager
        '''
        curdir = os.getcwd()
        try:
            os.chdir(os.path.dirname(self.filename))
            yield
        finally:
            os.chdir(curdir)

    @property
    def table_step(self):
        return self._table_step

    @table_step.setter
    def table_step(self, v):
        with self.chdir_run_dir():
            self._table_step = v
            mpy.table_set_step(self.strategy_table_dict, self._table_step)

    @property
    def number_table_step(self):
        return self.strategy_table_dict["numRows"]

    @property
    def step_name(self):
        return mpy.table_get_entry(self.strategy_table_dict,
                                   self.table_step, "stepName")

    @property
    def output_directory(self):
        return self.abs_filename(self.strategy_table_dict["outputDirectory"])

    @property
    def species_directory(self):
        return self.abs_filename(self.preferences["speciesDirectory"])

    @property
    def error_species(self):
        return self.strategy_table_dict["errorSpecies"]

    @property
    def number_fm_levels(self):
        return int(self.preferences["num_FMLevels"])

    @property
    def error_map_type(self):
        return self.strategy_table_dict["errorMaptype"]

    @property
    def retrieval_type(self):
        return self.table_entry("retrievalType")

    def retrieval_elements(self, stp=None):
        '''This is the retrieval elements for the given step, defaulting to
        self.table_step if not specified.'''
        return mpy.table_get_unpacked_entry(
            self.strategy_table_dict, stp if stp is not None else self.table_step,
            "retrievalElements")

    @property
    def retrieval_elements_all_step(self):
        '''All the retrieval elements found in any of the steps.'''
        # table_get_all_values only includes muses-py species list. So we can
        # just generate this by going through all the steps
        #return mpy.table_get_all_values(self.strategy_table_dict, 'retrievalElements')
        res = set()
        for i in range(self.number_table_step):
            res.update(set(self.retrieval_elements(i)))
        res.discard('')
        return order_species(list(res))
        
    def error_analysis_interferents(self, stp=None):
        '''Interferent species/StateElement used in error analysis for the given
        step (defaults to self.table_step.'''
        # The muses-py kind of has an odd convention for an empty list here.
        # Use this convention, and just translate this to an empty list
        r = mpy.table_get_unpacked_entry(
            self.strategy_table_dict, stp if stp is not None else self.table_step,
            "errorAnalysisInterferents")
        r = mpy.flat_list(r)
        if r[0] in ('-',  ''):
            return []
        return order_species(r)
    
    @property
    def error_analysis_interferents_all_step(self):
        '''All the interferent species found in any of the steps.'''
        # table_get_all_values only includes muses-py species list. So we can
        # just generate this by going through all the steps
        #return mpy.table_get_all_values(self.strategy_table_dict, 'errorAnalysisInterferents')
        res = set()
        for i in range(self.number_table_step):
            res.update(set(self.error_analysis_interferents(i)))
        res.discard('')
        return order_species(list(res))
        
    
    @property
    def microwindows(self, stp=None):
        return mpy.table_new_mw_from_step(self.strategy_table_dict,
                                          stp if stp is not None else self.table_step)
    
    def table_entry(self, nm, stp=None):
        return mpy.table_get_entry(self.strategy_table_dict,
                                   stp if stp is not None else self.table_step, nm)
        
__all__ = ["StrategyTable", ]
