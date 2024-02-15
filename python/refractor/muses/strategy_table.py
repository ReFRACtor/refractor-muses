import refractor.muses.muses_py as mpy
import refractor.framework as rf
from .misc import osp_setup
from contextlib import contextmanager
import os
from .order_species import order_species
import numpy as np

class StrategyTable:
    '''This wraps the existing muses-py routines working with the
    strategy table into a python object. '''
    def __init__(self, filename : str, osp_dir=None):
        '''Read the given strategy table.  Note that the strategy table file tends to use
        a lot of relative paths. We either assume that the directory structure is set up,
        changing to the directory of table file name. Or if the osp_dir is supplied, we set up
        a temporary directory for reading this file (useful for example to read a file sitting
        in the refractor_test_data directory).'''
        self.filename = os.path.abspath(filename)
        self._table_step = -1
        self.osp_dir = osp_dir
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

    def ils_method(self, instrument_name):
        if(instrument_name == "OMI"):
            res =  self.preferences["ils_omi_xsection"].upper()
        elif(instrument_name == "TROPOMI"):
            res =  self.preferences["ils_tropomi_xsection"].upper()
        else:
            raise RuntimeError("instrument_name must be either 'OMI' or 'TROPOMI'")
        # NOAPPLY is alias of POSTCONV
        if(res == "NOAPPLY"):
            res = "POSTCONV"
        return res
        
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
        # If we have an osp_dir, then set up a temporary directory with the OSP
        # set up
        if(self.osp_dir is not None):
            with osp_setup(osp_dir=self.osp_dir):
                yield
        else:
            # Otherwise we assume that this is in a run directory
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
        self.table_step if not specified.

        The data is returned ordered by order_species, because some of the
        muses-py code expects that.'''
        # The muses-py kind of has an odd convention for an empty list here.
        # Use this convention, and just translate this to an empty list
        r = mpy.table_get_unpacked_entry(
            self.strategy_table_dict, stp if stp is not None else self.table_step,
            "retrievalElements")
        r = mpy.flat_list(r)
        if r[0] in ('-',  ''):
            return []
        return order_species(r)
    
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
        step (defaults to self.table_step).

        The data is returned ordered by order_species, because some of the
        muses-py code expects that.'''
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
        

    def spectral_window(self, instrument_name, stp=None):
        '''This creates a rf.SpectralWindowRange for the given instrument and
        step (defaults to self.table_step). Note that a SpectralWindow has a number of
        microwindows associated with it - RefRACtor doesn't really distinguish this and
        just uses the whole SpectralWindow to choose which frequencies pass the SpectralWindow.

        This doesn't include bad sample masking, although that can be added to the
        rf.SpectralWindowRange returned.'''
        mwall = [mw for mw in self.microwindows(stp=stp) if mw['instrument'] == instrument_name]
        # May need to update logic here, not sure how to handle multiple spectral channels in
        # an instrument.
        nspec = 1
        nmw = len(mwall)
        mw_range = np.zeros((nspec, nmw, 2))
        spec_channel = 0
        for i, mw in enumerate(mwall):
            mw_range[spec_channel,i,0] = mw['start']
            mw_range[spec_channel,i,1] = mw['endd']
        mw_range = rf.ArrayWithUnit_double_3(mw_range, rf.Unit("nm"))
        return rf.SpectralWindowRange(mw_range)
        
    def microwindows(self, stp=None):
        return mpy.table_new_mw_from_step(self.strategy_table_dict,
                                          stp if stp is not None else self.table_step)
    
    def table_entry(self, nm, stp=None):
        return mpy.table_get_entry(self.strategy_table_dict,
                                   stp if stp is not None else self.table_step, nm)

class FakeStrategyTable:
    '''For testing purposes, it is useful to create a StrategyTable from a UIP, see
    StateInfo.create_from_uip for more details about this. This class supplies the handful of
    functions we need for testing. This is pretty minimal, but is sufficient for what we need.'''
    def __init__(self, ils_method="APPLY"):
        self._ils_method = ils_method

    def ils_method(self, instrument_name):
        return self._ils_method
    
__all__ = ["StrategyTable", "FakeStrategyTable"]
