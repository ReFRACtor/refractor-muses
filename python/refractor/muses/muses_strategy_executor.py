from .retrieval_strategy_step_new import RetrievalStrategyStepSetNew
from .refractor_capture_directory import muses_py_call
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .retrieval_info import RetrievalInfo
from .strategy_table import StrategyTable
from .error_analysis import ErrorAnalysis
from .order_species import order_species
from .spectral_window_handle import SpectralWindowHandleSet
from .current_state import CurrentStateStateInfo
from .muses_spectral_window import MusesSpectralWindow
from .qa_data_handle import QaDataHandleSet
from .refractor_uip import RefractorUip
import refractor.muses.muses_py as mpy
import abc
import copy
from loguru import logger
import time
import functools
import numpy as np
import numpy.testing as npt

def log_timing(f):
    '''Decorator to log the timing of a function.'''
    @functools.wraps(f)
    def log_tm(*args, **kwargs):
        start_date = time.strftime("%c")
        start_time = time.time()
        res = f(*args, **kwargs)
        stop_date = time.strftime("%c")
        stop_time = time.time()
        elapsed_time = stop_time - start_time
        elapsed_time_seconds = stop_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60.0
        logger.info('\n---')    
        logger.info(f"start_date {start_date}")
        logger.info(f"stop_date {stop_date}")
        logger.info(f"elapsed_time {elapsed_time}")
        logger.info(f"elapsed_time_seconds {elapsed_time_seconds}")
        logger.info(f"elapsed_time_minutes {elapsed_time_minutes}")
        return res
    return log_tm

def struct_compare(s1, s2, skip_list=None, verbose=False):
    if(skip_list is None):
        skip_list = []
    for k in s1.keys():
        if(k in skip_list):
            if(verbose):
                print(f"Skipping {k}")
            continue
        if(verbose):
            print(k)
        if(isinstance(s1[k], np.ndarray) and
           np.can_cast(s1[k], np.float64)):
           npt.assert_allclose(s1[k], s2[k])
        elif(isinstance(s1[k], np.ndarray)):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]

def array_compare(s1, s2, skip_list=None, verbose=False):
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        struct_compare(s1[i], s2[i], skip_list=skip_list, verbose=verbose)

class MusesStrategyExecutor(object, metaclass=abc.ABCMeta):
    '''This is the base class for executing a strategy.

    Note that there a refractor.framework class StrategyExecutor. This class has a
    similar intention as that older StrategyExecutor class, however this really
    is a complete rewrite of this for the way py-retrieve does this. It is possible
    that these classes might get merged at some point, but for now it is better
    to think of these as completely separate classes that just happen to have similar
    names.

    The canonical way of determining the strategy is to read the old strategy table
    ("Table.asc") that amuse-me populates.

    This base class provides an abstract interface so we can have different implementations
    of executing a strategy.
    '''
    pass

class CurrentStrategyStep(object, metaclass=abc.ABCMeta):
    '''This contains information about the current strategy step. This is
    little more than a dict giving several properties, but we abstract this
    out so we can test things without needing to use a full MusesStrategyExecutor,
    also so we document what information is expected from a strategy step.'''
    @abc.abstractproperty
    def retrieval_elements(self) -> 'list(str)':
        '''List of retrieval elements that we retrieve for this step.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> 'list(str)':
        '''List of instruments used in this step.'''
        raise NotImplementedError()
        
    @abc.abstractproperty
    def step_name(self) -> str:
        '''A name for the current strategy step.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def step_number(self) -> int:
        '''The number of the current strategy step, starting with 0.'''
        raise NotImplementedError()
    

    @abc.abstractproperty
    def microwindow_file_name_override(self) -> 'Optional(str)':
        '''The microwindows file to use, overriding the normal logic that was
        in the old mpy.table_get_spectral_filename. If None, then we don't have
        an override.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def max_num_iterations(self) -> int:
        '''Maximum number of iterations to used in a retrieval step.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_type(self) -> str:
        '''The retrieval type.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def do_not_update_list(self) -> "list(str)":
        raise NotImplementedError()
        
    @abc.abstractproperty
    def spectral_window_dict(self) -> "dict(str, MusesSpectralWindow)":
        '''Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that.'''
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_info(self) -> RetrievalInfo:
        '''The RetrievalInfo.'''
        # Note it would probably be good to remove this if we can. Right now
        # this is only used by RetrievalL2Output. But at least for now, we
        # need to to generate the output
        raise NotImplementedError()
    
class CurrentStrategyStepDict(CurrentStrategyStep):
    '''Implementation of CurrentStrategyStep that uses a dict'''
    def __init__(self, current_strategy_step_dict : dict):
        self.current_strategy_step_dict = current_strategy_step_dict

    @classmethod
    def current_step(cls, strategy_table : StrategyTable):
        '''Create a current strategy step, leaving out the RetrievalInfo stuff. Mostly
        meant for testing, MusesStrategyExecutor will normally create the current_step
        but testing at a lower level might not have a MusesStrategyExecutor available.'''
        return cls(
            {'retrieval_elements' : strategy_table.retrieval_elements(),
             'instrument_name' : strategy_table.instrument_name(),
             'step_name' : strategy_table.step_name,
             'step_number' : strategy_table.table_step,
             'max_num_iterations' : int(strategy_table.max_num_iterations),
             'retrieval_type' : strategy_table.retrieval_type,
             'do_not_update_list' : strategy_table.do_not_update_list,
             'spectral_window_dict' : None,
             'retrieval_info' : None
             })
    
    @property
    def retrieval_elements(self) -> 'list(str)':
        '''List of retrieval elements that we retrieve for this step.'''
        return self.current_strategy_step_dict['retrieval_elements']

    @property
    def instrument_name(self) -> 'list(str)':
        '''List of instruments used in this step.'''
        return self.current_strategy_step_dict['instrument_name']
    
    @property
    def step_name(self) -> str:
        '''A name for the current strategy step.'''
        return self.current_strategy_step_dict['step_name']

    @property
    def step_number(self) -> int:
        '''The number of the current strategy step, starting with 0.'''
        return self.current_strategy_step_dict['step_number']
    

    @property
    def microwindow_file_name_override(self) -> 'Optional(str)':
        '''The microwindows file to use, overriding the normal logic that was
        in the old mpy.table_get_spectral_filename. If None, then we don't have
        an override.'''
        return self.current_strategy_step_dict.get("microwindow_file_name_override")

    @property
    def max_num_iterations(self) -> int:
        '''Maximum number of iterations to used in a retrieval step.'''
        return self.current_strategy_step_dict['max_num_iterations']
        

    @property
    def retrieval_type(self) -> str:
        '''The retrieval type.'''
        return self.current_strategy_step_dict['retrieval_type']

    @property
    def do_not_update_list(self) -> "list(str)":
        return self.current_strategy_step_dict['do_not_update_list']

    @property
    def spectral_window_dict(self) -> "dict(str, MusesSpectralWindow)":
        '''Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that.'''
        return self.current_strategy_step_dict['spectral_window_dict']
    
    @property
    def retrieval_info(self) -> RetrievalInfo:
        '''The RetrievalInfo.'''
        # Note it would probably be good to remove this if we can. Right now
        # this is only used by RetrievalL2Output. But at least for now, we
        # need to to generate the output
        return self.current_strategy_step_dict["retrieval_info"]
    
class MusesStrategyExecutorRetrievalStrategyStep(MusesStrategyExecutor):
    '''Much of the time our strategy is going to depend on having
    a RetrievalStrategyStepSet to get the RetrievalStrategyStep based
    off a retrieval type name. This adds that functionality.'''
    def __init__(self, retrieval_strategy_step_set=None,
                 spectral_window_handle_set=None,
                 qa_data_handle_set=None):
        if(retrieval_strategy_step_set is None):
            self._retrieval_strategy_step_set = copy.deepcopy(RetrievalStrategyStepSetNew.default_handle_set())
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set
        if(spectral_window_handle_set is None):
            self._spectral_window_handle_set = copy.deepcopy(SpectralWindowHandleSet.default_handle_set())
        else:
            self._spectral_window_handle_set = spectral_window_handle_set
        if(qa_data_handle_set is None):
            self._qa_data_handle_set = copy.deepcopy(QaDataHandleSet.default_handle_set())
        else:
            self._qa_data_handle_set = qa_data_handle_set
            

    @property
    def filter_list_dict(self) -> 'dict(str,list[str])':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        raise NotImplementedError

    @property
    def number_retrieval_step(self) -> int:
        '''Total number of retrieval steps. Note that this might change as
        we work through the retrieval based off decisions from early steps.'''
        # I think this is only needed for logging messages, we might be
        # able to remove this if it proves hard to calculate
        raise NotImplementedError
    
    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        '''Return the CurrentStrategyStep for the current step.'''
        raise NotImplementedError()
    
    @property
    def spectral_window_handle_set(self):
        '''The SpectralWindowHandleSet to use for getting the MusesSpectralWindow.'''
        return self._spectral_window_handle_set

    @property
    def qa_data_handle_set(self):
        '''The QaDataHandleSet to use to get the QA flag filename.'''
        return self._qa_data_handle_set
    
    @property
    def retrieval_strategy_step_set(self):
        '''The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep.'''
        return self._retrieval_strategy_step_set

    def uip_func(self, instrument=None, do_systematic=False, jacobian_speciesIn=None):
        '''To reduce coupling, you can give the instrument name to use. The default
        is None, which means to create all instruments used in this step.'''
        if(do_systematic):
            rinfo = self.retrieval_info.retrieval_info_systematic()
        else:
            rinfo = self.retrieval_info
        o_xxx = {"AIRS" : None, "TES" : None, "CRIS" : None, "OMI" : None,
                 "TROPOMI" : None, "OCO2" : None}
        o_airs = None
        o_tes = None
        o_cris = None
        o_omi = None
        o_tropomi = None
        o_oco2 = None
        cstep = self.current_strategy_step
        for iname in cstep.instrument_name:
            if iname in o_xxx:
                if(instrument is None or iname == instrument):
                    obs = self.rs.observation_handle_set.observation(
                        iname, None, cstep.spectral_window_dict[iname],None)
                    if hasattr(obs, "muses_py_dict"):
                        o_xxx[iname] = obs.muses_py_dict
        mwin = []
        for swin in cstep.spectral_window_dict.values():
            mwin.extend(swin.muses_microwindows())
        if False:
            # Can check that we are getting the right microwindows, if needed
            mwin2 = self.stable.microwindows()
            array_compare(mwin, mwin2, skip_list=["THROW_AWAY_WINDOW_INDEX"])
        # Dummy strategy table, with the information needed by
        # RefractorUip.create_uip
        fake_table = { 'preferences' : self.retrieval_config,
                       'vlidort_dir' : f"{self.rs.run_dir}/Step{self.current_strategy_step.step_number:02d}_{self.current_strategy_step.step_name}/vlidort/",
                       'numRows' : cstep.step_number,
                       'numColumns' : 1,
                       'step' : cstep.step_number,
                       'labels1' : "retrievalType",
                       'data' : [cstep.retrieval_type] * cstep.step_number }
        with muses_py_call(self.rs.run_dir,
                           vlidort_cli=self.rs.vlidort_cli):
            return RefractorUip.create_uip(
                self.state_info, fake_table, mwin, rinfo,
                o_xxx["AIRS"], o_xxx["TES"], o_xxx["CRIS"],
                o_xxx["OMI"], o_xxx["TROPOMI"], o_xxx["OCO2"],
                jacobian_speciesIn=jacobian_speciesIn,
                only_create_instrument=instrument)

    def create_cost_function(self, do_systematic=False, include_bad_sample=False,
                             fix_apriori_size=False, jacobian_speciesIn=None):
        '''Create a CostFunction, for use either in retrieval or just for running
        the forward model (the CostFunction is a little overkill for just a
        forward model run, but it has all the pieces needed so no reason not to
        just generate everything).

        If do_systematic is True, then we use the systematic species list. '''
        cstate = CurrentStateStateInfo(
            self.state_info, self.retrieval_info,
            f"{self.rs.run_dir}/Step{self.current_strategy_step.step_number:02d}_{self.current_strategy_step.step_name}",
            do_systematic=do_systematic,
            retrieval_state_element_override=jacobian_speciesIn)
        
        # TODO Would probably be good to remove include_bad_sample, it isn't clear that
        # we ever want to run the forward model for bad samples. But right now the
        # existing
        # py-retrieve code requires this is a few places.a
        return self.rs._cost_function_creator.cost_function(
            self.current_strategy_step.instrument_name,
            cstate,
            self.current_strategy_step.spectral_window_dict,
            functools.partial(self.uip_func, do_systematic=do_systematic,
                              jacobian_speciesIn=jacobian_speciesIn),
            include_bad_sample=include_bad_sample, **self.rs._kwargs)

class MusesStrategyExecutorOldStrategyTable(MusesStrategyExecutorRetrievalStrategyStep):
    '''Placeholder that wraps the muses-py strategy table up, so we can get the
    infrastructure in place before all the pieces are ready'''

    def __init__(self, filename : str, rs : 'RetrievalStrategy', osp_dir=None,
                 retrieval_strategy_step_set=None,
                 spectral_window_handle_set=None,
                 qa_data_handle_set=None):
        super().__init__(retrieval_strategy_step_set = retrieval_strategy_step_set,
                         spectral_window_handle_set = spectral_window_handle_set,
                         qa_data_handle_set = qa_data_handle_set)
        self.stable = StrategyTable(filename, osp_dir=osp_dir)
        self.rs = rs
        self.retrieval_config = rs.retrieval_config
        self.retrieval_info = None
        self.spectral_window_dict = None

    @property
    def strategy_table_filename(self):
        return self.stable.filename

    @strategy_table_filename.setter
    def strategy_table_filename(self, v):
        self.filename = v
        
    @property
    def filter_list_dict(self) -> 'dict(str,list[str])':
        '''The complete list of filters we will be processing (so for all retrieval steps)
        '''
        return self.stable.filter_list_all()

    @property
    def number_retrieval_step(self) -> int:
        '''Total number of retrieval steps. Note that this might change as
        we work through the retrieval based off decisions from early steps.'''
        return self.stable.number_table_step

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        '''Return the CurrentStrategyStep for the current step.'''
        return CurrentStrategyStepDict(
            {'retrieval_elements' : self.stable.retrieval_elements(),
             'instrument_name' : self.stable.instrument_name(),
             'step_name' : self.stable.step_name,
             'step_number' : self.stable.table_step,
             'max_num_iterations' : int(self.stable.max_num_iterations),
             'retrieval_type' : self.stable.retrieval_type,
             'do_not_update_list' : self.stable.do_not_update_list,
             'spectral_window_dict' : self.spectral_window_dict,
             'retrieval_info' : self.retrieval_info
             })

    def restart(self):
        '''Set step to the first one.'''
        self.stable.table_step = 0
        self.spectral_window_dict = self.spectral_window_handle_set.spectral_window_dict(self.current_strategy_step)

    def next_step(self):
        '''Advance to the next step'''
        self.stable.table_step = self.stable.table_step+1

    def is_done(self):
        '''Return true if we are done, otherwise false.'''
        return self.stable.table_step >= self.stable.number_table_step

    @property
    def instrument_name_all_step(self):
        return self.stable.instrument_name(all_step=True)

    def get_initial_guess(self):
        '''Set retrieval_info, errorInitial and errorCurrent for the current step.'''
        # Temp, we'll want to get this update done automatically. But do this
        # to figure out issue
        self.spectral_window_dict = self.spectral_window_handle_set.spectral_window_dict(self.current_strategy_step)
        self.retrieval_info = RetrievalInfo(
            self.error_analysis, self.stable,
            self.current_strategy_step,
            self.state_info)

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self.retrieval_info.n_totalParameters
        logger.info(f"Step: {self.current_strategy_step.step_number}, Total Parameters: {nparm}")

        if nparm > 0:
            xig = self.retrieval_info.initial_guess_list[0:nparm]
            self.state_info.update_state(self.retrieval_info, xig, [],
                          self.retrieval_config, self.current_strategy_step.step_number)

    def number_steps_left(self, retrieval_element_name : str):
        '''This returns the number of retrieval steps left that contain a given
        retrieval element name. This is an odd seeming function, but is used by
        RetrievalL2Output to name files. So for example we have Products_L2-O3-0.nc
        for the last step that retrieves O3, Products_L2-O3-1.nc for the previous step
        retrieving O3, etc.
        
        I'm not sure if this is something we can calculate in general for a
        StrategyExecutor (what if some decision is added if a future step is run or not?)
        If this occurs, we can perhaps come up with a different naming convention.
        Right now, this function is *only* used in RetrievalL2Output, so we can
        update this if needed.
        '''
        step_number_start = self.current_strategy_step.step_number
        res = 0
        self.next_step()
        while(not self.is_done()):
            if retrieval_element_name in self.current_strategy_step.retrieval_elements:
                res += 1
            self.next_step()
        self.stable.table_step = step_number_start
        return res
            
    def run_step(self):
        '''Run a the current step.'''
        self.spectral_window_dict = self.spectral_window_handle_set.spectral_window_dict(self.current_strategy_step)
        self.rs._state_info.copy_current_initial()
        logger.info(f'\n---')
        logger.info(f"Step: {self.current_strategy_step.step_number}, Step Name: {self.current_strategy_step.step_name}, Total Steps: {self.stable.number_table_step}")
        logger.info(f'\n---')
        self.get_initial_guess()
        self.rs.notify_update("done get_initial_guess")
        logger.info(f"Step: {self.current_strategy_step.step_number}, Retrieval Type {self.current_strategy_step.retrieval_type}")
        self.rs._retrieval_strategy_step_set.retrieval_step(self.current_strategy_step.retrieval_type, self.rs)
        self.rs.notify_update("done retrieval_step")
        self.rs._state_info.next_state_to_current()
        self.rs.notify_update("done next_state_to_current")
        logger.info(f"Done with step {self.current_strategy_step.step_number}")
        
    @log_timing
    def execute_retrieval(self, stop_at_step=None):
        '''Run through all the steps, i.e., do a full retrieval.
        
        Note for various testing purposes, you can have the retrieval stop at the given
        step. This can be useful for looking at problems with an individual step, or
        to run a simulation at a particular step.'''
        # Currently the initialization etc. code assumes we are in the run directory.
        # Hopefully we can remove this in the future, but for now we need this
        with muses_py_call(self.rs.run_dir,
                           vlidort_cli=self.rs.vlidort_cli):
            self.execute_retrieval_body(stop_at_step=stop_at_step)
        
    def execute_retrieval_body(self, stop_at_step=None):
        '''Run through all the steps, i.e., do a full retrieval.
        
        Note for various testing purposes, you can have the retrieval stop at the given
        step. This can be useful for looking at problems with an individual step, or
        to run a simulation at a particular step.'''
        self.state_info = self.rs._state_info
        with self.stable.chdir_run_dir():
            self.state_info.init_state(self.stable,
                                       self.rs.observation_handle_set,
                                       self.instrument_name_all_step,
                                       self.rs.run_dir)
            
        # List of state elements we need covariance from. This is all the elements
        # we will retrieve, plus any interferents that get added in. This list
        # is unique elements, sorted by the order_species sorting
        
        covariance_state_element_name = order_species(
            set(self.stable.retrieval_elements_all_step) |
            set(self.stable.error_analysis_interferents_all_step))

        self.restart()
        self.error_analysis = ErrorAnalysis(
            self.current_strategy_step,
            self.state_info,
            covariance_state_element_name)
        self.rs.notify_update("initial set up done")
        
        # Note the original muses-py ran through all the initial guess steps at
        # the beginning to make sure there weren't any issues. I think we can remove
        # this, it isn't particularly important to fail early and it seems a waste
        # of time to go through this twice.
        #
        # Have, the output actually changes if we don't run this. This is bad, our
        # initial guess shouldn't modify future running. We should track this down
        # when we start working on the initial guess/state info portion. But for now,
        # leave this in place until we understand this
        self.restart()
        while(not self.is_done()):
            self.get_initial_guess()
            self.next_step()
        # Not sure that this is needed or used anywhere, but for now go ahead and this
        # this until we know for sure it doesn't matter.
        self.rs._state_info.copy_current_initialInitial()
        self.rs.notify_update("starting retrieval steps")
        self.restart()
        while(not self.is_done()):
            if(stop_at_step is not None and
               stop_at_step == self.current_strategy_step.step_number):
                return
            self.rs.notify_update("starting run_step")
            self.run_step()
            self.next_step()

    def continue_retrieval(self):
        '''After saving a pickled step, you can continue the processing starting
        at that step to diagnose a problem.'''
        while(not self.is_done()):
            self.rs.notify_update("starting run_step")
            self.run_step()
            self.next_step()
            
        
__all__ = ["MusesStrategyExecutor", "CurrentStrategyStep", "CurrentStrategyStepDict",
           "MusesStrategyExecutorRetrievalStrategyStep",
           "MusesStrategyExecutorOldStrategyTable"]
    
