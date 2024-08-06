from .retrieval_strategy_step_new import RetrievalStrategyStepSetNew
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .strategy_table import StrategyTable
from .error_analysis import ErrorAnalysis
from .spectral_window_handle import SpectralWindowHandleSet
import abc
import copy
import logging
import time
import functools
logger = logging.getLogger("py-retrieve")

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

class CurrentStrategyStepDict(CurrentStrategyStep):
    '''Implementation of CurrentStrategyStep that uses a dict'''
    def __init__(self, current_strategy_step_dict : dict):
        self.current_strategy_step_dict = current_strategy_step_dict

    @property
    def retrieval_elements(self) -> 'list(str)':
        '''List of retrieval elements that we retrieve for this step.'''
        return self.current_strategy_step_dict['retrieval_elements']
    
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
        
    
class MusesStrategyExecutorRetrievalStrategyStep(MusesStrategyExecutor):
    '''Much of the time our strategy is going to depend on having
    a RetrievalStrategyStepSet to get the RetrievalStrategyStep based
    off a retrieval type name. This adds that functionality.'''
    def __init__(self, retrieval_strategy_step_set=None,
                 spectral_window_handle_set=None):
        if(retrieval_strategy_step_set is None):
            self._retrieval_strategy_step_set = copy.deepcopy(RetrievalStrategyStepSetNew.default_handle_set())
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set
        if(spectral_window_handle_set is None):
            self._spectral_window_handle_set = copy.deepcopy(SpectralWindowHandleSet.default_handle_set())
        else:
            self._spectral_window_handle_set = spectral_window_handle_set
            

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        '''Return the CurrentStrategyStep for the current step.'''
        raise NotImplementedError()
    
    @property
    def spectral_window_handle_set(self):
        '''The SpectralWindowHandleSet to use for getting the MusesSpectralWindow.'''
        return self._spectral_window_handle_set

    @property
    def retrieval_strategy_step_set(self):
        '''The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep.'''
        return self._retrieval_strategy_step_set

class MusesStrategyExecutorOldStrategyTable(MusesStrategyExecutorRetrievalStrategyStep):
    '''Placeholder that wraps the muses-py strategy table up, so we can get the
    infrastructure in place before all the pieces are ready'''

    def __init__(self, filename : str, rs : 'RetrievalStrategy', osp_dir=None):
        super().__init__(retrieval_strategy_step_set = copy.deepcopy(RetrievalStrategyStepSet.default_handle_set()))
        self.stable = StrategyTable(filename, osp_dir=osp_dir)
        self.rs = rs

    def filter_list_all(self):
        return self.stable.filter_list_all()

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        '''Return the CurrentStrategyStep for the current step.'''
        return CurrentStrategyStepDict(
            {'retrieval_elements' : self.stable.retrieval_elements(),
             'step_name' : self.stable.step_name,
             'step_number' : self.stable.table_step,
             'max_num_iterations' : self.stable.max_num_iterations,
             'retrieval_type' : self.stable.retrieval_type
             })

    def restart(self):
        '''Set step to the first one.'''
        self.stable.table_step = 0

    def next_step(self):
        '''Advance to the next step'''
        self.stable.table_step = self.stable.table_step+1

    def is_done(self):
        '''Return true if we are done, otherwise false.'''
        return self.stable.table_step >= self.stable.number_table_step
    
    def run_step(self):
        '''Run a the current step.'''
        self.rs._swin_dict = self.spectral_window_handle_set.spectral_window_dict(self.current_strategy_step)
        self.rs._state_info.copy_current_initial()
        self.rs._strategy_table = self.stable
        logger.info(f'\n---')
        logger.info(f"Step: {self.current_strategy_step.step_number}, Step Name: {self.current_strategy_step.step_name}, Total Steps: {self.stable.number_table_step}")
        logger.info(f'\n---')
        self.rs.get_initial_guess()
        self.rs.notify_update("done get_initial_guess")
        logger.info(f"Step: {self.current_strategy_step.step_number}, Retrieval Type {self.current_strategy_step.retrieval_type}")
        self.rs._retrieval_strategy_step_set.retrieval_step(self.current_strategy_step.retrieval_type, self.rs)
        self.rs.notify_update("done retrieval_step")
        self.rs._state_info.next_state_to_current()
        self.rs.notify_update("done next_state_to_current")
        logger.info(f"Done with step {self.current_strategy_step.step_number}")

    @log_timing
    def execute_retrieval(self):
        '''Run through all the steps, i.e., do a full retrieval.'''
        self.state_info = self.rs._state_info
        with self.stable.chdir_run_dir():
            self.state_info.init_state(self.stable,
                                       self.rs.observation_handle_set,
                                       self.stable.instrument_name(all_step=True),
                                       self.rs.run_dir)
        self.error_analysis = ErrorAnalysis(self.stable, self.state_info)
        self.rs._error_analysis = self.error_analysis
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
            self.rs.get_initial_guess()
            self.next_step()
        # Not sure that this is needed or used anywhere, but for now go ahead and this
        # this until we know for sure it doesn't matter.
        self.rs._state_info.copy_current_initialInitial()
        self.rs.notify_update("starting retrieval steps")
        self.restart()
        while(not self.is_done()):
            self.run_step()
            self.next_step()
        
__all__ = ["MusesStrategyExecutor", "CurrentStrategyStep", "CurrentStrategyStepDict",
           "MusesStrategyExecutorRetrievalStrategyStep",
           "MusesStrategyExecutorOldStrategyTable"]
    
