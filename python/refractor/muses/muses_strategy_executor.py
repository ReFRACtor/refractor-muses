from .retrieval_strategy_step_new import RetrievalStrategyStepSetNew
import abc
import copy

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
    def __init__(self, retrieval_strategy_step_set=None):
        if(retrieval_strategy_step_set is None):
            self._retrieval_strategy_step_set = copy.deepcopy(RetrievalStrategyStepSetNew.default_handle_set())
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set

    @property
    def current_strategy_step(self):
        '''Return the CurrentStrategyStep for the current step.'''
        raise NotImplementedError()
    
    @property
    def retrieval_strategy_step_set(self):
        '''The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep.'''
        return self._retrieval_strategy_step_set

__all__ = ["MusesStrategyExecutor", "CurrentStrategyStep", "CurrentStrategyStepDict",
           "MusesStrategyExecutorRetrievalStrategyStep"]
    
