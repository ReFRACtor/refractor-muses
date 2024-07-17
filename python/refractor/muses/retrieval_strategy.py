from .refractor_capture_directory import (RefractorCaptureDirectory,
                                          muses_py_call)
from .retrieval_l2_output import RetrievalL2Output
from .retrieval_radiance_output import RetrievalRadianceOutput
from .retrieval_jacobian_output import RetrievalJacobianOutput
from .retrieval_debug_output import (RetrievalInputOutput, RetrievalPickleResult,
                                     RetrievalPlotRadiance, RetrievalPlotResult)
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .strategy_table import StrategyTable
from .retrieval_configuration import RetrievalConfiguration
from .cost_function_creator import CostFunctionCreator
from .muses_observation import MeasurementIdFile
from .forward_model_handle import ForwardModelHandleSet
from .observation_handle import ObservationHandleSet
from .state_info import StateElementHandleSet
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .error_analysis import ErrorAnalysis
from .muses_strategy_executor import MusesStrategyExecutorOldStrategyTable
import logging
import refractor.muses.muses_py as mpy
import os
import copy
import numpy as np
import numpy.testing as npt
import pickle
from pathlib import Path
from pprint import pformat, pprint
import time
from contextlib import contextmanager
from .retrieval_info import RetrievalInfo
from .state_info import StateInfo
logger = logging.getLogger("py-retrieve")

# We could make this an rf.Observable, but no real reason to push this to a C++
# level. So we just have a simple observation set here
class RetrievalStrategy(mpy.ReplaceFunctionObject if mpy.have_muses_py else object):
    '''
    This is a replacement for script_retrieval_ms, that tries to do a
    few things:

    1. Simplifies the core code, the script_retrieval_ms is really
        pretty long and is a sequence of "do one thing, then another,
        then aother". We do this by:

    2. Moving output out of this class, and having separate classes
       handle this. We use the standard ReFRACtor approach of having
       observers. This tend to give a much cleaner interface with
       clear seperation.

    3. Adopt a extensively, configurable way to handle the initial
       guess (similiar to the OCO-2 InitialGuess structure)

    4. Handle species information as a separate class, which allows us
       to easily extend the list of jacobian parameters (e.g, add
       EOFs). The existing code uses long lists of hardcoded values,
       this attempts to be a more adaptable.

    This has a number of advantages, for example having InitialGuess
    separated out allows us to do unit testing in ways that don't
    require updating the OSP directories with new covariance stuff,
    for example.

    Note that there is a lot of overlap between this class and the
    MusesStrategyExecutor class. It isn't clear that long term there will
    actually be two separate classes. However for right now this is a
    useful division of responsibilities:

    1. RetrievalStrategy worries about the interface with external classes.
       What does this look like to the muses-py driver? What is exposed to
       the output classes? How does configuration modify things?
    2. MusesStrategyExecutor worries about actually running the strategy. How
       do we determine the retrieval steps? How do we run the retrieval
       steps?

    This may well merge once we have the external interface sorted out.

    Note that is class has a number of internal variables, with the normal
    python "private" suggestion of using a leading "_", e.g. "_capture_directory".
    It is a normal python convention that external classes not use this private
    variables. But this should be even stronger for this class - one of the primary
    things we are trying to figure out is what should be visible as the external
    interface. So classes should only access things through the public properties
    of this class. If something is missing, that is a finding about the needed interface
    and this class should be updated rather than working around the issue by "knowing" how to
    get what we want from the internal variables.
    '''
    # TODO Add handling of writeOutput, writePlots, debug. I think we
    # can probably do that by just adding Observers
    def __init__(self, filename, vlidort_cli=None, writeOutput=False, writePlots=False,
                 **kwargs):
        logger.info(f"Strategy table filename {filename}")
        self._capture_directory = RefractorCaptureDirectory()
        self._observers = set()
        self._vlidort_cli = vlidort_cli
        self._table_step = -1

        self._retrieval_strategy_step_set  = copy.deepcopy(RetrievalStrategyStepSet.default_handle_set())
        self._cost_function_creator = CostFunctionCreator(rs=self)
        self._forward_model_handle_set = self._cost_function_creator.forward_model_handle_set
        self._observation_handle_set = self._cost_function_creator.observation_handle_set
        self._kwargs = kwargs
        self._kwargs["vlidort_cli"] = vlidort_cli
        
        self._state_info = StateInfo()
        self._state_element_handle_set = self._state_info.state_element_handle_set

        # Right now, we hardcode the output observers. Probably want to
        # rework this
        self.add_observer(RetrievalJacobianOutput())
        self.add_observer(RetrievalRadianceOutput())
        self.add_observer(RetrievalL2Output())
        # Similarly logic here is hardcoded
        if(writeOutput):
            # Depends on internal objects like strategy_table_dict. For now,
            # skip this
            #self.add_observer(RetrievalInputOutput())
            self.add_observer(RetrievalPickleResult())
            if(writePlots):
                self.add_observer(RetrievalPlotResult())
                self.add_observer(RetrievalPlotRadiance())
                
        # For calling from py-retrieve, it is useful to delay the filename. See
        # script_retrieval_ms below
        if(filename is not None):
            self.update_target(filename)

    def register_with_muses_py(self):
        '''Register run_ms as a replacement for script_retrieval_ms'''
        mpy.register_replacement_function("script_retrieval_ms", self)

    def should_replace_function(self, func_name, parms):
        return True

    def replace_function(self, func_name, parms):
        if(func_name == "script_retrieval_ms"):
            return self.script_retrieval_ms(**parms)

    def update_target(self, filename):
        '''Set up to process a target, given the filename for the strategy table.
        
        A number of objects related to this one might do caching based on the
        target, e.g., read the input files once. py-retrieve can call script_retrieval_ms
        multiple times with different targets, so we need to notify all the objects
        when this changes in case they need to clear out any caching.'''
        self._filename = os.path.abspath(filename)
        self._capture_directory.rundir = os.path.dirname(self.strategy_table_filename)
        self._strategy_executor = MusesStrategyExecutorOldStrategyTable(self.strategy_table_filename, self)
        self._strategy_table = self._strategy_executor.stable
        self._retrieval_config = RetrievalConfiguration.create_from_strategy_file(self.strategy_table_filename)
        self._measurement_id = MeasurementIdFile(f"{self.run_dir}/Measurement_ID.asc",
                                                self.retrieval_config,
                                                self._strategy_table.filter_list_all())
        self._cost_function_creator.update_target(self.measurement_id)
        self._retrieval_strategy_step_set.notify_update_target(self)
        self._state_info.notify_update_target(self)
        self.notify_update("update target")
        

    def script_retrieval_ms(self, filename, writeOutput=False, writePlots=False,
                            debug=False, update_product_format=False):
        # Ignore arguments other than filename.
        # We can clean this up if needed, perhaps delay the
        # initialization or something.
        self.update_target(filename)
        return self.retrieval_ms()
    
    def add_observer(self, obs):
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._observers.add(obs)
        if(hasattr(obs, "notify_add")):
            obs.notify_add(self)

    def remove_observer(self, obs):
        self._observers.discard(obs)
        if(hasattr(obs, "notify_remove")):
            obs.notify_remove(self)

    def clear_observers(self):
        # We change self._observers, in our loop so grab a copy of the list
        # before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)
        
    def notify_update(self, location, **kwargs):
        for obs in self._observers:
            obs.notify_update(self, location, **kwargs)

    def retrieval_ms(self):
        '''This is script_retrieval_ms in muses-py'''
        # Wrapper around calling mpy. We can perhaps pull some this out, but
        # for now we'll do that.
        with muses_py_call(self.run_dir,
                           vlidort_cli=self._vlidort_cli):
            self._strategy_executor.execute_retrieval()
            exitcode = 37
            logger.info(f"Done")
            logger.info('\n---')    
            logger.info(f"signaling successful completion w/ exit code {exitcode}")
            logger.info('\n---')    
            logger.info('\n---')    
            return exitcode

    def get_initial_guess(self):
        '''Set retrieval_info, errorInitial and errorCurrent for the current step.'''
        self._retrieval_info = RetrievalInfo(self._error_analysis, self._strategy_table,
                                             self._state_info)

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self._retrieval_info.n_totalParameters
        logger.info(f"Step: {self.table_step}, Total Parameters: {nparm}")

        if nparm > 0:
            xig = self._retrieval_info.initial_guess_list[0:nparm]
            self._state_info.update_state(self._retrieval_info, xig, [],
                                         self._cloud_prefs, self.table_step)

    @property
    def run_dir(self) -> str:
        '''Directory we are running in (e.g. where the strategy table and measurement id files
        are)'''
        return self._capture_directory.rundir

    @property
    def strategy_table_filename(self) -> str:
        '''Name of the strategy table we are using.'''
        return self._filename

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        '''Configuration parameters for the retrieval.'''
        return self._retrieval_config

    @property
    def measurement_id(self) -> MeasurementIdFile:
        '''Measurement ID for the current target.'''
        return self._measurement_id

    @property
    def forward_model_handle_set(self) -> ForwardModelHandleSet:
        '''The set of handles we use for mapping instrument name to a ForwardModel'''
        return self._forward_model_handle_set

    @property
    def observation_handle_set(self) -> ObservationHandleSet:
        '''The set of handles we use for mapping instrument name to a MusesObservation'''
        return self._observation_handle_set

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        '''The set of handles we use for each state element.'''
        return self._state_element_handle_set
    
    @property
    def _cloud_prefs(self):
        (_, fileID) = mpy.read_all_tes_cache(self._strategy_table.cloud_parameters_filename)
        cloudPrefs = fileID['preferences']
        return cloudPrefs
        
    @property
    def table_step(self) -> int:
        return self._strategy_table.table_step

    @table_step.setter
    def table_step(self, v : int):
        self._strategy_table.table_step = v

    @property
    def number_table_step(self) -> int:
        return self._strategy_table.number_table_step

    @property
    def step_name(self) -> str:
        return self._strategy_table.step_name

    @property
    def step_directory(self) -> str:
        return self._strategy_table.step_directory

    @property
    def retrieval_type(self) -> str:
        return self._retrieval_info.type.lower()

    @property
    def retrieval_info(self) -> RetrievalInfo:
        '''RetrievalInfo for current retrieval step. Note it might be good to remove this
        if possible, right now this is just used by RetrievalL2Output. But at least for now
        we need this to get the required information for the output.'''
        return self._retrieval_info

    def retrieval_elements(self, step_num: int) -> 'list(str)':
        # This is just used in RetrievalL2Output to figure out the output name.
        # We can perhaps clean up this interface
        return self._strategy_table.table_entry('retrievalElements', step_num).split(",")

    @property
    def state_info(self):
        # Can hopefully replace this with CurrentState
        return self._state_info

    def save_pickle(self, save_pickle_file, **kwargs):
        '''Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy.'''
        self._capture_directory.save_directory(self.run_dir, vlidort_input=None)
        pickle.dump([self, kwargs], open(save_pickle_file, "wb"))

    @classmethod
    def load_retrieval_strategy(cls, save_pickle_file, path=".",
                                change_to_dir = False,
                                osp_dir=None, gmao_dir=None,
                                vlidort_cli=None):
        '''This pairs with save_pickle.'''
        res, kwargs = pickle.load(open(save_pickle_file, "rb"))
        res._capture_directory.rundir = f"{os.path.abspath(path)}/{res._capture_directory.runbase}"
        res._strategy_table.filename = f"{res.run_dir}/{os.path.basename(res._strategy_table.filename)}"
        res._filename = res._strategy_table.filename
        res._retrieval_config.osp_dir = osp_dir
        res._retrieval_config.base_dir = res.run_dir
        res._capture_directory.extract_directory(path=path,
                              change_to_dir=change_to_dir, osp_dir=osp_dir,
                              gmao_dir=gmao_dir)
        if(vlidort_cli is not None):
            res._vlidort_cli = vlidort_cli
        return res, kwargs

class RetrievalStrategyCaptureObserver:
    '''Helper class, pickles RetrievalStrategy at each time notify_update is
    called. Intended for unit tests and other kinds of debugging.'''
    def __init__(self, basefname, location_to_capture):
        self.basefname = basefname
        self.location_to_capture = location_to_capture

    def notify_update(self, retrieval_strategy, location, **kwargs):
        if(location != self.location_to_capture):
            return
        fname = f"{self.basefname}_{retrieval_strategy.table_step}.pkl"
        retrieval_strategy.save_pickle(fname, **kwargs)

class RetrievalStrategyMemoryUse:
    def __init__(self):
        self.tr = None

    def notify_update(self, retrieval_strategy, location, **kwargs):
        # Need pympler here, but don't generally need it. Include this
        # so this isn't a requirement, unless we are running with this
        # observer
        from pympler import tracker
        if(location == "starting retrieval steps"):
            self.tr = tracker.SummaryTracker()
        elif(location in ("done copy_current_initial",
                          "done get_initial_guess",
                          "done create_windows",
                          "done retrieval_step",
                          "done next_state_to_current")):
            logger.info(f"Memory change when {location}")
            self.tr.print_diff()
            
        
__all__ = ["RetrievalStrategy", "RetrievalStrategyCaptureObserver",
           "RetrievalStrategyMemoryUse"]    

    
