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
from .replace_function_helper import (suppress_replacement,
                                      register_replacement_function_in_block)
from .error_analysis import ErrorAnalysis
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
    '''This is an attempt to make the muses-py script_retrieval_ms
    more like our JointRetrieval stuff (pretty dated, but
    https://github.jpl.nasa.gov/refractor/joint_retrieval)

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

    '''
    # TODO Add handling of writeOutput, writePlots, debug. I think we
    # can probably do that by just adding Observers
    def __init__(self, filename, vlidort_cli=None, writeOutput=False, writePlots=False,
                 **kwargs):
        logger.info(f"Strategy table filename {filename}")
        self.capture_directory = RefractorCaptureDirectory()
        self._observers = set()
        self.vlidort_cli = vlidort_cli
        self._table_step = -1

        self.retrieval_strategy_step_set  = copy.deepcopy(RetrievalStrategyStepSet.default_handle_set())
        self.cost_function_creator = CostFunctionCreator(rs=self)
        self.forward_model_handle_set = self.cost_function_creator.forward_model_handle_set
        self.observation_handle_set = self.cost_function_creator.observation_handle_set
        self.kwargs = kwargs
        self.kwargs["vlidort_cli"] = vlidort_cli

        self.state_info = StateInfo()
        self.state_element_handle_set = self.state_info.state_element_handle_set

        # Right now, we hardcode the output observers. Probably want to
        # rework this
        self.add_observer(RetrievalJacobianOutput())
        self.add_observer(RetrievalRadianceOutput())
        self.add_observer(RetrievalL2Output())
        # Similarly logic here is hardcoded
        if(writeOutput):
            self.add_observer(RetrievalInputOutput())
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
        self.filename = os.path.abspath(filename)
        self.run_dir = os.path.dirname(self.filename)
        self.strategy_table = StrategyTable(self.filename)
        self.retrieval_config = RetrievalConfiguration.create_from_strategy_file(self.filename)
        self.measurement_id = MeasurementIdFile(f"{self.run_dir}/Measurement_ID.asc",
                                                self.retrieval_config,
                                                self.strategy_table.filter_list_all())
        self.cost_function_creator.update_target(self.measurement_id)
        self.retrieval_strategy_step_set.notify_update_target(self)
        self.state_info.notify_update_target(self)
        self.notify_update("update target")
        

    def script_retrieval_ms(self, filename, writeOutput=False, writePlots=False,
                            debug=False, update_product_format=False):
        # Ignore arguments other than filename.
        # We can clean this up if needed, perhaps delay the
        # initialization or something.
        self.update_target(filename)
        return self.retrieval_ms()
    
    @property
    def run_dir(self):
        return self.capture_directory.rundir

    @run_dir.setter
    def run_dir(self, v):
        self.capture_directory.rundir = v
        
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
                           vlidort_cli=self.vlidort_cli):
            return self.retrieval_ms_body()

    def retrieval_ms_body(self):
        start_date = time.strftime("%c")
        start_time = time.time()

        self.instrument_name_all = self.strategy_table.instrument_name(all_step=True)
        self.state_info.init_state(self.strategy_table,
                                   self.cost_function_creator.observation_handle_set,
                                   self.instrument_name_all, self.run_dir)

        self.error_analysis = ErrorAnalysis(self.strategy_table, self.state_info)
        self.retrieval_info = None
        self.notify_update("initial set up done")
        
        # Go through all the steps once, to make sure we can get all the information
        # we need. This way we fail up front, rather than after multiple retrieval
        # steps
        for stp in range(self.number_table_step):
            self.table_step = stp
            self.get_initial_guess()
        self.state_info.copy_current_initialInitial()
        self.notify_update("starting retrieval steps")
        # Now go back through and actually do retrievals.
        # Note that a BT step might change the number of steps we have, it
        # modifies the strategy table. So we can't use a normal for
        # loop here, we need to recalculate self.number_table_step each time.
        # So we use a while loop
        stp = -1
        while stp < self.number_table_step - 1:
            stp += 1
            self.retrieval_ms_body_step(stp)
            
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
        logger.info(f"Done")
        
        exitcode = 37
        logger.info('\n---')    
        logger.info(f"signaling successful completion w/ exit code {exitcode}")
        logger.info('\n---')    
        logger.info('\n---')    
        return exitcode

    def retrieval_ms_body_step(self, stp):
        '''This is the body of the step loop in retrieval_ms_body. We pull this
        out as a separate function because it is nice to be able to call this
        in isolation when debugging - e.g., use RetrievalStrategyCaptureObserver to
        capture each step, then load it back and rerun the step in a debugging
        session.'''
        self.table_step = stp
        self.notify_update("start retrieval_ms_body_step")
        self.state_info.copy_current_initial()
        self.notify_update("done copy_current_initial")
        logger.info(f'\n---')
        logger.info(f"Step: {self.table_step}, Step Name: {self.step_name}, Total Steps: {self.number_table_step}")
        logger.info(f'\n---')
        self.instruments = self.strategy_table.instrument_name()
        self.get_initial_guess()
        self.notify_update("done get_initial_guess")
        logger.info(f"Step: {self.table_step}, Retrieval Type {self.retrieval_type}")
        self.retrieval_strategy_step_set.retrieval_step(self.retrieval_type, self)
        self.notify_update("done retrieval_step")
        self.state_info.next_state_to_current()
        self.notify_update("done next_state_to_current")
        logger.info(f"Done with step {self.table_step}")
        
    def get_initial_guess(self):
        '''Set retrieval_info, errorInitial and errorCurrent for the current step.'''
        self.retrieval_info = RetrievalInfo(self.error_analysis, self.strategy_table,
                                            self.state_info)

        # Update state with initial guess so that the initial guess is
        # mapped properly, if doing a retrieval, for each retrieval step.
        nparm = self.retrieval_info.n_totalParameters
        logger.info(f"Step: {self.table_step}, Total Parameters: {nparm}")

        if nparm > 0:
            xig = self.retrieval_info.initialGuessList[0:nparm]
            self.state_info.update_state(self.retrieval_info, xig, [],
                                         self.cloud_prefs, self.table_step)

    @property
    def cloud_prefs(self):
        (_, fileID) = mpy.read_all_tes_cache(self.strategy_table.cloud_parameters_filename)
        cloudPrefs = fileID['preferences']
        return cloudPrefs
        
    @property
    def table_step(self):
        return self.strategy_table.table_step

    @table_step.setter
    def table_step(self, v):
        self.strategy_table.table_step = v

    @property
    def number_table_step(self):
        return self.strategy_table.number_table_step

    @property
    def step_name(self):
        return self.strategy_table.step_name

    @property
    def retrieval_type(self):
        return self.retrieval_info.type.lower()

    @property
    def output_directory(self):
        return self.strategy_table.output_directory

    def save_pickle(self, save_pickle_file, **kwargs):
        '''Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy.'''
        self.capture_directory.save_directory(self.run_dir, vlidort_input=None)
        pickle.dump([self, kwargs], open(save_pickle_file, "wb"))

    @classmethod
    def load_retrieval_strategy(cls, save_pickle_file, path=".",
                                change_to_dir = False,
                                osp_dir=None, gmao_dir=None,
                                vlidort_cli=None):
        '''This pairs with save_pickle.'''
        res, kwargs = pickle.load(open(save_pickle_file, "rb"))
        res.run_dir = f"{os.path.abspath(path)}/{res.capture_directory.runbase}"
        res.strategy_table.filename = f"{res.run_dir}/{os.path.basename(res.strategy_table.filename)}"
        res.capture_directory.extract_directory(path=path,
                              change_to_dir=change_to_dir, osp_dir=osp_dir,
                              gmao_dir=gmao_dir)
        if(vlidort_cli is not None):
            res.vlidort_cli = vlidort_cli
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

    
