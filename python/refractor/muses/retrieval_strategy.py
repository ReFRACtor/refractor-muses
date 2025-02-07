from __future__ import annotations
from .refractor_capture_directory import RefractorCaptureDirectory
from .retrieval_l2_output import RetrievalL2Output
from .retrieval_irk_output import RetrievalIrkOutput
from .retrieval_radiance_output import RetrievalRadianceOutput
from .retrieval_jacobian_output import RetrievalJacobianOutput
from .retrieval_debug_output import (
    RetrievalPickleResult,
    RetrievalPlotRadiance,
    RetrievalPlotResult,
)
from .retrieval_strategy_step import (
    RetrievalStrategyStepSet,
    RetrievalStepCaptureObserver,
)
from .retrieval_configuration import RetrievalConfiguration
from .muses_observation import MeasurementIdFile
from .muses_strategy_executor import MusesStrategyExecutorOldStrategyTable
from .spectral_window_handle import SpectralWindowHandleSet
from .qa_data_handle import QaDataHandleSet
from .state_info import StateInfo
from .cost_function_creator import CostFunctionCreator
from loguru import logger
import refractor.muses.muses_py as mpy  # type: ignore
import os
import copy
import pickle
from pathlib import Path
from typing import Any
import typing

if typing.TYPE_CHECKING:
    from .error_analysis import ErrorAnalysis
    from .forward_model_handle import ForwardModelHandleSet
    from .observation_handle import ObservationHandleSet
    from .state_info import StateElementHandleSet
    from .retrieval_info import RetrievalInfo
    from .current_state import CurrentState
    from .muses_strategy_executor import CurrentStrategyStep
    from .cost_function import CostFunction
    from .muses_strategy import MusesStrategy
    from .identifier import RetrievalType, InstrumentIdentifier


# We could make this an rf.Observable, but no real reason to push this to a C++
# level. So we just have a simple observation set here
class RetrievalStrategy(mpy.ReplaceFunctionObject):
    """This is a replacement for script_retrieval_ms, that tries to do
    a few things:

    1. Simplifies the core code, the script_retrieval_ms is really
       pretty long and is a sequence of "do one thing, then another,
       then another".

    2. Moving output out of this class, and having separate classes
       handle this. We use the standard ReFRACtor approach of having
       observers. This tend to give a much cleaner interface with
       clear separation.

    3. Adopt a extensively, configurable way to change behavior
       through configuration rather than software change. See
       CreatorHandle.

    4. Handle species information as a separate class, which allows us
       to easily extend the list of jacobian parameters (e.g, add
       EOFs). The existing code uses long lists of hardcoded values,
       this attempts to be a more adaptable.

    This has a number of advantages, for example having initial guess
    separated out (through CreatorHandle) allows us to do unit testing
    in ways that don't require updating the OSP directories with new
    covariance stuff, for example.

    Note that there is a lot of overlap between this class and the
    MusesStrategyExecutor class. It isn't clear that long term there will
    actually be two separate classes. However for right now this is a
    useful division of responsibilities:

    1. RetrievalStrategy worries about the interface with external
       classes.  What does this look like to other classes? What is
       exposed to the output classes? How does configuration modify
       things?

    2. MusesStrategyExecutor worries about actually running the
       strategy. How do we determine the retrieval steps? How do we
       run the retrieval steps?

    This may well merge once we have the external interface sorted out.

    Note that is class has a number of internal variables, with the
    normal python "private" suggestion of using a leading "_",
    e.g. "_capture_directory".  It is a normal python convention that
    external classes not use this private variables. But this should
    be even stronger for this class - one of the primary things we are
    trying to figure out is what should be visible as the external
    interface. So classes should only access things through the public
    properties of this class. If something is missing, that is a
    finding about the needed interface and this class should be
    updated rather than working around the issue by "knowing" how to
    get what we want from the internal variables.

    """

    # TODO Add handling of writeOutput, writePlots, debug. I think we
    # can probably do that by just adding Observers
    def __init__(
        self,
        filename: str | os.PathLike[str],
        vlidort_cli: str | os.PathLike[str] | None = None,
        writeOutput=False,
        writePlots=False,
        osp_dir: str | os.PathLike[str] | None = None,
        **kwargs,
    ):
        logger.info(f"Strategy table filename {filename}")
        self._capture_directory = RefractorCaptureDirectory()
        self._observers: set[Any] = set()
        self._vlidort_cli = Path(vlidort_cli) if vlidort_cli is not None else None

        self._retrieval_strategy_step_set = copy.deepcopy(
            RetrievalStrategyStepSet.default_handle_set()
        )
        self._spectral_window_handle_set = copy.deepcopy(
            SpectralWindowHandleSet.default_handle_set()
        )
        self._qa_data_handle_set = copy.deepcopy(QaDataHandleSet.default_handle_set())
        self._cost_function_creator = CostFunctionCreator(rs=self)
        self._forward_model_handle_set = (
            self._cost_function_creator.forward_model_handle_set
        )
        self._observation_handle_set = (
            self._cost_function_creator.observation_handle_set
        )
        self._kwargs = kwargs
        self._kwargs["vlidort_cli"] = self._vlidort_cli

        self._state_info = StateInfo()
        self._state_element_handle_set = self._state_info.state_element_handle_set
        self.osp_dir = osp_dir
        if self.osp_dir is None:
            self.osp_dir = os.environ.get("MUSES_OSP_PATH", None)

        # Right now, we hardcode the output observers. Probably want to
        # rework this
        self.add_observer(RetrievalJacobianOutput())
        self.add_observer(RetrievalRadianceOutput())
        self.add_observer(RetrievalL2Output())
        self.add_observer(RetrievalIrkOutput())
        # Similarly logic here is hardcoded.
        # JLL: some MUSES diagnostics (esp. the solver steps in the levmar code)
        # aren't observers yet, until they are, I need this boolean to turn them
        # on.
        self.write_output = writeOutput
        if writeOutput:
            # Depends on internal objects like strategy_table_dict. For now,
            # skip this
            # self.add_observer(RetrievalInputOutput())
            self.add_observer(RetrievalPickleResult())
            if writePlots:
                self.add_observer(RetrievalPlotResult())
                self.add_observer(RetrievalPlotRadiance())

        # For calling from py-retrieve, it is useful to delay the filename. See
        # script_retrieval_ms below
        if filename is not None:
            self.update_target(filename)

    def register_with_muses_py(self):
        """Register run_ms as a replacement for script_retrieval_ms"""
        mpy.register_replacement_function("script_retrieval_ms", self)

    def should_replace_function(self, func_name: str, parms) -> bool:
        return True

    def replace_function(self, func_name: str, parms):
        if func_name == "script_retrieval_ms":
            return self.script_retrieval_ms(**parms)

    def update_target(self, filename: str | os.PathLike[str]):
        """Set up to process a target, given the filename for the
        strategy table.

        A number of objects related to this one might do caching based
        on the target, e.g., read the input files once. py-retrieve
        can call script_retrieval_ms multiple times with different
        targets, so we need to notify all the objects when this
        changes in case they need to clear out any caching.

        """
        if False:
            # Clear any caching of files muses-py did.  Don't exactly
            # understand these caches, but this causes an error with
            # threading. So just skip, I don't think we actually need
            # this
            mpy.clear_cache()
        filename = Path(filename)
        self._filename = filename.absolute()
        self._capture_directory.rundir = self.strategy_table_filename.parent
        self._retrieval_config = RetrievalConfiguration.create_from_strategy_file(
            self.strategy_table_filename, osp_dir=self.osp_dir
        )
        self._strategy_executor = MusesStrategyExecutorOldStrategyTable(
            self.strategy_table_filename,
            self,
            retrieval_strategy_step_set=self._retrieval_strategy_step_set,
            spectral_window_handle_set=self._spectral_window_handle_set,
            qa_data_handle_set=self._qa_data_handle_set,
        )
        self._measurement_id = MeasurementIdFile(
            self.run_dir / "Measurement_ID.asc",
            self.retrieval_config,
            self.strategy_executor.filter_list_dict,
        )
        self._cost_function_creator.notify_update_target(self.measurement_id)
        self.strategy_executor.spectral_window_handle_set.notify_update_target(
            self.measurement_id
        )
        self.strategy_executor.qa_data_handle_set.notify_update_target(
            self.measurement_id
        )
        self._retrieval_strategy_step_set.notify_update_target(self)
        self._state_info.notify_update_target(self)
        self.notify_update("update target")

    def script_retrieval_ms(
        self,
        filename: str | os.PathLike[str],
        writeOutput=False,
        writePlots=False,
        debug=False,
        update_product_format=False,
    ):
        # Ignore arguments other than filename.
        # We can clean this up if needed, perhaps delay the
        # initialization or something.
        self.update_target(filename)
        return self.retrieval_ms()

    def add_observer(self, obs: Any):
        # Often we want weakref, so we don't prevent objects from
        # being deleted just because they are observing this. But in
        # this particular case, we actually do want to maintain the
        # lifetime. These observers will do things like write out
        # output, but have no real life outside of being attached to
        # this class.  It is easy enough to change this to weakref if
        # that proves useful
        self._observers.add(obs)
        if hasattr(obs, "notify_add"):
            obs.notify_add(self)

    def remove_observer(self, obs: Any):
        self._observers.discard(obs)
        if hasattr(obs, "notify_remove"):
            obs.notify_remove(self)

    def clear_observers(self):
        # We change self._observers, in our loop so grab a copy of the
        # list before we start
        lobs = list(self._observers)
        for obs in lobs:
            self.remove_observer(obs)

    def notify_update(self, location: str, **kwargs):
        for obs in self._observers:
            obs.notify_update(self, location, **kwargs)

    @property
    def vlidort_cli(self) -> Path | None:
        return self._vlidort_cli

    @vlidort_cli.setter
    def vlidort_cli(self, v: str | os.PathLike[str]):
        self._vlidort_cli = Path(v)
        self._kwargs["vlidort_cli"] = Path(v)

    @property
    def keyword_arguments(self) -> dict:
        """Keyword arguments, which can be used to pass arguments down
        to lower level classes (e.g., options for the forward model
        like use_lrad)

        """
        return self._kwargs

    def retrieval_ms(self) -> int:
        """This is script_retrieval_ms in muses-py"""
        self.strategy_executor.execute_retrieval()
        exitcode = 37
        logger.info("Done")
        logger.info("\n---")
        logger.info(f"signaling successful completion w/ exit code {exitcode}")
        logger.info("\n---")
        logger.info("\n---")
        return exitcode

    def continue_retrieval(self, stop_after_step: int | None = None) -> None:
        """After saving a pickled step, you can continue the
        processing starting at that step to diagnose a problem.

        """
        self.strategy_executor.continue_retrieval(stop_after_step=stop_after_step)

    @property
    def strategy_executor(self) -> MusesStrategyExecutorOldStrategyTable:
        """The MusesStrategyExecutor used to run through the strategy"""
        return self._strategy_executor

    @property
    def strategy(self) -> MusesStrategy:
        """The MusesStrategy used to describe the strategy"""
        return self.strategy_executor.strategy

    @property
    def run_dir(self) -> Path:
        """Directory we are running in (e.g. where the strategy table and measurement id files
        are)"""
        return Path(self._capture_directory.rundir)

    @property
    def strategy_table_filename(self) -> Path:
        """Name of the strategy table we are using."""
        return self._filename

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        """Configuration parameters for the retrieval."""
        return self._retrieval_config

    @property
    def measurement_id(self) -> MeasurementIdFile:
        """Measurement ID for the current target."""
        return self._measurement_id

    @property
    def forward_model_handle_set(self) -> ForwardModelHandleSet:
        """The set of handles we use for mapping instrument name to a
        ForwardModel"""
        return self._forward_model_handle_set

    @property
    def observation_handle_set(self) -> ObservationHandleSet:
        """The set of handles we use for mapping instrument name to a
        MusesObservation"""
        return self._observation_handle_set

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        """The set of handles we use for each state element."""
        return self._state_element_handle_set

    @property
    def retrieval_strategy_step_set(self) -> RetrievalStrategyStepSet:
        """The set of handles for determining the RetrievalStrategyStep."""
        return self._retrieval_strategy_step_set

    @property
    def spectral_window_handle_set(self) -> SpectralWindowHandleSet:
        """The set of handles for determining the MusesSpectralWindow."""
        return self._spectral_window_handle_set

    @property
    def qa_data_handle_set(self) -> QaDataHandleSet:
        """The set of handles for determining the QA flag file name."""
        return self._qa_data_handle_set

    @property
    def step_number(self) -> int:
        return self.current_strategy_step.step_number

    @property
    def instrument_name_all_step(self) -> list[InstrumentIdentifier]:
        return self.strategy_executor.instrument_name_all_step

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        return self.strategy_executor.current_strategy_step

    def current_state(
        self, do_systematic=False, jacobian_speciesIn=None
    ) -> CurrentState:
        return self.strategy_executor.current_state(
            do_systematic=do_systematic, jacobian_speciesIn=jacobian_speciesIn
        )

    def number_steps_left(self, retrieval_element_name: str) -> int:
        """This returns the number of retrieval steps left that
        contain a given retrieval element name. This is an odd seeming
        function, but is used by RetrievalL2Output to name files. So
        for example we have Products_L2-O3-0.nc for the last step that
        retrieves O3, Products_L2-O3-1.nc for the previous step
        retrieving O3, etc.

        """
        return self.strategy_executor.number_steps_left(retrieval_element_name)

    @property
    def step_name(self) -> str:
        return self.current_strategy_step.step_name

    @property
    def retrieval_type(self) -> RetrievalType:
        return self.current_strategy_step.retrieval_type

    @property
    def retrieval_info(self) -> RetrievalInfo:
        """RetrievalInfo for current retrieval step. Note it might be
        good to remove this if possible, right now this is just used
        by RetrievalL2Output. But at least for now we need this to get
        the required information for the output.

        """
        return self.current_strategy_step.retrieval_info

    @property
    def state_info(self) -> StateInfo:
        # Can hopefully replace this with CurrentState
        return self._state_info

    @property
    def cost_function_creator(self) -> CostFunctionCreator:
        return self._cost_function_creator

    @property
    def error_analysis(self) -> ErrorAnalysis:
        """Error analysis"""
        # TODO Not really clear what the coupling should be here. But
        # for now, this is used by RetrievalStrategyStep. Perhaps we
        # can just pass this in the constructor? Perhaps this can be
        # handled like our QaDataHandleSet, where we have
        # configuration to select this? Isn't clear that we would ever
        # want this replaced with a different kind of
        # ErrorAnalysis. For now, just make it clear that we have this
        # coupling and we can figure out how this should be handled.
        return self.strategy_executor.error_analysis

    def create_cost_function(
        self,
        do_systematic=False,
        include_bad_sample=False,
        fix_apriori_size=False,
        jacobian_speciesIn=None,
    ) -> CostFunction:
        """Create cost function"""
        # Similiar to error_analysis, this gets uses in
        # RetrievalStrategyStep and perhaps we should just pass the
        # strategy_executor to the constructor.  But for now, make
        # explicit that we need this.
        return self.strategy_executor.create_cost_function(
            do_systematic=do_systematic,
            include_bad_sample=include_bad_sample,
            fix_apriori_size=fix_apriori_size,
            jacobian_speciesIn=jacobian_speciesIn,
        )

    def save_pickle(self, save_pickle_file: str | os.PathLike[str], **kwargs):
        """Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy."""
        self._capture_directory.save_directory(self.run_dir, vlidort_input=None)
        pickle.dump([self, kwargs], open(save_pickle_file, "wb"))

    def state_state_info(self, save_pickle_file: str | os.PathLike[str]):
        """Dump a pickled version of the StateInfo object.
        We may play with this, but currently we gzip this.
        We would like to have something a bit more stable, not tied to the
        object structure. But for now, just do a straight json dump of the object"""

        # Doesn't work yet. Not overly important, looks like a bug in
        # jsonpickle Just use normal pickle for now, we want to change
        # what gets saved anyways
        #
        # with gzip.GzipFile(save_pickle_file, "wb") as fh:
        #    fh.write(jsonpickle.encode(self.state_info).encode('utf-8'))
        pickle.dump(self.state_info, open(save_pickle_file, "wb"))

    def load_state_info(
        self,
        state_info_pickle_file: str | os.PathLike[str],
        step_number: int,
        ret_state_file: str | os.PathLike[str] | None = None,
    ):
        """This pairs with state_state_info. Instead of pickling the
        entire RetrievalStrategy, we just save the state. We then
        set up to process the given target_filename with the given
        state, jumping to the given retrieval step_number.

        Note for some tests in addition to the StateInfo we want the
        results saved by RetrievalStepCaptureObserver (e.g., we want
        to test the output writing). You can optionally pass in the
        json file for this and we will also pass that information to
        the RetrievalStrategyStep.

        """
        # self._state_info = jsonpickle.decode(
        #    gzip.open(state_info_pickle_file, "rb").read())
        self._state_info = pickle.load(open(state_info_pickle_file, "rb"))
        self._state_info.retrieval_config.base_dir = self.run_dir
        self._state_info.retrieval_config.osp_dir = (
            Path(self.osp_dir) if self.osp_dir is not None else None
        )
        self.strategy_executor.state_info = self._state_info
        self.strategy_executor.set_step(step_number)
        if ret_state_file is not None:
            t = RetrievalStepCaptureObserver.load_retrieval_state(ret_state_file)
            self._kwargs["ret_state"] = t
            self.strategy_executor.kwargs["ret_state"] = t

    @classmethod
    def load_retrieval_strategy(
        cls,
        save_pickle_file: str | os.PathLike[str],
        path: str | os.PathLike[str] = ".",
        change_to_dir=False,
        osp_dir: str | os.PathLike[str] | None = None,
        gmao_dir: str | os.PathLike[str] | None = None,
        vlidort_cli: str | os.PathLike[str] | None = None,
    ):
        """This pairs with save_pickle.

        This is pretty direct to use, but as an example we can do
        something like:

        osp_dir = os.environ["MUSES_OSP_PATH"]
        gmao_dir = os.environ["MUSES_GMAO_PATH"]
        subprocess.run("rm -r ./try_it", shell=True)
        dir_in = "./retrieval_strategy_cris_tropomi/20190807_065_04_08_5"
        step_number=10
        rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
            f"{dir_in}/retrieval_step_{step_number}.pkl",
            path="./try_it",
            osp_dir=osp_dir, gmao_dir=gmao_dir,change_to_dir=True)
        rs.continue_retrieval()

        """
        res, kwargs = pickle.load(open(save_pickle_file, "rb"))
        res._capture_directory.rundir = (
            Path(path).absolute() / res._capture_directory.runbase
        )
        res._filename = res.run_dir / res.strategy_table_filename.name
        res._strategy_executor.strategy_table_filename = res._filename
        res._retrieval_config.osp_dir = osp_dir
        res._retrieval_config.base_dir = res.run_dir
        res._capture_directory.extract_directory(
            path=path, change_to_dir=change_to_dir, osp_dir=osp_dir, gmao_dir=gmao_dir
        )
        if vlidort_cli is not None:
            res.vlidort_cli = vlidort_cli
            kwargs["vlidort_cli"] = vlidort_cli
        return res, kwargs


class RetrievalStrategyCaptureObserver:
    """Helper class, pickles RetrievalStrategy at each time
    notify_update is called. Intended for unit tests and other kinds
    of debugging.

    """

    def __init__(self, basefname: str, location_to_capture: str):
        self.basefname = basefname
        self.location_to_capture = location_to_capture

    def notify_update(
        self, retrieval_strategy: RetrievalStrategy, location: str, **kwargs
    ):
        if location != self.location_to_capture:
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        fname = f"{self.basefname}_{retrieval_strategy.step_number}.pkl"
        # Don't want this class included in the pickle
        retrieval_strategy.remove_observer(self)
        retrieval_strategy.save_pickle(fname, **kwargs)
        retrieval_strategy.add_observer(self)


class StateInfoCaptureObserver:
    """Helper class, pickles RetrievalStrategy.state_info at each time
    notify_update is called. Intended for unit tests and other kinds
    of debugging.

    """

    def __init__(self, basefname: str, location_to_capture: str):
        self.basefname = basefname
        self.location_to_capture = location_to_capture

    def notify_update(
        self, retrieval_strategy: RetrievalStrategy, location: str, **kwargs
    ):
        if location != self.location_to_capture:
            return
        logger.debug(f"Call to {self.__class__.__name__}::notify_update")
        # fname = f"{self.basefname}_{retrieval_strategy.step_number}.json.gz"
        fname = f"{self.basefname}_{retrieval_strategy.step_number}.pkl"
        retrieval_strategy.state_state_info(fname)


class RetrievalStrategyMemoryUse:
    def __init__(self):
        self.tr = None

    def notify_update(
        self, retrieval_strategy: RetrievalStrategy, location: str, **kwargs
    ):
        # Need pympler here, but don't generally need it. Include this
        # so this isn't a requirement, unless we are running with this
        # observer
        from pympler import tracker

        if location == "starting retrieval steps":
            self.tr = tracker.SummaryTracker()
        elif location in (
            "done copy_current_initial",
            "done get_initial_guess",
            "done create_windows",
            "done retrieval_step",
            "done next_state_to_current",
        ):
            logger.info(f"Memory change when {location}")
            self.tr.print_diff()


__all__ = [
    "RetrievalStrategy",
    "RetrievalStrategyCaptureObserver",
    "RetrievalStrategyMemoryUse",
    "StateInfoCaptureObserver",
]
