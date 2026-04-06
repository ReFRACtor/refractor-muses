from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .refractor_capture_directory import RefractorCaptureDirectory
from .retrieval_l2_output import RetrievalL2Output
from .retrieval_irk_output import RetrievalIrkOutput
from .retrieval_radiance_output import RetrievalRadianceOutput
from .retrieval_jacobian_output import RetrievalJacobianOutput
from .muses_strategy_executor import MusesStrategyExecutor
from .retrieval_debug_output import (
    RetrievalPickleResult,
    RetrievalPlotRadiance,
    RetrievalPlotResult,
)
from .retrieval_configuration import RetrievalConfiguration
from .process_location_observable import ProcessLocationObservable
from .muses_strategy_executor import MusesStrategyExecutorMusesStrategy
from .muses_strategy_context import MusesStrategyContext, MusesStrategyContextMixin
from .muses_levmar_solver import (
    VerboseSolverLogging,
    SolverLogFileWriter,
)
from .state_element import StateElement, StateElementHandleSet
from .cross_state_element import CrossStateElement, CrossStateElementHandleSet
from .creator_dict import CreatorDict
from .identifier import ProcessLocation
from .input_file_helper import InputFileHelper
from loguru import logger
import os
import pickle
from pathlib import Path
from typing import Any
import typing
import pystac

if typing.TYPE_CHECKING:
    from .forward_model_handle import ForwardModelHandleSet
    from .observation_handle import ObservationHandleSet
    from .current_state import CurrentState


# We could make this an rf.Observable, but no real reason to push this to a C++
# level. So we just have a simple observation set here
# This implements mpy.ReplaceFunctionObject, but we don't actually derive from
# that so we don't depend on mpy being available.
# class RetrievalStrategy(mpy.ReplaceFunctionObject):
class RetrievalStrategy(MusesStrategyContextMixin):
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
        filename: str | os.PathLike[str] | None,
        writeOutput: bool = False,
        writePlots: bool = False,
        ifile_hlp: InputFileHelper | None = None,
        use_stac: bool = False,
        **kwargs: Any,
    ) -> None:
        MusesStrategyContextMixin.__init__(self, MusesStrategyContext())
        self._capture_directory = RefractorCaptureDirectory()
        self._observers: set[Any] = set()
        self._creator_dict = CreatorDict(self.strategy_context)
        self._process_location_observable = ProcessLocationObservable()

        self._kwargs: dict[str, Any] = kwargs

        self._ifile_hlp = ifile_hlp if ifile_hlp is not None else InputFileHelper()
        self._strategy_executor = MusesStrategyExecutorMusesStrategy(
            self.creator_dict, self.process_location_observable
        )

        # Right now, we hardcode the output observers. Probably want to
        # rework this
        self.add_observer(RetrievalJacobianOutput(self.creator_dict))
        self.add_observer(RetrievalRadianceOutput(self.creator_dict))
        self.add_observer(RetrievalL2Output(self.creator_dict))
        self.add_observer(RetrievalIrkOutput(self.creator_dict))
        # Assume we always want verbose logging in solver
        self.add_observer(VerboseSolverLogging())
        if writeOutput:
            levmar_log_file = f"{self.retrieval_config['output_directory']}/Step{self.step_number:02d}_{self.step_name}/LevmarSolver-{self.step_name}.log"
            self.add_observer(SolverLogFileWriter(levmar_log_file))
            # Depends on internal objects like strategy_table_dict. For now,
            # skip this
            # self.add_observer(RetrievalInputOutput())
            self.add_observer(RetrievalPickleResult(self.creator_dict))
            if writePlots:
                self.add_observer(RetrievalPlotResult(self.creator_dict))
                self.add_observer(RetrievalPlotRadiance(self.creator_dict))

        # For calling from py-retrieve, it is useful to delay the filename. See
        # script_retrieval_ms below
        if filename is not None:
            if use_stac:
                self.update_stac(filename)
            else:
                self.update_target(filename)

    def register_with_muses_py(self) -> None:
        """Register run_ms as a replacement for script_retrieval_ms.

        This is done so that py-retrieve top level executable can turn
        around and use refractor to actually do the retrieval. For
        now, this is a useful functionality, but I'm not sure how long
        this will still be useful (since we have a refractor-retrieve
        top level executable.

        We can remove this when no longer useful.
        """
        from refractor.old_py_retrieve_wrapper import (
            muses_py_register_replacement_function,
        )

        muses_py_register_replacement_function("script_retrieval_ms", self)

    def should_replace_function(self, func_name: str, parms: list[Any]) -> bool:
        return True

    def replace_function(self, func_name: str, parms: dict) -> int | None:
        if func_name == "script_retrieval_ms":
            return self.script_retrieval_ms(**parms)
        return None

    @property
    def creator_dict(self) -> CreatorDict:
        return self._creator_dict

    def update_stac(self, filename: str | os.PathLike[str]) -> None:
        """Not exactly clear how we want to handle a stac file vs
        a single target. For now, we just keep this code separate.
        As we get a bit more experience, we may be able to merge with
        a single target - at some level these are the same things
        "what is needed to run processing", but we don't know that
        that interface should look like."""
        logger.info(f"Strategy table filename: {filename}")

        filename = Path(filename)
        self._filename = filename.absolute()
        self._capture_directory.rundir = filename.absolute().parent
        rconf = RetrievalConfiguration.create_from_strategy_file(
            self.strategy_table_filename,
            self._ifile_hlp,
        )

        stac = pystac.Catalog.from_file(
            self._output_directory / "config.json",
        )
        self.strategy_context.update_strategy_context(
            stac_catalog=stac,
            retrieval_config=rconf,
            strategy_table_filename=self.strategy_table_filename,
            creator_dict=self.creator_dict,
        )
        self.notify_process_location(ProcessLocation("update stac"))

    def update_target(self, filename: str | os.PathLike[str]) -> None:
        """Set up to process a target, given the filename for the
        strategy table.

        A number of objects related to this one might do caching based
        on the target, e.g., read the input files once. py-retrieve
        can call script_retrieval_ms multiple times with different
        targets, so we need to notify all the objects when this
        changes in case they need to clear out any caching.

        """

        logger.info(f"Strategy table filename: {filename}")

        filename = Path(filename)
        self._filename = filename.absolute()
        self._capture_directory.rundir = filename.absolute().parent
        self.strategy_context.create_from_table_filename(
            self.strategy_table_filename,
            ifile_hlp=self._ifile_hlp,
            creator_dict=self._creator_dict,
        )
        self.notify_process_location(ProcessLocation("update target"))

    def script_retrieval_ms(
        self,
        filename: str | os.PathLike[str],
        writeOutput: bool = False,
        writePlots: bool = False,
        debug: bool = False,
        update_product_format: bool = False,
    ) -> int:
        # Ignore arguments other than filename.
        # We can clean this up if needed, perhaps delay the
        # initialization or something.
        self.update_target(filename)
        return self.retrieval_ms()

    @property
    def process_location_observable(self) -> ProcessLocationObservable:
        return self._process_location_observable

    def add_observer(self, obs: Any) -> None:
        self.process_location_observable.add_observer(obs)

    def remove_observer(self, obs: Any) -> None:
        self.process_location_observable.remove_observer(obs)

    def clear_observers(self) -> None:
        self.process_location_observable.clear_observers()

    def notify_process_location(
        self, location: ProcessLocation | str, **kwargs: Any
    ) -> None:
        loc = location
        if not isinstance(loc, ProcessLocation):
            loc = ProcessLocation(loc)
        self.process_location_observable.notify_process_location(location, **kwargs)

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
    def strategy_executor(self) -> MusesStrategyExecutorMusesStrategy:
        """The MusesStrategyExecutor used to run through the strategy"""
        return self._strategy_executor

    @property
    def input_file_helper(self) -> InputFileHelper:
        """The InputFileHelper used to read input data."""
        return self._ifile_hlp

    @input_file_helper.setter
    def input_file_helper(self, val: InputFileHelper) -> None:
        self._ifile_hlp = val

    @property
    def _output_directory(self) -> Path:
        """Directory we are running in (e.g. where the strategy table and measurement id files
        are)"""
        return Path(self._capture_directory.rundir)

    @property
    def strategy_table_filename(self) -> Path:
        """Name of the strategy table we are using."""
        return self._filename

    @property
    def forward_model_handle_set(self) -> ForwardModelHandleSet:
        """The set of handles we use for mapping instrument name to a
        ForwardModel"""
        return self.creator_dict[rf.ForwardModel]

    @property
    def observation_handle_set(self) -> ObservationHandleSet:
        """The set of handles we use for mapping instrument name to a
        MusesObservation"""
        return self.creator_dict[rf.Observation]

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        """The set of handles we use for each state element."""
        return self.creator_dict[StateElement]

    @property
    def cross_state_element_handle_set(self) -> CrossStateElementHandleSet:
        """The set of handles we use for each state element."""
        return self.creator_dict[CrossStateElement]

    @property
    def current_state(self) -> CurrentState:
        return self.strategy_executor.current_state

    def save_pickle(
        self, save_pickle_file: str | os.PathLike[str], **kwargs: Any
    ) -> None:
        """Dump a pickled version of this object, along with the working
        directory. Pairs with load_retrieval_strategy."""
        self._capture_directory.save_directory(self._output_directory)
        pickle.dump([self, kwargs], open(save_pickle_file, "wb"))

    def load_step_info(
        self,
        current_state_replay_file: str | os.PathLike[str],
        step_number: int,
        ret_state_file: str | os.PathLike[str] | None = None,
    ) -> None:
        self._strategy_executor.load_step_info(
            current_state_replay_file, step_number, ret_state_file
        )

    @classmethod
    def load_retrieval_strategy(
        cls,
        save_pickle_file: str | os.PathLike[str],
        path: str | os.PathLike[str] = ".",
        change_to_dir: bool = False,
        ifile_hlp: InputFileHelper | None = None,
    ) -> tuple[RetrievalStrategy, dict]:
        """This pairs with save_pickle.

        This is pretty direct to use, but as an example we can do
        something like:

        subprocess.run("rm -r ./try_it", shell=True)
        dir_in = "./retrieval_strategy_cris_tropomi/20190807_065_04_08_5"
        step_number=10
        rs, kwargs = RetrievalStrategy.load_retrieval_strategy(
            f"{dir_in}/retrieval_step_{step_number}.pkl",
            path="./try_it",
            change_to_dir=True)
        rs.continue_retrieval()

        """
        res, kwargs = pickle.load(open(save_pickle_file, "rb"))
        output_directory = Path(path).absolute() / res._capture_directory.runbase  # noqa: SLF001
        res._capture_directory.rundir = output_directory  # noqa: SLF001
        res._filename = output_directory / res.strategy_table_filename.name  # noqa: SLF001
        res._strategy_executor.strategy_table_filename = res._filename  # noqa: SLF001
        res._ifile_hlp = ifile_hlp if ifile_hlp is not None else InputFileHelper()  # noqa: SLF001
        res._retrieval_config.ifile_hlp = res.input_file_helper  # noqa: SLF001
        res._retrieval_config.base_dir = output_directory  # noqa: SLF001
        res._capture_directory.extract_directory(path=path, change_to_dir=change_to_dir)  # noqa: SLF001
        return res, kwargs


class RetrievalStrategyCaptureObserver:
    """Helper class, pickles RetrievalStrategy at each time
    notify_process_location is called. Intended for unit tests and other kinds
    of debugging.

    """

    def __init__(
        self,
        basefname: str,
        location_to_capture: str | ProcessLocation,
        retrieval_strategy: RetrievalStrategy,
    ) -> None:
        self.basefname = basefname
        self.retrieval_strategy = retrieval_strategy
        if isinstance(location_to_capture, ProcessLocation):
            self.location_to_capture = location_to_capture
        else:
            self.location_to_capture = ProcessLocation(location_to_capture)

    @property
    def observing_process_location(self) -> list[ProcessLocation]:
        return [
            self.location_to_capture,
        ]

    def notify_process_location(
        self,
        location: ProcessLocation,
        **kwargs: Any,
    ) -> None:
        logger.debug(f"Call to {self.__class__.__name__}::notify_process_location")
        fname = (
            self.retrieval_strategy.retrieval_config["output_directory"]
            / f"{self.basefname}_{self.retrieval_strategy.strategy_executor.step_number}.pkl"
        )
        # Don't want this class included in the pickle
        self.retrieval_strategy.process_location_observable.remove_observer(self)
        self.retrieval_strategy.save_pickle(fname, **kwargs)
        self.retrieval_strategy.process_location_observable.add_observer(self)


class RetrievalStrategyMemoryUse:
    def __init__(self) -> None:
        # Need pympler here, but don't generally need it. Include this
        # so this isn't a requirement, unless we are running with this
        # observer
        from pympler import tracker

        self.tr: None | tracker.SummaryTracker = None

    def notify_process_location(
        self,
        retrieval_strategy: RetrievalStrategy,
        location: ProcessLocation,
        **kwargs: Any,
    ) -> None:
        # Need pympler here, but don't generally need it. Include this
        # so this isn't a requirement, unless we are running with this
        # observer
        from pympler import tracker

        if location == ProcessLocation("starting retrieval steps"):
            self.tr = tracker.SummaryTracker()
        elif location in (
            "done copy_current_initial",
            "done create_windows",
            "done retrieval_step",
            "done next_state_to_current",
        ):
            logger.info(f"Memory change when {location}")
            if self.tr is not None:
                self.tr.print_diff()


__all__ = [
    "RetrievalStrategy",
    "RetrievalStrategyCaptureObserver",
    "RetrievalStrategyMemoryUse",
]
