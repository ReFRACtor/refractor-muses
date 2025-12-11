from __future__ import annotations
from contextlib import contextmanager
from .retrieval_strategy_step import RetrievalStrategyStepSet
from .current_state_state_info import CurrentStateStateInfo
from .qa_data_handle import QaDataHandleSet
from .muses_strategy import (
    MusesStrategy,
    CurrentStrategyStep,
)
from .observation_handle import ObservationHandleSet
from .identifier import StateElementIdentifier, ProcessLocation
from .muses_strategy import MusesStrategyHandleSet
from .spectral_window_handle import SpectralWindowHandleSet
from .retrieval_strategy_step import RetrievalStepCaptureObserver
from .record_and_play_func import CurrentStateRecordAndPlay
import refractor.framework as rf  # type: ignore
import abc
import copy
import os
from loguru import logger
import time
import numpy as np
import numpy.testing as npt
from pathlib import Path
import typing
from typing import Generator, Any

if typing.TYPE_CHECKING:
    from .forward_model_combine import ForwardModelCombine
    from .retrieval_strategy import RetrievalStrategy
    from .muses_observation import MeasurementId
    from .cost_function import CostFunction
    from .retrieval_configuration import RetrievalConfiguration
    from .cost_function_creator import CostFunctionCreator
    from .identifier import InstrumentIdentifier, FilterIdentifier
    from .state_info import StateElementHandleSet
    from .cross_state_element import CrossStateElementHandleSet


@contextmanager
def log_timing() -> Generator[None, None, None]:
    start_date = time.strftime("%c")
    start_time = time.time()
    yield
    stop_date = time.strftime("%c")
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    elapsed_time_seconds = stop_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60.0
    logger.info("\n---")
    logger.info(f"start_date {start_date}")
    logger.info(f"stop_date {stop_date}")
    logger.info(f"elapsed_time {elapsed_time}")
    logger.info(f"elapsed_time_seconds {elapsed_time_seconds}")
    logger.info(f"elapsed_time_minutes {elapsed_time_minutes}")


def struct_compare(
    s1: dict, s2: dict, skip_list: None | list[str] = None, verbose: bool = False
) -> None:
    if skip_list is None:
        skip_list = []
    for k in s1.keys():
        if k in skip_list:
            if verbose:
                print(f"Skipping {k}")
            continue
        if verbose:
            print(k)
        if isinstance(s1[k], np.ndarray) and np.can_cast(s1[k], np.float64):
            npt.assert_allclose(s1[k], s2[k])
        elif isinstance(s1[k], np.ndarray):
            assert np.all(s1[k] == s2[k])
        else:
            assert s1[k] == s2[k]


def array_compare(
    s1: list[dict],
    s2: list[dict],
    skip_list: None | list[str] = None,
    verbose: bool = False,
) -> None:
    assert len(s1) == len(s2)
    for i in range(len(s1)):
        struct_compare(s1[i], s2[i], skip_list=skip_list, verbose=verbose)


class MusesStrategyExecutor(object, metaclass=abc.ABCMeta):
    """This is the base class for executing a strategy.

    Note that there a refractor.framework class StrategyExecutor. This
    class has a similar intention as that older StrategyExecutor
    class, however this really is a complete rewrite of this for the
    way py-retrieve does this. It is possible that these classes might
    get merged at some point, but for now it is better to think of
    these as completely separate classes that just happen to have
    similar names.

    The canonical way of determining the strategy is to read the old
    strategy table ("Table.asc") that amuse-me populates.

    This base class provides an abstract interface so we can have
    different implementations of executing a strategy.

    It isn't clear how much flexibility we actually need here, a lot
    of the configuration/customization happens at a lower level of
    processing (e.g., ForwardModelHandleSet), but we'll go ahead a set
    up the inheritance structure since this is pretty cheap. If we
    only end up with one implementation that's fine.

    """

    pass


class MusesStrategyExecutorRetrievalStrategyStep(MusesStrategyExecutor):
    """Much of the time our strategy is going to depend on having
    a RetrievalStrategyStepSet to get the RetrievalStrategyStep based
    off a retrieval type name. This adds that functionality."""

    def __init__(
        self,
        rs: RetrievalStrategy,
        observation_handle_set: ObservationHandleSet | None = None,
        muses_strategy_handle_set: MusesStrategyHandleSet | None = None,
        retrieval_strategy_step_set: RetrievalStrategyStepSet | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        qa_data_handle_set: QaDataHandleSet | None = None,
        **kwargs: Any,
    ) -> None:
        self.rs = rs
        self.current_state = CurrentStateStateInfo()
        self.kwargs = copy.copy(kwargs)
        if observation_handle_set is None:
            self.observation_handle_set = copy.deepcopy(
                ObservationHandleSet.default_handle_set()
            )
        else:
            self.observation_handle_set = observation_handle_set
        if muses_strategy_handle_set is None:
            self._muses_strategy_handle_set = copy.deepcopy(
                MusesStrategyHandleSet.default_handle_set()
            )
        else:
            self._muses_strategy_handle_set = muses_strategy_handle_set
        if retrieval_strategy_step_set is None:
            self._retrieval_strategy_step_set = copy.deepcopy(
                RetrievalStrategyStepSet.default_handle_set()
            )
        else:
            self._retrieval_strategy_step_set = retrieval_strategy_step_set

        if spectral_window_handle_set is None:
            self._spectral_window_handle_set = copy.deepcopy(
                SpectralWindowHandleSet.default_handle_set()
            )
        else:
            self._spectral_window_handle_set = spectral_window_handle_set

        if qa_data_handle_set is None:
            self._qa_data_handle_set = copy.deepcopy(
                QaDataHandleSet.default_handle_set()
            )
        else:
            self._qa_data_handle_set = qa_data_handle_set

        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Have updated the target we are processing."""
        self.measurement_id = measurement_id
        self.qa_data_handle_set.notify_update_target(self.measurement_id)

    @property
    def retrieval_config(self) -> RetrievalConfiguration:
        return self.rs.retrieval_config

    @property
    def state_element_handle_set(self) -> StateElementHandleSet:
        return self.current_state.state_element_handle_set

    @property
    def cross_state_element_handle_set(self) -> CrossStateElementHandleSet:
        return self.current_state.cross_state_element_handle_set

    @property
    def run_dir(self) -> Path:
        return self.rs.run_dir

    @property
    def cost_function_creator(self) -> CostFunctionCreator:
        return self.rs.cost_function_creator

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        raise NotImplementedError

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        raise NotImplementedError()

    @property
    def qa_data_handle_set(self) -> QaDataHandleSet:
        """The QaDataHandleSet to use to get the QA flag filename."""
        return self._qa_data_handle_set

    @property
    def muses_strategy_handle_set(self) -> MusesStrategyHandleSet:
        """The MusesStrategyHandleSet used for getting MusesStrategy"""
        return self._muses_strategy_handle_set

    @property
    def retrieval_strategy_step_set(self) -> RetrievalStrategyStepSet:
        """The RetrievalStrategyStepSet to use for getting RetrievalStrategyStep."""
        return self._retrieval_strategy_step_set

    @property
    def spectral_window_handle_set(self) -> SpectralWindowHandleSet:
        """The SpectralWindowHandleSet to use for getting MusesSpectralWindow."""
        return self._spectral_window_handle_set

    def create_forward_model(self) -> rf.ForwardModel:
        """Create a forward model for the current step."""
        if len(self.current_strategy_step.instrument_name) != 1:
            raise RuntimeError(
                "create_forward_model can only work with one instrument, we don't have handling for multiple."
            )
        iname = self.current_strategy_step.instrument_name[0]
        obs = self.observation_handle_set.observation(
            iname, None, self.current_strategy_step.spectral_window_dict[iname], None
        )
        fm_sv = rf.StateVector()
        return self.rs.forward_model_handle_set.forward_model(
            iname,
            self.current_state,
            obs,
            fm_sv,
        )

    def create_forward_model_combine(
        self,
        use_systematic: bool = False,
        include_bad_sample: bool = False,
    ) -> ForwardModelCombine:
        """Like create_cost_function, but create just a ForwardModelCombine instead of
        a full CostFunction."""
        return self.cost_function_creator.forward_model(
            self.current_strategy_step.instrument_name,
            self.current_state,
            self.current_strategy_step.spectral_window_dict,
            use_systematic=use_systematic,
            include_bad_sample=include_bad_sample,
            **self.kwargs,
        )

    def create_cost_function(
        self,
    ) -> CostFunction:
        """Create a CostFunction for use in a retrieval."""
        return self.cost_function_creator.cost_function(
            self.current_strategy_step.instrument_name,
            self.current_state,
            self.current_strategy_step.spectral_window_dict,
            **self.kwargs,
        )


class MusesStrategyExecutorMusesStrategy(MusesStrategyExecutorRetrievalStrategyStep):
    """This is a strategy executor that uses a MusesStrategy to determine the
    strategy.

    It isn't clear if we will ever need a different strategy executor, having different
    MusesStrategy may be all the flexibility we need. But we go ahead and set up
    the infrastructure here since it is fairly cheap to do so, just in case we need
    a different implementation in the future.
    """

    def __init__(
        self,
        rs: RetrievalStrategy,
        muses_strategy_handle_set: None | MusesStrategyHandleSet = None,
        retrieval_strategy_step_set: None | RetrievalStrategyStepSet = None,
        spectral_window_handle_set: None | SpectralWindowHandleSet = None,
        qa_data_handle_set: None | QaDataHandleSet = None,
    ) -> None:
        super().__init__(
            rs,
            observation_handle_set=rs.observation_handle_set,
            muses_strategy_handle_set=muses_strategy_handle_set,
            retrieval_strategy_step_set=retrieval_strategy_step_set,
            spectral_window_handle_set=spectral_window_handle_set,
            qa_data_handle_set=qa_data_handle_set,
            **rs.keyword_arguments,
        )
        self._strategy: MusesStrategy | None = None
        self.measurement_id: MeasurementId | None = None

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        super().notify_update_target(measurement_id)
        self._strategy = self.muses_strategy_handle_set.muses_strategy(
            measurement_id,
            osp_dir=measurement_id.osp_abs_dir,
            spectral_window_handle_set=self.spectral_window_handle_set,
        )
        self.strategy.notify_update_target(measurement_id)
        # Only do notify_update_target if we already have the filter_list_dict filled in.
        # If we don't just skip this
        if len(measurement_id.filter_list_dict) > 0:
            self.current_state.notify_update_target(
                measurement_id,
                self.retrieval_config,
                self.strategy,
                self.observation_handle_set,
            )

    @property
    def strategy(self) -> MusesStrategy:
        if self._strategy is None:
            raise RuntimeError("Call update_target before this function")
        return self._strategy

    def notify_update(self, location: str | ProcessLocation, **kwargs: Any) -> None:
        self.rs.notify_update(location, **kwargs)

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        return self.strategy.filter_list_dict

    @property
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        cstep = self.strategy.current_strategy_step()
        if cstep is None:
            raise RuntimeError("current_strategy_step called after the last step.")
        return cstep

    def restart(self) -> None:
        """Set step to the first one."""
        self.strategy.restart()
        self.current_state.notify_start_retrieval(
            self.current_strategy_step, self.retrieval_config
        )

    def notify_start_step(self, skip_initial_guess_update: bool = False) -> None:
        """Called to notify other object that we are on a new retrieval step."""
        self.current_state.notify_start_step(
            self.current_strategy_step,
            self.retrieval_config,
            skip_initial_guess_update=skip_initial_guess_update,
        )

    def set_step(self, step_number: int) -> None:
        """Go to the given step."""
        self.restart()
        while self.current_strategy_step.strategy_step.step_number < step_number:
            self.notify_start_step(skip_initial_guess_update=True)
            self.next_step()

    def next_step(self) -> None:
        """Advance to the next step."""
        self.strategy.next_step(self.current_state)
        cstep = self.strategy.current_strategy_step()
        if cstep is not None:
            logger.info(str(cstep.strategy_step))

    def is_done(self) -> bool:
        """Return true if we are done, otherwise false."""
        return self.strategy.is_done()

    @property
    def instrument_name_all_step(self) -> list[InstrumentIdentifier]:
        return self.strategy.instrument_name

    @property
    def error_analysis_interferents_all_step(self) -> list[StateElementIdentifier]:
        return self.strategy.error_analysis_interferents

    @property
    def retrieval_elements_all_step(self) -> list[StateElementIdentifier]:
        return self.strategy.retrieval_elements

    def run_step(self) -> None:
        """Run a the current step."""
        logger.info("\n---")
        logger.info(str(self.current_strategy_step.strategy_step))
        logger.info("\n---")
        logger.info(
            f"Step: {self.current_strategy_step.strategy_step.step_number}, Retrieval Type {self.current_strategy_step.retrieval_type}"
        )
        self.retrieval_strategy_step_set.retrieval_step(
            self.current_strategy_step.retrieval_type,
            self.rs,
            **self.current_strategy_step.retrieval_step_parameters,
            **self.kwargs,
        )
        self.notify_update(ProcessLocation("done retrieval_step"))
        logger.info(f"Done with {str(self.current_strategy_step.strategy_step)}")

    def execute_retrieval(self, stop_at_step: None | int = None) -> None:
        """Run through all the steps, i.e., do a full retrieval.

        Note for various testing purposes, you can have the retrieval
        stop at the given step. This can be useful for looking at
        problems with an individual step, or to run a simulation at a
        particular step.
        """
        with log_timing():
            self.execute_retrieval_body(stop_at_step=stop_at_step)

    def execute_retrieval_body(self, stop_at_step: None | int = None) -> None:
        """Run through all the steps, i.e., do a full retrieval.

        Note for various testing purposes, you can have the retrieval
        stop at the given step. This can be useful for looking at
        problems with an individual step, or to run a simulation at a
        particular step.
        """
        if self.measurement_id is None:
            raise RuntimeError(
                "Need to call notify_update_target before calling this function"
            )
        self.restart()
        self.notify_update(ProcessLocation("initial set up done"))
        while not self.is_done():
            self.notify_update(ProcessLocation("starting run_step"))
            self.notify_start_step()
            self.notify_update(ProcessLocation("notify_start_step done"))
            if (
                stop_at_step is not None
                and stop_at_step == self.current_strategy_step.strategy_step.step_number
            ):
                return
            self.run_step()
            self.next_step()
        self.notify_update("retrieval done")

    def continue_retrieval(self, stop_after_step: None | int = None) -> None:
        """After saving a pickled step, you can continue the processing starting
        at that step to diagnose a problem."""
        while not self.is_done():
            self.notify_start_step()
            self.notify_update(ProcessLocation("starting run_step"))
            self.run_step()
            if (
                stop_after_step is not None
                and stop_after_step
                == self.current_strategy_step.strategy_step.step_number
            ):
                return
            self.next_step()
        self.notify_update("retrieval done")

    def load_step_info(
        self,
        current_state_replay_file: str | os.PathLike[str],
        step_number: int,
        ret_state_file: str | os.PathLike[str] | None = None,
    ) -> None:
        """This pairs with CurrentStateRecordAndPlay. Instead of
        pickling the entire RetrievalStrategy, we just save functions
        used on CurrentState. We then set up to process the given
        target_filename jumping to the given retrieval step_number.

        Note for some tests in addition to the CurrentState we want the
        results saved by RetrievalStepCaptureObserver (e.g., we want
        to test the output writing). You can optionally pass in the
        json file for this and we will also pass that information to
        the RetrievalStrategyStep.

        Take a look at the capture_data_test.py examples for how to
        save this data
        """
        rplay = CurrentStateRecordAndPlay(current_state_replay_file)
        # Note we go to the end of the previous step. We then are ready
        # for the current retrieval step.
        if step_number > 0:
            rplay.replay(
                self.current_state,
                self.strategy,
                self.retrieval_config,
                step_number - 1,
            )
            self.next_step()
        if ret_state_file is not None:
            t = RetrievalStepCaptureObserver.load_retrieval_state(ret_state_file)
            self.kwargs["ret_state"] = t


__all__ = [
    "MusesStrategyExecutor",
    "MusesStrategyExecutorRetrievalStrategyStep",
    "MusesStrategyExecutorMusesStrategy",
]
