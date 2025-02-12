from __future__ import annotations
from .strategy_table import StrategyTable
from .muses_spectral_window import MusesSpectralWindow
from .identifier import RetrievalType, StrategyStepIdentifier
import os
import abc
import typing
import numpy as np
import copy

if typing.TYPE_CHECKING:
    from .muses_observation import MeasurementId
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import CurrentState
    from .spectral_window_handle import SpectralWindowHandleSet
    from .retrieval_info import RetrievalInfo
    from .identifier import (
        InstrumentIdentifier,
        StateElementIdentifier,
        RetrievalType,
        FilterIdentifier,
    )


class CurrentStrategyStep(object, metaclass=abc.ABCMeta):
    """This contains information about the current strategy step."""

    @abc.abstractproperty
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        raise NotImplementedError()

    @abc.abstractproperty
    def spectral_window_dict(self) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that."""
        raise NotImplementedError()

    @abc.abstractproperty
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Return a list of the error analysis interferents."""
        raise NotImplementedError()

    @abc.abstractmethod
    def update_state(
        self,
        current_state: CurrentState,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
    ):
        """Update the CurrentState with the results. We have this as part of
        CurrentStrategyStep so we can support any sort of more complicated logic
        for updating the state (e.g., update the apriori)"""
        raise NotImplementedError()

    @abc.abstractmethod
    def muses_microwindows_fname(self):
        """This is very specific, but there is some complicated code used to generate the
        microwindows file name. This is used to create the MusesSpectralWindow (by
        one of the handlers). Also the QA data file name depends on this. It would be nice to
        remove this dependency, but for now we can at least isolate this and make it clear where
        we depend on this."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_step_parameters(self) -> dict:
        """Any keywords to pass on to the RetrievalStrategyStep retrieve_step (e.g
        arguments for cost function"""
        raise NotImplementedError()

    @abc.abstractproperty
    def strategy_step(self) -> StrategyStepIdentifier:
        """Return the strategy step identifier"""
        raise NotImplementedError()


class CurrentStrategyStepDict(CurrentStrategyStep):
    """Implementation of CurrentStrategyStep that uses a dict"""

    def __init__(self, current_strategy_step_dict: dict, measurement_id: MeasurementId):
        self.current_strategy_step_dict = current_strategy_step_dict
        self.measurement_id = measurement_id

    @classmethod
    def current_step(
        cls, strategy_table: StrategyTable, measurement_id: MeasurementId
    ) -> CurrentStrategyStepDict:
        """Create a current strategy step, leaving out the
        RetrievalInfo stuff.
        """
        # Various convergence criteria for solver. This is the MusesLevmarSolver. Note the
        # different convergence depending on the step type. The chi2_tolerance is calculated
        # in RetrievalStrategyStepRetrieve if we don't fill it in - this depends on the
        # size of the radiance data
        cost_function_params = {
            "max_iter": int(strategy_table.max_num_iterations),
            "delta_value": int(measurement_id["LMDelta"].split()[0]),
            "conv_tolerance": [
                float(measurement_id["ConvTolerance_CostThresh"]),
                float(measurement_id["ConvTolerance_pThresh"]),
                float(measurement_id["ConvTolerance_JacThresh"]),
            ],
            "chi2_tolerance": None,  # Filled in by RetrievalStrategyStepRetrieve
        }
        if strategy_table.retrieval_type == RetrievalType("bt_ig_refine"):
            cost_function_params["conv_tolerance"] = [0.00001, 0.00001, 0.00001]
            cost_function_params["chi2_tolerance"] = 0.00001
        return cls(
            {
                "retrieval_elements": strategy_table.retrieval_elements(),
                "instrument_name": strategy_table.instrument_name(),
                "strategy_step": StrategyStepIdentifier(
                    strategy_table.table_step, strategy_table.step_name
                ),
                "retrieval_step_parameters": {
                    "cost_function_params": cost_function_params,
                },
                "retrieval_type": strategy_table.retrieval_type,
                "error_analysis_interferents": strategy_table.error_analysis_interferents(),
                "spectral_window_dict": None,
                "do_not_update_list": strategy_table.do_not_update_list,
            },
            measurement_id,
        )

    @property
    def retrieval_step_parameters(self) -> dict:
        return self.current_strategy_step_dict["retrieval_step_parameters"]

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        return self.current_strategy_step_dict["retrieval_elements"]

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        return self.current_strategy_step_dict["instrument_name"]

    @property
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        return self.current_strategy_step_dict["retrieval_type"]

    @property
    def spectral_window_dict(self) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that."""
        return self.current_strategy_step_dict["spectral_window_dict"]

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Return a list of the error analysis interferents."""
        return self.current_strategy_step_dict["error_analysis_interferents"]

    def update_state(
        self,
        current_state: CurrentState,
        retrieval_info: RetrievalInfo,
        results_list: np.ndarray,
    ):
        """Update the CurrentState with the results. We have this as part of
        CurrentStrategyStep so we can support any sort of more complicated logic
        for updating the state (e.g., update the apriori)"""
        current_state.update_state(
            retrieval_info,
            results_list,
            self.current_strategy_step_dict["do_not_update_list"],
            self.measurement_id,
            self.strategy_step.step_number,
        )

    def muses_microwindows_fname(self):
        """This is very specific, but there is some complicated code used to generate the
        microwindows file name. This is used to create the MusesSpectralWindow (by
        one of the handlers). Also the QA data file name depends on this. It would be nice to
        remove this dependency, but for now we can at least isolate this and make it clear where
        we depend on this."""
        if self.measurement_id is None:
            raise RuntimeError("Call notify_update_target before this function")
        return MusesSpectralWindow.muses_microwindows_fname_from_muses_py(
            self.measurement_id["viewingMode"],
            self.measurement_id["spectralWindowDirectory"],
            self.retrieval_elements,
            self.strategy_step.step_name,
            self.retrieval_type,
            self.current_strategy_step_dict.get("microwindow_file_name_override"),
        )

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        """Return the strategy step identifier"""
        return self.current_strategy_step_dict["strategy_step"]


class MusesStrategy(object, metaclass=abc.ABCMeta):
    """A MusesStrategy is a list of steps to be executed by a
    MusesStrategyExecutor.  Each step is represented by a
    StrategyStep, which give the list of retrieval elements, spectral
    windows, etc.

    The canonical MusesStrategy is to read a Table.asc file (e.g.,
    StrategyTable) to get the information about each step.

    Note that later steps can be changed based off of the results of
    previous steps. This part of the code may well need to be more
    developed, the only example we have right now is choosing steps
    based off of brightness temperature of the observation. If we get
    for examples of this, we may want to rework how conditional
    processing is handled.

    """

    def notify_update_target(self, measurement_id: MeasurementId):
        pass

    @abc.abstractmethod
    def is_next_bt(self):
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        # TODO Possibly rework this interface by changing
        # RetrievalStrategyStepBT
        raise NotImplementedError()

    @abc.abstractproperty
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all
        retrieval steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all
        retrieval steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval
        steps)

        Note that because we want to allow a strategy to be dynamic,
        we don't really know in general what this list will be. Right
        now, this is needed by other code - perhaps we can remove this
        dependency. But for now this is required. I believe this can
        be a super set - so if we have a item listed here that isn't
        actually used that is ok.

        """
        # TODO Can this go away?
        raise NotImplementedError()

    @abc.abstractmethod
    def restart(self) -> None:
        """Set step to the first one."""
        raise NotImplementedError()

    @abc.abstractmethod
    def next_step(self, current_state: CurrentState) -> None:
        """Advance to the next step."""
        raise NotImplementedError()

    def set_step(self, step_number: int, current_state: CurrentState) -> None:
        """Set the step number to the given value"""
        # Default is just to restart and proceed through the given number of steps.
        self.restart()
        for i in range(step_number):
            self.next_step(current_state)

    @abc.abstractmethod
    def is_done(self) -> bool:
        """True if we have reached the last step"""
        raise NotImplementedError()

    @abc.abstractmethod
    def current_strategy_step(self) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        raise NotImplementedError


class MusesStrategyImp(MusesStrategy):
    """Base class for the way we generally implement a MusesStrategy"""

    def __init__(
        self, spectral_window_handle_set: SpectralWindowHandleSet | None = None
    ):
        self._measurement_id: MeasurementId | None = None
        if spectral_window_handle_set is None:
            self._spectral_window_handle_set = copy.deepcopy(
                SpectralWindowHandleSet.default_handle_set()
            )
        else:
            self._spectral_window_handle_set = spectral_window_handle_set

    @property
    def measurement_id(self) -> MeasurementId | None:
        return self._measurement_id

    @property
    def spectral_window_handle_set(self):
        """The SpectralWindowHandleSet to use for getting the MusesSpectralWindow."""
        return self._spectral_window_handle_set

    def notify_update_target(self, measurement_id: MeasurementId):
        self._measurement_id = measurement_id
        self.spectral_window_handle_set.notify_update_target(self.measurement_id)


class MusesStrategyOldStrategyTable(MusesStrategyImp):
    """This wraps the old py-retrieve StrategyTable code as a
    MusesStrategy.  Note that this class has largely been replaced
    with MusesStrategyTable, but we leave this in place for backwards
    testings.

    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
    ):
        super().__init__(spectral_window_handle_set)
        self._stable = StrategyTable(filename, osp_dir=osp_dir)

    def is_next_bt(self):
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        return self._stable.is_next_bt()

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for all retrieval steps)"""
        # TODO Can this go away?
        return self._stable.filter_list_all()

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all retrieval steps)"""
        # TODO Can this go away?
        return self._stable.retrieval_elements_all_step

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all retrieval steps)"""
        # TODO Can this go away?
        return self._stable.error_analysis_interferents_all_step

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval steps)"""
        # TODO Can this go away?
        return self._stable.instrument_name(all_step=True)

    def restart(self) -> None:
        """Set step to the first one."""
        self._stable.table_step = 0

    def next_step(self, current_state: CurrentState) -> None:
        """Advance to the next step."""
        self._stable.next_step(current_state)

    def is_done(self) -> bool:
        """True if we have reached the last step"""
        return self._stable.is_done()

    def current_strategy_step(self) -> CurrentStrategyStep:
        if self.is_done():
            raise RuntimeError("Past end of strategy")
        if self.measurement_id is None:
            raise RuntimeError(
                "Need to call notify_update_target before calling this function."
            )
        cstep = CurrentStrategyStepDict.current_step(self._stable, self.measurement_id)
        cstep.current_strategy_step_dict["spectral_window_dict"] = (
            self.spectral_window_handle_set.spectral_window_dict(cstep)
        )
        return cstep


__all__ = [
    "MusesStrategy",
    "MusesStrategyOldStrategyTable",
    "CurrentStrategyStep",
    "CurrentStrategyStepDict",
]
