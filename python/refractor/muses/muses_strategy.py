from __future__ import annotations
from .strategy_table import StrategyTable
import os
import abc
import typing

if typing.TYPE_CHECKING:
    from .retrieval_info import RetrievalInfo
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import CurrentState
    from .identifier import (
        InstrumentIdentifier,
        StateElementIdentifier,
        RetrievalType,
        FilterIdentifier,
    )


class CurrentStrategyStep(object, metaclass=abc.ABCMeta):
    """This contains information about the current strategy step. This
    is little more than a dict giving several properties, but we
    abstract this out so we can test things without needing to use a
    full MusesStrategyExecutor, also so we document what information
    is expected from a strategy step.

    """

    @abc.abstractproperty
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_name(self) -> str:
        """A name for the current strategy step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def step_number(self) -> int:
        """The number of the current strategy step, starting with 0."""
        raise NotImplementedError()

    @abc.abstractproperty
    def microwindow_file_name_override(self) -> str | None:
        """The microwindows file to use, overriding the normal logic that was
        in the old mpy.table_get_spectral_filename. If None, then we don't have
        an override."""
        raise NotImplementedError()

    @abc.abstractproperty
    def max_num_iterations(self) -> int:
        """Maximum number of iterations to used in a retrieval step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        raise NotImplementedError()

    @abc.abstractproperty
    def do_not_update_list(self) -> list[StateElementIdentifier]:
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

    @abc.abstractproperty
    def retrieval_info(self) -> RetrievalInfo:
        """The RetrievalInfo."""
        # Note it would probably be good to remove this if we can. Right now
        # this is only used by RetrievalL2Output. But at least for now, we
        # need to to generate the output
        raise NotImplementedError()


class CurrentStrategyStepDict(CurrentStrategyStep):
    """Implementation of CurrentStrategyStep that uses a dict"""

    def __init__(self, current_strategy_step_dict: dict):
        self.current_strategy_step_dict = current_strategy_step_dict

    @classmethod
    def current_step(cls, strategy_table: StrategyTable) -> CurrentStrategyStepDict:
        """Create a current strategy step, leaving out the
        RetrievalInfo stuff.
        """
        return cls(
            {
                "retrieval_elements": strategy_table.retrieval_elements(),
                "instrument_name": strategy_table.instrument_name(),
                "step_name": strategy_table.step_name,
                "step_number": strategy_table.table_step,
                "max_num_iterations": int(strategy_table.max_num_iterations),
                "retrieval_type": strategy_table.retrieval_type,
                "do_not_update_list": strategy_table.do_not_update_list,
                "error_analysis_interferents": strategy_table.error_analysis_interferents(),
                "spectral_window_dict": None,
                "retrieval_info": None,
            }
        )

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        return self.current_strategy_step_dict["retrieval_elements"]

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        return self.current_strategy_step_dict["instrument_name"]

    @property
    def step_name(self) -> str:
        """A name for the current strategy step."""
        return self.current_strategy_step_dict["step_name"]

    @property
    def step_number(self) -> int:
        """The number of the current strategy step, starting with 0."""
        return self.current_strategy_step_dict["step_number"]

    @property
    def microwindow_file_name_override(self) -> str | None:
        """The microwindows file to use, overriding the normal logic that was
        in the old mpy.table_get_spectral_filename. If None, then we don't have
        an override."""
        return self.current_strategy_step_dict.get("microwindow_file_name_override")

    @property
    def max_num_iterations(self) -> int:
        """Maximum number of iterations to used in a retrieval step."""
        return self.current_strategy_step_dict["max_num_iterations"]

    @property
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        return self.current_strategy_step_dict["retrieval_type"]

    @property
    def do_not_update_list(self) -> list[StateElementIdentifier]:
        return self.current_strategy_step_dict["do_not_update_list"]

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Return a list of the error analysis interferents."""
        return self.current_strategy_step_dict["error_analysis_interferents"]

    @property
    def spectral_window_dict(self) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that."""
        return self.current_strategy_step_dict["spectral_window_dict"]

    @property
    def retrieval_info(self) -> RetrievalInfo:
        """The RetrievalInfo."""
        # Note it would probably be good to remove this if we can. Right now
        # this is only used by RetrievalL2Output. But at least for now, we
        # need to to generate the output
        return self.current_strategy_step_dict["retrieval_info"]


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
    def current_strategy_step(
        self,
        spectral_window_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        retrieval_info: RetrievalInfo | None,
    ) -> CurrentStrategyStep:
        """Return the CurrentStrategyStep for the current step."""
        raise NotImplementedError


class MusesStrategyOldStrategyTable(MusesStrategy):
    """This wraps the old py-retrieve StrategyTable code as a
    MusesStrategy.  Note that this class has largely been replaced
    with MusesStrategyTable, but we leave this in place for backwards
    testings.

    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
    ):
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

    def current_strategy_step(
        self,
        spectral_window_dict: dict[InstrumentIdentifier, MusesSpectralWindow] | None,
        retrieval_info: RetrievalInfo | None,
    ) -> CurrentStrategyStep:
        if self.is_done():
            raise RuntimeError("Past end of strategy")
        cstep = CurrentStrategyStepDict.current_step(self._stable)
        cstep.current_strategy_step_dict["spectral_window_dict"] = spectral_window_dict
        cstep.current_strategy_step_dict["retrieval_info"] = retrieval_info
        return cstep


__all__ = [
    "MusesStrategy",
    "MusesStrategyOldStrategyTable",
    "CurrentStrategyStep",
    "CurrentStrategyStepDict",
]
