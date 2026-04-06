from __future__ import annotations
from .muses_spectral_window import MusesSpectralWindow
from .identifier import (
    InstrumentIdentifier,
    StateElementIdentifier,
    RetrievalType,
    StrategyStepIdentifier,
)
from .spectral_window_handle import SpectralWindowHandleSet
from .current_state import CurrentState
from .retrieval_array import RetrievalGridArray
from .muses_strategy_context import MusesStrategyContext, MusesStrategyContextMixin
import copy
import abc
import typing

if typing.TYPE_CHECKING:
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import CurrentState
    from .input_file_helper import InputFilePath


class CurrentStrategyStep(object):
    """This is the base class for a strategy step. The content of the step
    depending on the kind of step (e.g., OE or ML).
    """

    @abc.abstractproperty
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        raise NotImplementedError()

    @abc.abstractproperty
    def strategy_step(self) -> StrategyStepIdentifier:
        """Return the strategy step identifier"""
        raise NotImplementedError()


class CurrentStrategyStepImp(CurrentStrategyStep, MusesStrategyContextMixin):
    """Most of the time the retrieval type is just a fixed value, and we want
    the MusesStrategyContext. This adds this common behavior"""

    def __init__(
        self,
        strategy_context: MusesStrategyContext,
        retrieval_type: RetrievalType,
        strategy_step: StrategyStepIdentifier,
    ) -> None:
        MusesStrategyContextMixin.__init__(self, strategy_context)
        self._retrieval_type = retrieval_type
        self._strategy_step = strategy_step
        self.is_skipped = False

    @property
    def retrieval_type(self) -> RetrievalType:
        return self._retrieval_type

    @property
    def strategy_step(self) -> StrategyStepIdentifier:
        return self._strategy_step


class CurrentStrategyStepOE(CurrentStrategyStepImp):
    """This has extra information needed in a OE retrieval  step"""

    @abc.abstractproperty
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_elements_not_updated(self) -> list[StateElementIdentifier]:
        """List of element that we include in the retrieval step, but
        should go back to the original value in the next step. This is
        always a subset of retrieval_elements (and often an empty
        subset)
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        raise NotImplementedError()

    @abc.abstractproperty
    def spectral_window_dict(self) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Return a dictionary that maps instrument name to the
        MusesSpectralWindow to use for that.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Return a list of the error analysis interferents."""
        raise NotImplementedError()

    @abc.abstractmethod
    def muses_microwindows_fname(self) -> InputFilePath:
        """This is very specific, but there is some complicated code
        used to generate the microwindows file name. This is used to
        create the MusesSpectralWindow (by one of the handlers). Also
        the QA data file name depends on this.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def retrieval_step_parameters(self) -> dict:
        """Any keywords to pass on to the RetrievalStrategyStep retrieve_step (e.g
        arguments for cost function"""
        raise NotImplementedError()

    @abc.abstractmethod
    def notify_step_solution(
        self, current_state: CurrentState, xsol: RetrievalGridArray
    ) -> None:
        """Update the CurrentState with the solution of a retrieval
        step. We have this as part of CurrentStrategyStep so we can
        support any sort of more complicated logic for updating the
        state (e.g., update the apriori)
        """
        raise NotImplementedError


class CurrentStrategyStepDict(CurrentStrategyStepOE):
    """Implementation of CurrentStrategyStep that uses a dict"""

    def __init__(
        self,
        current_strategy_step_dict: dict,
        strategy_context: MusesStrategyContext,
        spectral_window_handle_set: SpectralWindowHandleSet,
    ) -> None:
        super().__init__(
            strategy_context,
            current_strategy_step_dict["retrieval_type"],
            current_strategy_step_dict["strategy_step"],
        )
        self.spectral_window_handle_set = spectral_window_handle_set
        self.current_strategy_step_dict = current_strategy_step_dict
        self.is_skipped = False

    @property
    def retrieval_step_parameters(self) -> dict:
        # Fill in some of the data that comes in retrieval config
        res = copy.deepcopy(
            self.current_strategy_step_dict["retrieval_step_parameters"]
        )
        if self.has_retrieval_config:
            p = res["cost_function_params"]
            p["delta_value"] = int(self.retrieval_config["LMDelta"].split()[0])
            if p["conv_tolerance"] is None:
                p["conv_tolerance"] = [
                    float(self.retrieval_config["ConvTolerance_CostThresh"]),
                    float(self.retrieval_config["ConvTolerance_pThresh"]),
                    float(self.retrieval_config["ConvTolerance_JacThresh"]),
                ]
        return res

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        return StateElementIdentifier.sort_identifier(
            self.current_strategy_step_dict["retrieval_elements"]
        )

    @retrieval_elements.setter
    def retrieval_elements(self, v: list[StateElementIdentifier]) -> None:
        self.current_strategy_step_dict["retrieval_elements"] = v

    @property
    def retrieval_elements_not_updated(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        return StateElementIdentifier.sort_identifier(
            self.current_strategy_step_dict["retrieval_elements_not_updated"]
        )

    @retrieval_elements_not_updated.setter
    def retrieval_elements_not_updated(self, v: list[StateElementIdentifier]) -> None:
        self.current_strategy_step_dict["retrieval_elements_not_updated"] = v

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        sdict = self.spectral_window_handle_set.spectral_window_dict(self, None)
        return InstrumentIdentifier.sort_identifier(list(sdict.keys()))

    @property
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        return self.current_strategy_step_dict["retrieval_type"]

    @property
    def spectral_window_dict(self) -> dict[InstrumentIdentifier, MusesSpectralWindow]:
        """Return a dictionary that maps instrument name to the MusesSpectralWindow
        to use for that."""
        return self.spectral_window_handle_set.spectral_window_dict(
            self, self.filter_list_dict
        )

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Return a list of the error analysis interferents."""
        return StateElementIdentifier.sort_identifier(
            self.current_strategy_step_dict["error_analysis_interferents"]
        )

    def notify_step_solution(
        self, current_state: CurrentState, xsol: RetrievalGridArray
    ) -> None:
        current_state.notify_step_solution(xsol.view(RetrievalGridArray))
        for selem_id in self.current_strategy_step_dict["update_constraint_elements"]:
            v = current_state.state_value(selem_id)
            current_state.update_full_state_element(
                selem_id, next_constraint_vector_fm=v
            )

    def muses_microwindows_fname(self) -> InputFilePath:
        """This is very specific, but there is some complicated code used to generate the
        microwindows file name. This is used to create the MusesSpectralWindow (by
        one of the handlers). Also the QA data file name depends on this."""
        mid = self.measurement_id
        return MusesSpectralWindow.muses_microwindows_fname(
            mid["viewingMode"],
            mid["spectralWindowDirectory"],
            self.retrieval_elements,
            self.strategy_step.step_name,
            self.retrieval_type,
            self.current_strategy_step_dict.get("microwindow_file_name_override"),
        )

    def __eq__(self, other: object) -> bool:
        """This is useful for unit tests. I don't think we in general need to check
        equality, but if so might want to check the logic here. This is set up so
        we can check that two CurrentStrategyStepDict will give the same results in
        a retrieval step, which may or may not be the right criteria for a more general
        equality test."""

        # To diagnose problem, can be useful to know exactly where we fail.
        # Set to true to break on failure
        break_on_fail = False
        if not isinstance(other, CurrentStrategyStepDict):
            if break_on_fail:
                breakpoint()
            return False
        if sorted(list(self.current_strategy_step_dict.keys())) != sorted(
            list(other.current_strategy_step_dict.keys())
        ):
            if break_on_fail:
                breakpoint()
            return False
        for k in self.current_strategy_step_dict.keys():
            if k == "spectral_window_dict":
                if list(self.spectral_window_dict.keys()) != list(
                    other.spectral_window_dict.keys()
                ):
                    if break_on_fail:
                        breakpoint()
                    return False
                for k2 in self.spectral_window_dict.keys():
                    if (
                        self.spectral_window_dict[k2].muses_microwindows()
                        != other.spectral_window_dict[k2].muses_microwindows()
                    ):
                        if break_on_fail:
                            breakpoint()
                        return False
            elif k == "retrieval_step_parameters":
                if sorted(list(self.retrieval_step_parameters.keys())) != sorted(
                    list(other.retrieval_step_parameters.keys())
                ):
                    if break_on_fail:
                        breakpoint()
                    return False
                d1 = self.retrieval_step_parameters["cost_function_params"]
                d2 = other.retrieval_step_parameters["cost_function_params"]
                if sorted(list(d1.keys())) != sorted(list(d2.keys())):
                    if break_on_fail:
                        breakpoint()
                    return False
                for k2 in d1.keys():
                    if d1[k2] != d2[k2]:
                        if break_on_fail:
                            breakpoint()
                        return False
            elif k in ("retrieval_elements", "error_analysis_interferents"):
                if StateElementIdentifier.sort_identifier(
                    self.current_strategy_step_dict[k]
                ) != StateElementIdentifier.sort_identifier(
                    other.current_strategy_step_dict[k]
                ):
                    if break_on_fail:
                        breakpoint()
                    return False
            elif k == "strategy_step":
                if str(self.strategy_step) != str(other.strategy_step):
                    if break_on_fail:
                        breakpoint()
                    return False
            else:
                if (
                    self.current_strategy_step_dict[k]
                    != other.current_strategy_step_dict[k]
                ):
                    if break_on_fail:
                        breakpoint()
                    return False
        return True


__all__ = [
    "CurrentStrategyStep",
    "CurrentStrategyStepImp",
    "CurrentStrategyStepDict",
    "CurrentStrategyStepOE",
]
