from __future__ import annotations
from .muses_spectral_window import MusesSpectralWindow
from .identifier import (
    InstrumentIdentifier,
    StateElementIdentifier,
    RetrievalType,
    StrategyStepIdentifier,
)
from .creator_dict import CreatorDict
from .creator_handle import CreatorHandleWithContextSet, CreatorHandleWithContext
from .spectral_window_handle import SpectralWindowHandleSet
from .current_state import CurrentState
from .retrieval_array import RetrievalGridArray
from .muses_strategy_context import MusesStrategyContext, MusesStrategyContextMixin
import copy
import abc
import typing
from typing import Any

if typing.TYPE_CHECKING:
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import CurrentState
    from .input_file_helper import InputFilePath


class CurrentStrategyStep(object):
    """This is the base class for a strategy step. The content of the step
    depending on the kind of step (e.g., OE or ML).
    """

    @property
    def is_skipped(self) -> bool:
        """True if we should skip this step"""
        return False

    @is_skipped.setter
    def is_skipped(self, v: bool) -> None:
        raise NotImplementedError

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
        CurrentStrategyStep.__init__(self)
        self._retrieval_type = retrieval_type
        self._strategy_step = strategy_step
        self._is_skipped = False

    @property
    def is_skipped(self) -> bool:
        """True if we should skip this step"""
        return self._is_skipped

    @is_skipped.setter
    def is_skipped(self, v: bool) -> None:
        self._is_skipped = v

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


class CurrentStrategyStepOEImp(CurrentStrategyStepOE):
    def __init__(
        self,
        strategy_context: MusesStrategyContext,
        spectral_window_handle_set: SpectralWindowHandleSet,
        retrieval_type: RetrievalType,
        retrieval_elements: list[StateElementIdentifier],
        strategy_step: StrategyStepIdentifier,
        retrieval_step_parameters: dict[str, Any],
        error_analysis_interferents: list[StateElementIdentifier],
        retrieval_elements_not_updated: list[StateElementIdentifier],
        update_constraint_elements: list[StateElementIdentifier],
        microwindow_file_name_override: str | None,
    ) -> None:
        super().__init__(
            strategy_context,
            retrieval_type,
            strategy_step,
        )
        self.spectral_window_handle_set = spectral_window_handle_set
        self._retrieval_step_parameters = retrieval_step_parameters
        self._retrieval_elements = retrieval_elements
        self._error_analysis_interferents = error_analysis_interferents
        self._retrieval_elements_not_updated = retrieval_elements_not_updated
        self._update_constraint_elements = update_constraint_elements
        self._microwindow_file_name_override = microwindow_file_name_override

    @property
    def retrieval_step_parameters(self) -> dict[str, Any]:
        # Fill in some of the data that comes in retrieval config
        res = copy.deepcopy(self._retrieval_step_parameters)
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
        return StateElementIdentifier.sort_identifier(self._retrieval_elements)

    @retrieval_elements.setter
    def retrieval_elements(self, v: list[StateElementIdentifier]) -> None:
        self._retrieval_elements = v

    @property
    def retrieval_elements_not_updated(self) -> list[StateElementIdentifier]:
        """List of retrieval elements that we retrieve for this step."""
        return StateElementIdentifier.sort_identifier(
            self._retrieval_elements_not_updated
        )

    @retrieval_elements_not_updated.setter
    def retrieval_elements_not_updated(self, v: list[StateElementIdentifier]) -> None:
        self._retrieval_elements_not_updated = v

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """List of instruments used in this step."""
        sdict = self.spectral_window_handle_set.spectral_window_dict(self, None)
        return InstrumentIdentifier.sort_identifier(list(sdict.keys()))

    @property
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
        return self._retrieval_type

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
        return StateElementIdentifier.sort_identifier(self._error_analysis_interferents)

    def notify_step_solution(
        self, current_state: CurrentState, xsol: RetrievalGridArray
    ) -> None:
        current_state.notify_step_solution(xsol.view(RetrievalGridArray))
        for selem_id in self._update_constraint_elements:
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
            self._microwindow_file_name_override,
        )


class CurrentStrategyStepHandleSet(CreatorHandleWithContextSet):
    """Create a CurrentStrategyStep"""

    def __init__(self, strategy_context: MusesStrategyContext | None = None) -> None:
        super().__init__("current_strategy_step", strategy_context)

    # This is kind of specific to the kind of strategy files we have. For
    # now we just take a dict of keyword/values. We can perhaps rework this
    # if we get other examples.
    def create_current_strategy_step(
        self,
        index: int,
        table_row: dict,
        spectral_window_handle_set: SpectralWindowHandleSet,
    ) -> CurrentStrategyStep:
        """This does the QA calculation, and updates the given RetrievalResult.
        Returns the master quality flag results"""
        return self.handle(index, table_row, spectral_window_handle_set)


class CurrentStrategyStepHandle(CreatorHandleWithContext):
    def create_current_strategy_step(
        self,
        index: int,
        table_row: dict,
        spectral_window_handle_set: SpectralWindowHandleSet,
    ) -> CurrentStrategyStep:
        raise NotImplementedError()


class CurrentStrategyStepHandleOE(CreatorHandleWithContext):
    def create_current_strategy_step(
        self,
        index: int,
        table_row: dict,
        spectral_window_handle_set: SpectralWindowHandleSet,
    ) -> CurrentStrategyStep:
        cost_function_params: dict[str, Any] = {
            "max_iter": int(table_row["maxNumIterations"]),
            "chi2_tolerance": None,
            # Will fill in from strategy in CurrentStrategyStepDict
            "delta_value": None,
            "conv_tolerance": None,
        }
        if RetrievalType(table_row["retrievalType"]) == RetrievalType("bt_ig_refine"):
            cost_function_params["conv_tolerance"] = [0.00001, 0.00001, 0.00001]
            cost_function_params["chi2_tolerance"] = 0.00001
        retrieval_elements = [
            StateElementIdentifier(i) for i in table_row["retrievalElements"]
        ]
        strategy_step = StrategyStepIdentifier(index, table_row["stepName"])
        retrieval_step_parameters = {
            "cost_function_params": cost_function_params,
        }
        retrieval_type = RetrievalType(table_row["retrievalType"])
        error_analysis_interferents = [
            StateElementIdentifier(i) for i in table_row["errorAnalysisInterferents"]
        ]
        # List of elements that we include in this step, but then
        # set back to their original value for the next step
        retrieval_elements_not_updated = [
            StateElementIdentifier(i) for i in table_row["donotupdate"]
        ]
        update_constraint_elements = []
        microwindow_file_name_override = table_row.get("specFile", None)
        # The py-retrieve strategy table just "knows" that certain
        # retrieval types also update the apriori value. We duplicate this
        # behavior, although it would be nice to have a cleaner way of doing this
        # (e.g., maybe just have a update_constraint_elements column in the table?)
        if retrieval_type == RetrievalType("tropomicloud_ig_refine"):
            update_constraint_elements.append(
                StateElementIdentifier("TROPOMICLOUDFRACTION")
            )
        if retrieval_type == RetrievalType("omicloud_ig_refine"):
            update_constraint_elements.append(
                StateElementIdentifier("OMICLOUDFRACTION")
            )
        return CurrentStrategyStepOEImp(
            self.strategy_context,
            spectral_window_handle_set,
            retrieval_type,
            retrieval_elements,
            strategy_step,
            retrieval_step_parameters,
            error_analysis_interferents,
            retrieval_elements_not_updated,
            update_constraint_elements,
            microwindow_file_name_override,
        )


CurrentStrategyStepHandleSet.add_default_handle(CurrentStrategyStepHandleOE())
# Register creator set
CreatorDict.register(CurrentStrategyStep, CurrentStrategyStepHandleSet)

__all__ = [
    "CurrentStrategyStep",
    "CurrentStrategyStepImp",
    "CurrentStrategyStepOE",
    "CurrentStrategyStepOEImp",
    "CurrentStrategyStepHandleSet",
    "CurrentStrategyStepHandle",
    "CurrentStrategyStepHandleOE",
]
