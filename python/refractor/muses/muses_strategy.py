from __future__ import annotations
from .muses_spectral_window import MusesSpectralWindow
from .creator_handle import CreatorHandle, CreatorHandleSet
from .tes_file import TesFile
from .identifier import (
    InstrumentIdentifier,
    StateElementIdentifier,
    RetrievalType,
    FilterIdentifier,
    StrategyStepIdentifier,
    IdentifierSortByWaveLength,
)
import os
import abc
import typing
import numpy as np
import copy
from collections import defaultdict
from typing import Any, cast

if typing.TYPE_CHECKING:
    from .muses_observation import MeasurementId
    from .muses_spectral_window import MusesSpectralWindow
    from .current_state import CurrentState
    from .spectral_window_handle import SpectralWindowHandleSet
    from .retrieval_strategy import RetrievalStrategy

# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray
RetrievalGrid2dArray = np.ndarray
ForwardModelGrid2dArray = np.ndarray


class CurrentStrategyStep(object, metaclass=abc.ABCMeta):
    """This contains information about the current strategy step."""

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
    def retrieval_type(self) -> RetrievalType:
        """The retrieval type."""
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
    def muses_microwindows_fname(self) -> str:
        """This is very specific, but there is some complicated code
        used to generate the microwindows file name. This is used to
        create the MusesSpectralWindow (by one of the handlers). Also
        the QA data file name depends on this. It would be nice to
        remove this dependency, but for now we can at least isolate
        this and make it clear where we depend on this.

        """
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


class CurrentStrategyStepDict(CurrentStrategyStep):
    """Implementation of CurrentStrategyStep that uses a dict"""

    def __init__(
        self, current_strategy_step_dict: dict, measurement_id: MeasurementId | None
    ) -> None:
        self.current_strategy_step_dict = current_strategy_step_dict
        self.measurement_id = measurement_id
        self.is_skipped = False

    @property
    def retrieval_step_parameters(self) -> dict:
        return self.current_strategy_step_dict["retrieval_step_parameters"]

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
        return InstrumentIdentifier.sort_identifier(
            self.current_strategy_step_dict["instrument_name"]
        )

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
        return StateElementIdentifier.sort_identifier(
            self.current_strategy_step_dict["error_analysis_interferents"]
        )

    def notify_step_solution(
        self, current_state: CurrentState, xsol: RetrievalGridArray
    ) -> None:
        current_state.notify_step_solution(xsol)
        for selem_id in self.current_strategy_step_dict["update_constraint_elements"]:
            v = current_state.full_state_value(selem_id)
            current_state.update_full_state_element(selem_id, constraint_vector_fm=v)

    def muses_microwindows_fname(self) -> str:
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


class MusesStrategyHandle(CreatorHandle, metaclass=abc.ABCMeta):
    """Base class for MusesStrategyHandle. Note we use duck typing, so
    you don't need to actually derive from this object. But it can be
    useful because it 1) provides the interface and 2) documents that
    a class is intended for this.

    This can do caching based on assuming the target is the same
    between calls, see CreatorHandle for a discussion of this.
    """

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        """Clear any caching associated with assuming the target being
        retrieved is fixed"""
        # Default is to do nothing
        pass

    @abc.abstractmethod
    def muses_strategy(
        self,
        measurement_id: MeasurementId,
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        """Return MusesStrategy if we can process the given
        measurement_id, or None if we can't.
        """
        raise NotImplementedError()


class MusesStrategyHandleSet(CreatorHandleSet):
    """This takes the MeasurementId and creates a MusesStrategy for
    processing it."""

    def __init__(self) -> None:
        super().__init__("muses_strategy")

    def muses_strategy(
        self,
        measurement_id: MeasurementId,
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        **kwargs: Any,
    ) -> MusesStrategy:
        """Create a MusesStrategy for the given measurement_id."""
        return self.handle(
            measurement_id, osp_dir, spectral_window_handle_set, **kwargs
        )


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

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        pass

    @abc.abstractmethod
    def is_next_bt(self) -> bool:
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

        Note there is a assumption in muses-py that the
        list[FilterIdentifier] is sorted by the starting wavelength
        for each of filters, so we do that here.
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

        There is an assumption in the muses-py code that the instruments
        are sorted by whatever the smallest wavelength in the SpectralWindow
        for that instrument is, so this list is sorted by that logic.
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
    def current_strategy_step(self) -> CurrentStrategyStep | None:
        """Return the CurrentStrategyStep for the current step. Returns None is self.is_done()"""
        raise NotImplementedError


class MusesStrategyImp(MusesStrategy):
    """Base class for the way we generally implement a MusesStrategy"""

    def __init__(
        self, spectral_window_handle_set: SpectralWindowHandleSet | None = None
    ) -> None:
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
    def spectral_window_handle_set(self) -> SpectralWindowHandleSet:
        """The SpectralWindowHandleSet to use for getting the MusesSpectralWindow."""
        return self._spectral_window_handle_set

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        self._measurement_id = measurement_id
        self.spectral_window_handle_set.notify_update_target(self.measurement_id)


class MusesStrategyFileHandle(MusesStrategyHandle):
    def muses_strategy(
        self,
        measurement_id: MeasurementId,
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        """Return MusesStrategy if we can process the given
        measurement_id, or None if we can't.
        """
        return MusesStrategyStepList.create_from_strategy_table_file(
            measurement_id["run_dir"] / "Table.asc", osp_dir, spectral_window_handle_set
        )


class MusesStrategyStepList(MusesStrategyImp):
    """This implementation uses a list of CurrentStrategyStepDict."""

    def __init__(
        self,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
    ) -> None:
        """This uses a list of CurrentStrategyStep,
        self.current_strategy_list.  Note that there is a bit of
        information that depends on the measurement_id, we fill that
        in when notify_update_target. Also, we may end up skipping
        some of the steps. The current_strategy_list can get modified
        (e.g., a derived class or outside process changes this), which
        can be useful to modify a StrategyTable. This can be useful
        for doing a variation of a StrategyTable for a test or
        something like that, without needing to modify the original
        StrategyTable.

        The various StateElement lists (like retrieval_elements) need
        to be ordered - however you don't need to worry about that when
        adding to the list. We do the sorting automatically in
        CurrentStrategyStepDict.

        """
        super().__init__(spectral_window_handle_set)
        self._filter_list_dict: dict[InstrumentIdentifier, list[FilterIdentifier]] = {}
        self._instrument_name: list[InstrumentIdentifier] = []
        self.current_strategy_list: list[CurrentStrategyStepDict] = []
        self._cur_step_index = 0
        self._cur_step_count = 0

    @classmethod
    def create_from_strategy_table_file(
        cls,
        filename: str | os.PathLike[str],
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
    ) -> MusesStrategyStepList:
        """Create a MusesStrategyStepList from a strategy table file."""
        res = cls(spectral_window_handle_set)
        fin = TesFile(filename)
        if fin.table is None:
            raise RuntimeError(f"Trouble reading {filename}")
        i2 = -1
        for i, row in fin.table.iterrows():
            i2 += 1
            cost_function_params: dict[str, Any] = {
                "max_iter": int(row["maxNumIterations"]),
                "chi2_tolerance": None,  # Filled in by RetrievalStrategyStepRetrieve
                # Will fill in notify_update_target
                "delta_value": None,
                "conv_tolerance": None,
            }
            if RetrievalType(row["retrievalType"]) == RetrievalType("bt_ig_refine"):
                cost_function_params["conv_tolerance"] = [0.00001, 0.00001, 0.00001]
                cost_function_params["chi2_tolerance"] = 0.00001
            cstepdict = {
                "retrieval_elements": res._parse_state_elements(
                    row["retrievalElements"]
                ),
                # Will fill in notify_update_target
                "instrument_name": None,
                "strategy_step": StrategyStepIdentifier(i2, row["stepName"]),
                "retrieval_step_parameters": {
                    "cost_function_params": cost_function_params,
                },
                "retrieval_type": RetrievalType(row["retrievalType"]),
                "error_analysis_interferents": res._parse_state_elements(
                    row["errorAnalysisInterferents"]
                ),
                # Will fill in notify_update_target
                "spectral_window_dict": None,
                # List of elements that we include in this step, but then
                # set back to their original value for the next step
                "retrieval_elements_not_updated": res._parse_state_elements(
                    row["donotupdate"]
                ),
                # List of elements that we update the apriori to match what
                # we retrieve
                "update_constraint_elements": [],
            }
            # Will fill in measurement_id in notify_update_target
            cstep = CurrentStrategyStepDict(cstepdict, None)
            # The muses-py strategy table just "knows" that certain
            # retrieval types also update the apriori value. We duplicate this
            # behavior, although it would be nice to have a cleaner way of doing this
            # (e.g., maybe just have a update_constraint_elements column in the table?)
            if cstep.retrieval_type == RetrievalType("tropomicloud_ig_refine"):
                cstep.current_strategy_step_dict["update_constraint_elements"].append(
                    StateElementIdentifier("TROPOMICLOUDFRACTION")
                )
            if cstep.retrieval_type == RetrievalType("omicloud_ig_refine"):
                cstep.current_strategy_step_dict["update_constraint_elements"].append(
                    StateElementIdentifier("OMICLOUDFRACTION")
                )
            res.current_strategy_list.append(cstep)
        return res

    @property
    def filter_list_dict(self) -> dict[InstrumentIdentifier, list[FilterIdentifier]]:
        """The complete list of filters we will be processing (so for
        all retrieval steps)

        Note there is a assumption in muses-py that the
        list[FilterIdentifier] is sorted by the starting wavelength
        for each of filters, so we do that here.
        """
        return self._filter_list_dict

    def _next_step_peek(self) -> CurrentStrategyStepDict | None:
        """Return the value of the next step, without actually changing the current step.
        Returns None if we hit done before getting the next step"""
        i = 1
        while (
            self._cur_step_index + i < len(self.current_strategy_list)
            and self.current_strategy_list[self._cur_step_index + i].is_skipped
        ):
            i += 1
        if self._cur_step_index + i < len(self.current_strategy_list):
            return self.current_strategy_list[self._cur_step_index + i]
        return None

    def is_next_bt(self) -> bool:
        """Indicate if the next step is a BT step. This is a bit
        awkward, perhaps we can come up with another interface
        here. But RetrievalStrategyStepBT handles the calculation of the
        brightness temperature step differently depending on if the next
        step is a BT step or not."""
        nstep = self._next_step_peek()
        if nstep is None:
            return False
        return nstep.retrieval_type == RetrievalType("BT")

    @property
    def retrieval_elements(self) -> list[StateElementIdentifier]:
        """The complete list of retrieval elements (so for all
        retrieval steps)"""
        res: set[StateElementIdentifier] = set()
        for cstep in self.current_strategy_list:
            res.update(cstep.retrieval_elements)
        return StateElementIdentifier.sort_identifier(list(res))

    @property
    def error_analysis_interferents(self) -> list[StateElementIdentifier]:
        """Complete list of error analysis interferents (so for all
        retrieval steps)"""
        res: set[StateElementIdentifier] = set()
        for cstep in self.current_strategy_list:
            res.update(cstep.error_analysis_interferents)
        return StateElementIdentifier.sort_identifier(list(res))

    @property
    def instrument_name(self) -> list[InstrumentIdentifier]:
        """Complete list of instrument names (so for all retrieval
        steps)"""
        return InstrumentIdentifier.sort_identifier(self._instrument_name)

    def restart(self) -> None:
        """Set step to the first one."""
        self._cur_step_index = 0
        self._cur_step_count = 0

    def next_step(self, current_state: CurrentState) -> None:
        """Advance to the next step."""
        self._handle_bt(current_state)
        self._cur_step_count += 1
        self._cur_step_index += 1
        while (
            not self.is_done()
            and self.current_strategy_list[self._cur_step_index].is_skipped
        ):
            self._cur_step_index += 1

    def _handle_bt(self, current_state: CurrentState) -> None:
        # Nothing to do if this isn't a BT step, or if we don't have the
        # brightness_temperature_data
        cstep = self.current_strategy_step()
        if (
            cstep is None
            or cstep.retrieval_type != RetrievalType("BT")
            or self._cur_step_count not in current_state.brightness_temperature_data
        ):
            return

        # List of species determine in RetrievalStrategyStepBT
        species_igr = current_state.brightness_temperature_data[self._cur_step_count][
            "species_igr"
        ]
        if species_igr is None:
            return

        # Look at all the bt_ig_refine, and only activate the one that contains species_igr
        found = False
        for cstate in self.current_strategy_list[self._cur_step_index + 1 :]:
            if cstate.retrieval_type != RetrievalType("bt_ig_refine"):
                break
            if ",".join([str(s) for s in cstate.retrieval_elements]) != species_igr:
                cstate.is_skipped = True
            else:
                found = True
                cstate.is_skipped = False
        if not found:
            raise RuntimeError(
                "Specified IG refinement not found (MUST be retrievalType BT_IG_Refine AND species listed in correct order)"
            )

    def is_done(self) -> bool:
        """True if we have reached the last step"""
        return self._cur_step_index >= len(self.current_strategy_list)

    def current_strategy_step(self) -> CurrentStrategyStep | None:
        """Return the CurrentStrategyStep for the current step. Returns None is self.is_done()"""
        if self.is_done():
            return None
        cstate = self.current_strategy_list[self._cur_step_index]
        # Update the step number, which may be different than the initial step number
        # we had before we marked anything skip
        cstate.strategy_step.step_number = self._cur_step_count
        return cstate

    def _parse_state_elements(self, s: str) -> list[StateElementIdentifier]:
        """Small logic used to handle the state element in the file"""
        # We need to handle empty lists (which get expressed as "-", and also make
        # sure the elements get put in the right order.
        r = s.split(",")
        if r[0] == "-":
            return []
        return [StateElementIdentifier(i) for i in r]

    def notify_update_target(self, measurement_id: MeasurementId) -> None:
        super().notify_update_target(measurement_id)
        filter_list_dict_t: dict[
            InstrumentIdentifier, dict[FilterIdentifier, float]
        ] = defaultdict(dict)
        # Fill in measurement specific stuff
        for cstep in self.current_strategy_list:
            cstep.measurement_id = measurement_id
            cstep.is_skipped = False
            p = cstep.retrieval_step_parameters["cost_function_params"]
            p["delta_value"] = int(measurement_id["LMDelta"].split()[0])
            if p["conv_tolerance"] is None:
                p["conv_tolerance"] = [
                    float(measurement_id["ConvTolerance_CostThresh"]),
                    float(measurement_id["ConvTolerance_pThresh"]),
                    float(measurement_id["ConvTolerance_JacThresh"]),
                ]
            sdict = self.spectral_window_handle_set.spectral_window_dict(cstep, None)
            cstep.current_strategy_step_dict["instrument_name"] = list(sdict.keys())
            for k, v in sdict.items():
                for fid, swav in v.filter_name_list():
                    filter_list_dict_t[k][fid] = swav

        # Calculate filter_list_dict,
        self._filter_list_dict = {}
        silist = IdentifierSortByWaveLength()
        for k2, v2 in filter_list_dict_t.items():
            sflist = IdentifierSortByWaveLength()
            for fid, swav in v2.items():
                sflist.add(fid, swav)
            self._filter_list_dict[k2] = cast(
                list[FilterIdentifier], sflist.sorted_identifer()
            )
            silist.add(k2, min(v2.values()))
        self._instrument_name = cast(
            list[InstrumentIdentifier], silist.sorted_identifer()
        )

        # And use to populate the spectral_window_dict
        for cstep in self.current_strategy_list:
            cstep.current_strategy_step_dict["spectral_window_dict"] = (
                self.spectral_window_handle_set.spectral_window_dict(
                    cstep, self._filter_list_dict
                )
            )


class MusesStrategyModifyHandle(MusesStrategyHandle):
    """This is a handle useful for a quick test. We read in an existing Table.asc,
    but then modify the given step_number to have the given retrieval_elements and
    optionally change the max_iter.

    This is limited of course, but handles the bulk of what we have run into in our
    pytests. You can do a similar more complicated handle if useful, or even derive from
    MusesStrategyStepList for more complicated stuff."""

    def __init__(
        self,
        step_number: int,
        retrieval_elements: list[StateElementIdentifier],
        max_iter: int | None = None,
    ):
        super().__init__()
        self.step_number = step_number
        self.retrieval_elements = retrieval_elements
        self.max_iter = max_iter

    def muses_strategy(
        self,
        measurement_id: MeasurementId,
        osp_dir: str | os.PathLike[str] | None = None,
        spectral_window_handle_set: SpectralWindowHandleSet | None = None,
        **kwargs: Any,
    ) -> MusesStrategy | None:
        """Return MusesStrategy if we can process the given
        measurement_id, or None if we can't.
        """
        s = MusesStrategyStepList.create_from_strategy_table_file(
            measurement_id["run_dir"] / "Table.asc", osp_dir, spectral_window_handle_set
        )
        s.current_strategy_list[
            self.step_number
        ].retrieval_elements = self.retrieval_elements
        if self.max_iter is not None:
            s.current_strategy_list[self.step_number].retrieval_step_parameters[
                "max_iter"
            ] = self.max_iter
        return s


def modify_strategy_table(
    rs: RetrievalStrategy,
    step_number: int,
    retrieval_elements: list[StateElementIdentifier],
    max_iter: int | None = None,
) -> None:
    """Simple function to modify the strategy table in a RetrievalStrategy. Meant for
    things like unit tests."""
    h = MusesStrategyModifyHandle(step_number, retrieval_elements, max_iter)
    rs.muses_strategy_handle_set.add_handle(h, 100)


MusesStrategyHandleSet.add_default_handle(MusesStrategyFileHandle())

__all__ = [
    "MusesStrategy",
    "MusesStrategyImp",
    "MusesStrategyHandle",
    "MusesStrategyHandleSet",
    "MusesStrategyModifyHandle",
    "modify_strategy_table",
    "MusesStrategyStepList",
    "CurrentStrategyStep",
    "CurrentStrategyStepDict",
]
