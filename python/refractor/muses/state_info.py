from __future__ import annotations
from .current_state import PropagatedQA
from .sounding_metadata import SoundingMetadata
from .state_element import StateElementHandleSet, StateElement
from .cross_state_element import CrossStateElement, CrossStateElementHandleSet
from .identifier import StateElementIdentifier
import typing
import copy
from collections import UserDict

if typing.TYPE_CHECKING:
    from .muses_observation import MeasurementId, ObservationHandleSet
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .current_state import CurrentState
    from .retrieval_array import RetrievalGridArray, RetrievalGrid2dArray


class CrossStateInfo(UserDict):
    """This is a helper class for StateInfo that handles the CrossStateElement terms.

    This isn't really logically separate from StateInfo, but since we also treat
    StateInfo like a dict like object it is useful to have a dict like object to handle
    the CrossStateElement terms."""

    def __init__(
        self,
        state_info: StateInfo,
        cross_state_element_handle_set: CrossStateElementHandleSet | None = None,
    ) -> None:
        super().__init__()

        if cross_state_element_handle_set is not None:
            self.cross_state_element_handle_set = cross_state_element_handle_set
        else:
            self.cross_state_element_handle_set = copy.deepcopy(
                CrossStateElementHandleSet.default_handle_set()
            )
        self.state_info = state_info

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
        sounding_metadata: SoundingMetadata,
    ) -> None:
        self.cross_state_element_handle_set.notify_update_target(
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            sounding_metadata,
        )
        self.data = {}

    def __missing__(
        self, ky: tuple[StateElementIdentifier, StateElementIdentifier]
    ) -> CrossStateElement:
        self.data[ky] = self.cross_state_element_handle_set.cross_state_element(
            self.state_info[ky[0]], self.state_info[ky[1]]
        )
        return self.data[ky]

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        for celem in self.values():
            celem.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        for celem in self.values():
            celem.notify_start_retrieval(
                current_strategy_step,
                retrieval_config,
            )

    def notify_step_solution(
        self, xsol: RetrievalGridArray, current_state: CurrentState
    ) -> None:
        for celem in self.values():
            celem.notify_step_solution(
                xsol,
                current_state.retrieval_sv_slice(celem.state_element_id_1),
                current_state.retrieval_sv_slice(celem.state_element_id_2),
            )


class StateInfo(UserDict):
    """This class maintains the full state as we perform a retrieval.
    This class is closely tied to CurrentStateStateInfo, it isn't clear
    if we perhaps want to merge theses classes at some point. But right
    now StateInfo focuses on maintaining the state info, and CurrentState
    on making that state info available to other parts of software.

    This is basically a dict like object, where we create StateElement from
    a StateElementHandleSet on first usage.
    """

    def __init__(
        self,
        state_element_handle_set: StateElementHandleSet | None = None,
        cross_state_element_handle_set: CrossStateElementHandleSet | None = None,
        include_old_state_info: bool = True,
    ) -> None:
        super().__init__()
        if state_element_handle_set is not None:
            self.state_element_handle_set = state_element_handle_set
        else:
            self.state_element_handle_set = copy.deepcopy(
                StateElementHandleSet.default_handle_set()
            )
        self._state_element: dict[StateElementIdentifier, StateElement] = {}
        self._cross_state_info = CrossStateInfo(self, cross_state_element_handle_set)
        self._sounding_metadata: SoundingMetadata | None = None
        self.propagated_qa = PropagatedQA()
        self._current_state_old = None
        self._brightness_temperature_data: dict[int, dict[str, float | None]] = {}
        # For now, still integrate in with the old state info stuff from muses-py.
        # This should eventually go away, but this proved useful as both initial
        # scaffolding, and later as a way to compare against the "right" answer.
        # We need just one of the these floating around, so we grab this from
        # the StateElementOldWrapperHandle.
        if include_old_state_info:
            from refractor.old_py_retrieve_wrapper import (
                state_element_old_wrapper_handle,
            )

            self._current_state_old = (
                state_element_old_wrapper_handle._current_state_old
            )

    @property
    def cross_state_element_handle_set(self) -> CrossStateElementHandleSet:
        return self._cross_state_info.cross_state_element_handle_set

    @cross_state_element_handle_set.setter
    def cross_state_element_handle_set(self, val: CrossStateElementHandleSet) -> None:
        self._cross_state_info.cross_state_element_handle_set = val

    @property
    def cross_state_info(self) -> CrossStateInfo:
        return self._cross_state_info

    def cross_constraint_matrix(
        self,
        state_element_id_1: StateElementIdentifier,
        state_element_id_2: StateElementIdentifier,
    ) -> tuple[
        RetrievalGrid2dArray | None,
        RetrievalGrid2dArray | None,
        RetrievalGrid2dArray | None,
    ]:
        """Return a tuple, one for the cross term state_element_id_1 x
        state_element_id_1, one for state_element_id_1 x
        state_element_id_2 and one for state_element_id_2 x
        state_element_id_2.

        Note any or all of these can be "None", which means don't
        change the cross term constraint matrix. In particular for the
        state_element_id_1 x state_element_id_1 and state_element_id_2
        x state_element_id_2 that means use the constraint_matrix
        returned by those StateElement without change.
        """
        if state_element_id_1 > state_element_id_2:
            raise RuntimeError(
                "By convention, state_element_id_1 should always be < state_element_id_2"
            )
        return self.cross_state_info[
            (state_element_id_1, state_element_id_2)
        ].cross_constraint_matrix()

    @property
    def brightness_temperature_data(self) -> dict[int, dict[str, float | None]]:
        return self._brightness_temperature_data

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        if self._sounding_metadata is None:
            raise RuntimeError("Need to call notify_update_target first")
        return self._sounding_metadata

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self._sounding_metadata = SoundingMetadata.create_from_measurement_id(
            measurement_id,
            strategy.instrument_name[0],
            observation_handle_set.observation(
                strategy.instrument_name[0],
                None,
                None,
                None,
                osp_dir=retrieval_config.osp_dir,
            ),
        )
        if self._current_state_old is not None:
            self._current_state_old.notify_update_target(
                measurement_id, retrieval_config, strategy, observation_handle_set
            )
        self.state_element_handle_set.notify_update_target(
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            self._sounding_metadata,
        )
        self._cross_state_info.notify_update_target(
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            self._sounding_metadata,
        )
        self.data = {}
        self.propagated_qa = PropagatedQA()
        self._brightness_temperature_data = {}

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        if self._current_state_old is not None:
            self._current_state_old.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )
        # Make sure we have all the elements we are going to be using in
        # the retrieval, and we notify them about the step
        if current_strategy_step is not None:
            lst = current_strategy_step.retrieval_elements
            lst.extend(current_strategy_step.error_analysis_interferents)
            for sid in lst:
                _ = self[sid]
            for i, sid in enumerate(lst):
                for sid2 in lst[i + 1 :]:
                    _ = self._cross_state_info[(sid, sid2)]
        for selem in self.values():
            selem.notify_start_step(
                current_strategy_step,
                retrieval_config,
                skip_initial_guess_update,
            )
        self._cross_state_info.notify_start_step(
            current_strategy_step, retrieval_config, skip_initial_guess_update
        )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if self._current_state_old is not None:
            self._current_state_old.notify_start_retrieval(
                current_strategy_step, retrieval_config
            )
        for selem in self.values():
            selem.notify_start_retrieval(
                current_strategy_step,
                retrieval_config,
            )
        self._cross_state_info.notify_start_retrieval(
            current_strategy_step, retrieval_config
        )

    def notify_step_solution(
        self, xsol: RetrievalGridArray, current_state: CurrentState
    ) -> None:
        if self._current_state_old is not None:
            self._current_state_old.notify_step_solution(xsol)
        for selem in self.values():
            selem.notify_step_solution(
                xsol, current_state.retrieval_sv_slice(selem.state_element_id)
            )
        self._cross_state_info.notify_step_solution(xsol, current_state)

    def __missing__(self, state_element_id: StateElementIdentifier) -> StateElement:
        self.data[state_element_id] = self.state_element_handle_set.state_element(
            state_element_id
        )
        # Create all the cross terms, so any coupling gets set up
        for sid in self.keys():
            if sid != state_element_id:
                ky = tuple(
                    StateElementIdentifier.sort_identifier([sid, state_element_id])
                )
                _ = self.cross_state_info[ky]
        return self.data[state_element_id]


__all__ = ["StateInfo"]
