from __future__ import annotations
from .current_state import PropagatedQA, SoundingMetadata
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
    from .current_state import CurrentState, RetrievalGridArray, RetrievalGrid2dArray


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
        self.propagated_qa = PropagatedQA()
        # Temp, clumsy but this will go away
        for p in sorted(self.state_element_handle_set.handle_set.keys(), reverse=True):
            for h in self.state_element_handle_set.handle_set[p]:
                if hasattr(h, "_current_state_old"):
                    self._current_state_old = h._current_state_old

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
        # Right now, need the old brightness_temperature_data.
        # We can probably straighten this out later, but as we forward stuff
        # to the old current_state_old we need to have the data there
        return self._current_state_old.state_info.brightness_temperature_data

    @property
    def sounding_metadata(self) -> SoundingMetadata:
        # Right now, use the old SoundingMetadata. We'll want to move this over,
        # but that can wait a bit
        return self._current_state_old.sounding_metadata

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        smeta = SoundingMetadata.create_from_measurement_id(
            measurement_id, strategy.instrument_name[0]
        )
        self.state_element_handle_set.notify_update_target(
            measurement_id,
            retrieval_config,
            strategy,
            observation_handle_set,
            smeta,
        )
        self._cross_state_info.notify_update_target(
            measurement_id, retrieval_config, strategy, observation_handle_set, smeta
        )
        self.data = {}
        self.propagated_qa = PropagatedQA()

    def notify_start_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        # TODO, we want to remove this
        self._current_state_old.notify_start_step(
            current_strategy_step,
            retrieval_config,
            skip_initial_guess_update,
        )
        # Since we aren't actually doing the init stuff yet in our
        # new StateElement, make sure everything get created (since
        # this happens on first use)
        for sid in self._current_state_old.full_state_element_id:
            # Skip duplicates we are removing, and poltype that we handling differently now
            if sid not in (
                StateElementIdentifier("emissivity"),
                StateElementIdentifier("cloudEffExt"),
                StateElementIdentifier("nh3type"),
                StateElementIdentifier("ch3ohtype"),
                StateElementIdentifier("hcoohtype"),
            ):
                _ = self[sid]
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
        # TODO, we want to remove this
        self._current_state_old.notify_start_retrieval(
            current_strategy_step, retrieval_config
        )
        # Since we aren't actually doing the init stuff yet in our
        # new StateElement, make sure everything get created (since
        # this happens on first use)
        for sid in self._current_state_old.full_state_element_id:
            # Skip duplicates we are removing
            if sid not in (
                StateElementIdentifier("emissivity"),
                StateElementIdentifier("cloudEffExt"),
                StateElementIdentifier("nh3type"),
                StateElementIdentifier("ch3ohtype"),
                StateElementIdentifier("hcoohtype"),
            ):
                _ = self[sid]
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
        for selem in self.values():
            selem.notify_step_solution(
                xsol, current_state.retrieval_sv_slice(selem.state_element_id)
            )
        self._cross_state_info.notify_step_solution(xsol, current_state)

    def update_with_old(self) -> None:
        """Temporary, we have the StateInfoOld saved but not the new StateInfo in our
        capture tests. We will get to doing StateInfo, but for now use the old data to
        update the new data for the purpose of unit tests."""
        for k, v in self.items():
            try:
                if self._current_state_old.state_value_str(k) is None:
                    v1 = self._current_state_old.state_value(k)
                    v2 = self._current_state_old.state_constraint_vector(k)
                    if k == StateElementIdentifier("PCLOUD"):
                        # Special handling for PCLOUD
                        v1 = v1[0:1]
                        v2 = v2[0:1]
                    v.update_state_element(current_fm=v1, constraint_vector_fm=v2)
            except NotImplementedError:
                # Not all the old elements exist, we just skip any one that doesn't
                pass

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
