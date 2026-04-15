from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .current_state import PropagatedQA
from .sounding_metadata import SoundingMetadata
from .state_element import StateElement
from .cross_state_element import CrossStateElement
from .muses_strategy_context import MusesStrategyContext, MusesStrategyContextMixin
from .identifier import StateElementIdentifier
import typing
from collections import UserDict

if typing.TYPE_CHECKING:
    from .muses_strategy import CurrentStrategyStep
    from .creator_dict import CreatorDict
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
        creator_dict: CreatorDict,
    ) -> None:
        super().__init__()
        self.state_info = state_info
        self._creator_dict = creator_dict

    def notify_update_strategy_context(
        self, strategy_context: MusesStrategyContext
    ) -> None:
        self.data = {}

    def __missing__(
        self, ky: tuple[StateElementIdentifier, StateElementIdentifier]
    ) -> CrossStateElement:
        self.data[ky] = self._creator_dict[CrossStateElement].cross_state_element(
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


class StateInfo(UserDict, MusesStrategyContextMixin):
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
        creator_dict: CreatorDict,
        include_old_state_info: bool = False,
    ) -> None:
        UserDict.__init__(self)
        MusesStrategyContextMixin.__init__(self, creator_dict.strategy_context)
        self._creator_dict = creator_dict
        self._state_element: dict[StateElementIdentifier, StateElement] = {}
        self._cross_state_info = CrossStateInfo(self, creator_dict)
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
                state_element_old_wrapper_handle._current_state_old  # noqa: SLF001
            )
        self.strategy_context.add_observer(self)

    def __hash__(self) -> int:
        """Simple hash, just so this can be an observer. Dict aren't hashable in
        general, but we just want to identify if the same object has been added."""
        return id(self)

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
            raise RuntimeError("Need to call notify_update_strategy_context first")
        return self._sounding_metadata

    def notify_update_strategy_context(
        self, strategy_context: MusesStrategyContext
    ) -> None:
        if self.has_measurement_id:
            self._sounding_metadata = SoundingMetadata.create_from_measurement_id(
                self.measurement_id,
                self.strategy.instrument_name[0],
                self._creator_dict[rf.Observation].observation(
                    self.strategy.instrument_name[0],
                    None,
                    None,
                    None,
                ),
                self.retrieval_config.input_file_helper,
            )
        if self._current_state_old is not None:
            # We previously called notify_update_strategy_context
            # "notify_update_target" when we had a single target in
            # the run directory. We changed the name to reflect
            # support for a stac_catalog (which has multiple targets).
            # But is isn't worth updating the old_py_retrieve_wrapper
            # code, so just forward this to the old name if we are
            # using the old state info
            self._current_state_old.notify_update_target(
                self.measurement_id,
                self.retrieval_config,
                self.strategy,
                self._creator_dict[rf.Observation],
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
        if (
            current_strategy_step is not None
            and hasattr(current_strategy_step, "retrieval_elements")
            and hasattr(current_strategy_step, "error_analysis_interferents")
        ):
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
        self.data[state_element_id] = self._creator_dict[StateElement].state_element(
            state_element_id, self._creator_dict[rf.Observation], self
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
