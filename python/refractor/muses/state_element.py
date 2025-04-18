# Might end up breaking this file up, for now have all the stuff here
from __future__ import annotations
import refractor.framework as rf  # type: ignore
from .state_info import StateElement, StateElementHandle
import numpy as np
import numpy.testing as npt
import typing
from loguru import logger

if typing.TYPE_CHECKING:
    from .state_element_old_wrapper import (
        StateElementOldWrapper,
        StateElementOldWrapperHandle,
    )
    from .identifier import StateElementIdentifier
    from .muses_observation import ObservationHandleSet, MeasurementId
    from .muses_strategy import MusesStrategy, CurrentStrategyStep
    from .retrieval_configuration import RetrievalConfiguration
    from .error_analysis import ErrorAnalysis
    from refractor.old_py_retrieve_wrapper import StateElementOld  # type: ignore


# A couple of aliases, just so we can clearly mark what grid data is on
RetrievalGridArray = np.ndarray
ForwardModelGridArray = np.ndarray
RetrievalGrid2dArray = np.ndarray
ForwardModelGrid2dArray = np.ndarray


# Sample to start out using
class StateElementTemplate(StateElement):
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        selem_wrapper: StateElementOldWrapper,
    ):
        super().__init__(state_element_id)
        self._sold = selem_wrapper
        if hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess

    # Couple of things needed to work with StateElementOldWrapper. These can
    # perhaps go away once we have all the StateElementOldWrapper pulled out, but
    # for now we need this
    @property
    def retrieval_slice(self) -> slice | None:
        return self._sold.retrieval_slice

    @property
    def fm_slice(self) -> slice | None:
        return self._sold.retrieval_slice

    @property
    def _old_selem(self) -> StateElementOld:
        return self._sold._old_selem

    def _update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        return self._sold.update_initial_guess(current_strategy_step)

    # Rest of this is stuff we need for StateElement

    @property
    def basis_matrix(self) -> np.ndarray | None:
        return self._sold.basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        return self._sold.map_to_parameter_matrix

    @property
    def retrieval_sv_length(self) -> int:
        return self._sold.retrieval_sv_length

    @property
    def sys_sv_length(self) -> int:
        return self._sold.sys_sv_length

    @property
    def forward_model_sv_length(self) -> int:
        return self._sold.forward_model_sv_length

    @property
    def map_type(self) -> str:
        return self._sold.map_type

    @property
    def value(self) -> RetrievalGridArray:
        return self._sold.value

    @property
    def value_fm(self) -> ForwardModelGridArray:
        return self._sold.value_fm

    @property
    def apriori_value(self) -> RetrievalGridArray:
        return self._sold.apriori_value

    @property
    def apriori_value_fm(self) -> ForwardModelGrid2dArray:
        return self._sold.apriori_value_fm

    @property
    def apriori_cov(self) -> RetrievalGrid2dArray:
        return self._sold.apriori_cov

    @property
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        return self._sold.apriori_cov_fm

    @property
    def retrieval_initial_value(self) -> RetrievalGridArray:
        return self._sold.retrieval_initial_value

    @property
    def step_initial_value(self) -> RetrievalGridArray:
        return self._sold.step_initial_value

    @property
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        return self._sold.step_initial_value_fm

    def update_state_element(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        self._sold.update_state_element(
            current, apriori, step_initial, retrieval_initial, true_value
        )

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> ForwardModelGridArray | None:
        return self._sold.update_state(
            results_list, do_not_update, retrieval_config, step
        )

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        self._sold.notify_new_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        self._sold.notify_start_retrieval(current_strategy_step, retrieval_config)


# Start with just this one element, we can hopefully generalize this but work through
# this one first
class StateElementOmiodWavUv(StateElement):
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        selem_wrapper: StateElementOldWrapper,
    ):
        super().__init__(state_element_id)
        self._sold = selem_wrapper
        if hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess
        self._v = self._sold.value

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        """Called with the subset of parameters for this StateElement
        when the cost function changes."""
        self._v = param_subset
        logger.debug(f"notify_parameter_update update to {self._v}")

    # Couple of things needed to work with StateElementOldWrapper. These can
    # perhaps go away once we have all the StateElementOldWrapper pulled out, but
    # for now we need this
    @property
    def retrieval_slice(self) -> slice | None:
        return self._sold.retrieval_slice

    @property
    def fm_slice(self) -> slice | None:
        return self._sold.retrieval_slice

    @property
    def _old_selem(self) -> StateElementOld:
        return self._sold._old_selem

    def _update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        return self._sold.update_initial_guess(current_strategy_step)

    # Rest of this is stuff we need for StateElement

    @property
    def basis_matrix(self) -> np.ndarray | None:
        return self._sold.basis_matrix

    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        return self._sold.map_to_parameter_matrix

    @property
    def retrieval_sv_length(self) -> int:
        return self._sold.retrieval_sv_length

    @property
    def sys_sv_length(self) -> int:
        return self._sold.sys_sv_length

    @property
    def forward_model_sv_length(self) -> int:
        return self._sold.forward_model_sv_length

    @property
    def map_type(self) -> str:
        return self._sold.map_type

    @property
    def value(self) -> RetrievalGridArray:
        res = self._v
        if True:
            res2 = self._sold.value
            npt.assert_allclose(res, res2)
        return res2

    @property
    def value_fm(self) -> ForwardModelGridArray:
        # For other classes, we'll need to handle use of basis matrix.
        res = self._v
        if True:
            res2 = self._sold.value_fm
            npt.assert_allclose(res, res2)
        return res2

    @property
    def apriori_value(self) -> RetrievalGridArray:
        return self._sold.apriori_value

    @property
    def apriori_value_fm(self) -> ForwardModelGrid2dArray:
        return self._sold.apriori_value_fm

    @property
    def apriori_cov(self) -> RetrievalGrid2dArray:
        return self._sold.apriori_cov

    @property
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        return self._sold.apriori_cov_fm

    @property
    def retrieval_initial_value(self) -> RetrievalGridArray:
        return self._sold.retrieval_initial_value

    @property
    def step_initial_value(self) -> RetrievalGridArray:
        return self._sold.step_initial_value

    @property
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        return self._sold.step_initial_value_fm

    def update_state_element(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        self._sold.update_state_element(
            current, apriori, step_initial, retrieval_initial, true_value
        )

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> ForwardModelGridArray | None:
        return self._sold.update_state(
            results_list, do_not_update, retrieval_config, step
        )

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        self._sold.notify_new_step(
            current_strategy_step,
            error_analysis,
            retrieval_config,
            skip_initial_guess_update,
        )

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        self._sold.notify_start_retrieval(current_strategy_step, retrieval_config)


class StateElementScaffoldHandle(StateElementHandle):
    def __init__(
        self, cls: type, sid: StateElementIdentifier, hold: StateElementOldWrapperHandle
    ) -> None:
        self.obs_cls = cls
        self.sid = sid
        self.hold = hold

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        pass

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        sold = self.hold.state_element(state_element_id)
        return self.obs_cls(state_element_id, sold)


__all__ = [
    "StateElementScaffoldHandle",
    "StateElementTemplate",
    "StateElementOmiodWavUv",
]
