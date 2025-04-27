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

class StateElementImplementation(StateElement):
    '''A very common implementation of a StateElement just populates member variables for
    the value, apriori, etc. This class handles this common case, derived classes should
    fill in handling the various notify functions.'''
    # TODO See if we want the RetrievalGrid or ForwardModelGrid here. We'll need to wait
    # until we get to some of the state elements on levels before figuring this out
    def __init__(self, state_element_id: StateElementIdentifier,
                 value : RetrievalGridArray, apriori_value : RetrievalGridArray,
                 apriori_cov : RetrievalGrid2dArray, 
                 state_mapping : rf.StateMapping = rf.StateMappingLinear(),
                 initial_value : RetrievalGridArray | None = None,
                 true_value : RetrievalGridArray | None = None,
                 selem_wrapper: StateElementOldWrapper | None = None,
                 ):
        super().__init__(state_element_id)
        self._value = value
        self._apriori_value = apriori_value
        self._apriori_cov = apriori_cov
        self._state_mapping = state_mapping
        self._step_initial_value = initial_value if initial_value is not None else apriori_value
        self._retrieval_initial_value = self._step_initial_value.copy()
        self._true_value = true_value
        # Temp, until we have tested everything out
        self._sold = selem_wrapper
        if self._sold is not None and hasattr(self._sold, "update_initial_guess"):
            self.update_initial_guess = self._update_initial_guess

    def notify_parameter_update(self, param_subset: np.ndarray) -> None:
        """Called with the subset of parameters for this StateElement
        when the cost function changes."""
        self._value = param_subset

    def _update_initial_guess(self, current_strategy_step: CurrentStrategyStep) -> None:
        if(self._sold is None):
            raise RuntimeError("This shouldn't happen")
        self._sold.update_initial_guess(current_strategy_step)


    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def basis_matrix(self) -> np.ndarray | None:
        """Basis matrix going from retrieval vector to forward model
        vector. Would be nice to replace this with a general
        rf.StateMapping, but for now this is assumed in a lot of
        muses-py code."""
        res = np.eye(self._value.shape[0])
        if(self._sold is not None):
           res2 = self._sold.basis_matrix
           npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    # TODO This can perhaps go away? Replace with a mapping?
    @property
    def map_to_parameter_matrix(self) -> np.ndarray | None:
        """Go the other direction from the basis matrix, going from
        the forward model vector the retrieval vector."""
        res = np.eye(self._value.shape[0])
        if(self._sold is not None):
           res2 = self._sold.map_to_parameter_matrix
           npt.assert_allclose(res, res2, rtol=1e-12)
        return res
    
    @property
    def retrieval_sv_length(self) -> int:
        res = 0
        if(self._sold is not None):
           res2 = self._sold.retrieval_sv_length
           #assert res == res2
        return res2

    @property
    def sys_sv_length(self) -> int:
        res = 0
        if(self._sold is not None):
           res2 = self._sold.sys_sv_length
           #assert res == res2
        return res2

    @property
    def forward_model_sv_length(self) -> int:
        res = 0
        if(self._sold is not None):
           res2 = self._sold.forward_model_sv_length
           #assert res == res2
        return res2

    @property
    def map_type(self) -> str:
        res = "linear"
        if(self._sold is not None):
           res2 = self._sold.map_type
           assert res == res2
        return res
    
    @property
    def value(self) -> RetrievalGridArray:
        """Current value of StateElement"""
        res = self._value
        if self._sold is not None:
            res2 = self._sold.value
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def value_fm(self) -> ForwardModelGridArray:
        res = self._state_mapping.mapped_state(rf.ArrayAd_double_1(self.value)).value
        if self._sold is not None:
            res2 = self._sold.value_fm
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def apriori_value(self) -> RetrievalGridArray:
        res = self._apriori_value
        if self._sold is not None:
            res2 = self._sold.apriori_value
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def apriori_value_fm(self) -> ForwardModelGridArray:
        """Apriori value of StateElement"""
        res = self._state_mapping.mapped_state(rf.ArrayAd_double_1(self.apriori_value)).value
        if self._sold is not None:
            res2 = self._sold.apriori_value_fm
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res
    
    @property
    def constraint_matrix(self) -> RetrievalGrid2dArray:
        """Apriori Covariance"""
        #res = np.linalg.inv(self._apriori_cov)
        # We need to get this straightened out
        res = np.array([[2500.0,]])
        if self._sold is not None:
            res2 = self._sold.constraint_matrix
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def apriori_cov_fm(self) -> ForwardModelGrid2dArray:
        # TODO, get this mapped correctly
        res = self._apriori_cov
        if self._sold is not None:
            res2 = self._sold.apriori_cov_fm
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res
    
    @property
    def retrieval_initial_value(self) -> RetrievalGridArray:
        res = self._retrieval_initial_value
        if self._sold is not None:
            res2 = self._sold.retrieval_initial_value
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def step_initial_value(self) -> RetrievalGridArray:
        res = self._step_initial_value
        if self._sold is not None:
            res2 = self._sold.step_initial_value
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def step_initial_value_fm(self) -> ForwardModelGridArray:
        res = self._state_mapping.mapped_state(rf.ArrayAd_double_1(self.step_initial_value)).value
        if self._sold is not None:
            res2 = self._sold.step_initial_value_fm
            npt.assert_allclose(res, res2, rtol=1e-12)
        return res

    @property
    def true_value(self) -> RetrievalGridArray | None:
        return self._true_value

    @property
    def true_value_fm(self) -> ForwardModelGridArray | None:
        tv = self.true_value
        if(tv is not None):
            return self._state_mapping.mapped_state(rf.ArrayAd_double_1(tv)).value
        return None

    def update_state_element(
        self,
        current: np.ndarray | None = None,
        apriori: np.ndarray | None = None,
        step_initial: np.ndarray | None = None,
        retrieval_initial: np.ndarray | None = None,
        true_value: np.ndarray | None = None,
    ) -> None:
        if(current is not None):
            self._value = current
        if(apriori is not None):
            self._apriori = apriori
        if(step_initial is not None):
            self._step_initial_value = step_initial
        if(retrieval_initial is not None):
            self._retrieval_initial_value = retrieval_initial
        if(true_value is not None):
            self._true_value = true_value

    def update_state(
        self,
        results_list: np.ndarray,
        do_not_update: list[StateElementIdentifier],
        retrieval_config: RetrievalConfiguration | MeasurementId,
        step: int,
    ) -> ForwardModelGridArray | None:
        if(self._sold is not None):
            return self._sold.update_state(
                results_list, do_not_update, retrieval_config, step
            )
        return None

    def notify_new_step(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        error_analysis: ErrorAnalysis,
        retrieval_config: RetrievalConfiguration,
        skip_initial_guess_update: bool = False,
    ) -> None:
        if(self._sold is not None):
            self._sold.notify_new_step(
                current_strategy_step,
                error_analysis,
                retrieval_config,
                skip_initial_guess_update,
            )
        # Default initial guess is whatever we ended up with at the end of the
        # last step
        self._step_initial_value = self._value.copy()

    def notify_start_retrieval(
        self,
        current_strategy_step: CurrentStrategyStep | None,
        retrieval_config: RetrievalConfiguration,
    ) -> None:
        if(self._sold is not None):
            self._sold.notify_start_retrieval(current_strategy_step, retrieval_config)
        # Save the starting point at the start of the retrieval, this is used by the
        # error analysis
        self._retrieval_initial_value = self._step_initial_value.copy()
    
# Start with just this one element, we can hopefully generalize this but work through
# this one first
class StateElementOmiodWavUv(StateElementImplementation):
    def __init__(
        self,
        state_element_id: StateElementIdentifier,
        measurement_id: MeasurementId,
        selem_wrapper: StateElementOldWrapper,
    ):
        # Fill these in
        value = np.array([0.0,])
        apriori = np.array([0.0,])
        apriori_cov = np.array([[0.00039999998989515007]])
        super().__init__(state_element_id, value, apriori, apriori_cov,
                         selem_wrapper = selem_wrapper)

    # Couple of things needed to work with StateElementOldWrapper. These can
    # perhaps go away once we have all the StateElementOldWrapper pulled out, but
    # for now we need this
    @property
    def retrieval_slice(self) -> slice | None:
        # I think this can go away
        breakpoint()
        if(self._sold is not None):
            return self._sold.retrieval_slice
        return None

    @property
    def fm_slice(self) -> slice | None:
        # I think this can go away
        breakpoint()
        if(self._sold is not None):
            return self._sold.retrieval_slice
        return None

    @property
    def _old_selem(self) -> StateElementOld:
        # I think this can go away
        breakpoint()
        if(self._sold is None):
            raise RuntimeError("This should not happen")
        return self._sold._old_selem


class StateElementScaffoldHandle(StateElementHandle):
    def __init__(
        self, cls: type, sid: StateElementIdentifier, hold: StateElementOldWrapperHandle
    ) -> None:
        self.obs_cls = cls
        self.sid = sid
        self.hold = hold
        self.measurement_id : MeasurementId | None = None

    def notify_update_target(
        self,
        measurement_id: MeasurementId,
        retrieval_config: RetrievalConfiguration,
        strategy: MusesStrategy,
        observation_handle_set: ObservationHandleSet,
    ) -> None:
        self.measurement_id = measurement_id

    def state_element(
        self, state_element_id: StateElementIdentifier
    ) -> StateElement | None:
        if state_element_id != self.sid:
            return None
        if self.measurement_id is None:
            raise RuntimeError("Need to call notify_update_target first")
        sold = self.hold.state_element(state_element_id)
        return self.obs_cls(state_element_id, self.measurement_id, sold)


__all__ = [
    "StateElementScaffoldHandle",
    "StateElementOmiodWavUv",
]
