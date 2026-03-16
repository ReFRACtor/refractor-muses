from __future__ import annotations
import refractor.framework as rf  # type: ignore
import numpy as np
from typing import Self

class StateMappingUpdateArray(rf.StateMapping):
    '''This is similar to rf.StateMappingAtIndexes, but the indices
    that aren't mapped are actually in the state vector as
    placeholders. This is the way py-retrieve handled the state
    elements that depend on frequency (e.g., EmisState).

    It would probably be cleaner just to not have the unused parts of
    the state vector not there, i.e., just replace this mapping with a
    rf.StateMappingAtIndexes. However it isn't clear yet what parts of
    the code depend on having the full state vector including
    placeholders.

    For now, we duplicate what py-retrieve does. We may come back and clean
    this up.

    TODO - consider removing unused element from the StateVector.
    '''
    def __init__(self, update_array: np.ndarray) -> None:
        super().__init__()
        self.update_array = update_array
        self.full_state: None | rf.ArrayAd_double_1 = None

    def clone(self) -> Self:
        return StateMappingUpdateArray(self.update_array.copy())

    @property
    def name(self) -> str:
        return "update array"

    def mapped_state(self, retrieval_values: rf.ArrayAd_double_1) -> rf.ArrayAd_double_1:
        if self.full_state is None:
            raise RuntimeError("Full state has not yet been initialized")
        if self.full_state.rows != self.update_array.shape[0]:
            raise RuntimeError("full_state and update_array need to be the same size")
        if retrieval_values.rows != self.update_array.shape[0]:
            raise RuntimeError("retrieval_values update_array need to be the same size")
        if self.full_state.number_variable == 0 and retrieval_values.number_variable != 0:
            self.full_state = rf.ArrayAd_double_1(self.full_state.value, np.zeros(retrieval_values.jacobian.shape))
        for i in range(self.update_array.shape[0]):
            if self.update_array[i]:
                self.full_state[i] = retrieval_values[i]
        return self.full_state

    def retrieval_state(self, retrieval_values: rf.ArrayAd_double_1) -> rf.ArrayAd_double_1:
        self.full_state = retrieval_values.copy()
        if self.full_state.rows != self.update_array.shape[0]:
            raise RuntimeError("full_state and update_array need to be the same size")
        return retrieval_values
  
__all__ = ["StateMappingUpdateArray",]
